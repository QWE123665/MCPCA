from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union, List

import numpy as np

from .MCPCA_core import cov_tensor, choose_rank, MCPCA_decompose


ArrayLike = Union[np.ndarray]


@dataclass
class MCPCAResult:
    components: np.ndarray  # A: (p_features, r)
    loadings: np.ndarray    # B: (k_contexts, r)
    rank: int


class MCPCA:
    """
    sklearn-style wrapper for MCPCA.

    Parameters
    ----------
    n_components : int or None
        Rank r. If None, choose_rank is used.
    rank_range : sequence of int or None
        Candidate ranks for choose_rank when n_components is None.
    n_seed_pairs : int
        Number of random seed pairs used by choose_rank.
    cos_sim_threshold : float
        Threshold for choose_rank.
    random_state : int or None
        Seed for reproducible initialization in spm_21sym.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        rank_range: Optional[Sequence[int]] = None,
        n_seed_pairs: int = 5,
        cos_sim_threshold: float = 0.9,
        random_state: Optional[int] = None
    ) -> None:
        self.n_components = n_components
        self.rank_range = rank_range
        self.n_seed_pairs = n_seed_pairs
        self.cos_sim_threshold = cos_sim_threshold
        self.random_state = random_state

        # learned attributes (sklearn-style)
        self.components_: Optional[np.ndarray] = None
        self.loadings_: Optional[np.ndarray] = None
        self.means_: Optional[List[np.ndarray]] = None
        self.n_components_: Optional[int] = None
        self.n_features_in_: Optional[int] = None
        self.n_contexts_in_: Optional[int] = None
        self.T: Optional[np.ndarray] = None
        self.r_: Optional[int] = None
    # ----------------------------
    # Internal helpers
    # ----------------------------
    def _validate_df_list(self, X_list: Sequence[np.ndarray]) -> List[np.ndarray]:
        """Validate and normalize a list/tuple of context datasets.

        Accepts numpy arrays or list-of-lists and returns a list of 2D
        numpy arrays with a consistent number of features.
        """
        if not isinstance(X_list, (list, tuple)) or len(X_list) == 0:
            raise ValueError("X must be a non-empty list/tuple of arrays, one per context.")

        p = None
        out: List[np.ndarray] = []
        for i, X in enumerate(X_list):
            # Coerce list-of-lists or other array-likes to np.ndarray
            if isinstance(X, np.ndarray):
                X_arr = X
            else:
                X_arr = np.asarray(X)

            if X_arr.ndim != 2:
                raise ValueError(
                    f"X_list[{i}] must be 2D (n_samples, n_features). Got shape {X_arr.shape}."
                )

            if p is None:
                p = X_arr.shape[1]
            elif X_arr.shape[1] != p:
                raise ValueError(
                    "All contexts must share the same number of features. "
                    f"Got X_list[{i}].shape[1]={X_arr.shape[1]} vs {p}."
                )

            out.append(X_arr)

        return out

    def _set_random_state(self) -> None:
        if self.random_state is not None:
            np.random.seed(self.random_state)

    # ----------------------------
    # Public API
    # ----------------------------
    def fit(self, X_list: Sequence[np.ndarray], plot_A = False, plot_B = False, store_tensor = False, scree_plot = False) -> "MCPCA":
        """
        Fit MCPCA on a list of contexts (datasets), each of shape (n_i, p).

        Stores:
        - components_ : A (p, r)
        - loadings_   : B (k, r)
        """
        X_list = self._validate_df_list(X_list)
        self._set_random_state()
        self.means_ = [X.mean(axis=0, keepdims=True) for X in X_list]

        T = cov_tensor(X_list)  # (p, p, k)
        if store_tensor:
            self.T = T
        p, _, k = T.shape

        # choose rank if needed
        if self.n_components is None:
            if self.rank_range is None:
                raise ValueError("rank_range must be provided when n_components is None.")
            r = choose_rank(
                T,
                rank_range=list(self.rank_range),
                n_seed_pairs=self.n_seed_pairs,
                cos_sim_threshold=self.cos_sim_threshold,
                stats=False,
                scree_plot = scree_plot
            )
        else:
            r = int(self.n_components)
        
        self.r_ = r
        # run MCPCA (your function already does NNLS for B)
        A, B = MCPCA_decompose(T, r, plot_A=plot_A, plot_B=plot_B)
        
        self.components_ = A
        self.loadings_ = B
        self.n_components_ = r
        self.n_features_in_ = p
        self.n_contexts_in_ = k
        return self
    


    def transform(
        self,
        X_list: Sequence[np.ndarray],
        center: bool = True,
        return_list: bool = True,
    ):
        """
        Project each dataset onto the learned components A, producing sample-level scores.

        For each context dataset X (n x p), returns Z = X @ pinv(A^T) (n x r),
        i.e. the least-squares coefficients in X ≈ Z A^T.

        Parameters
        ----------
        X_list : list of arrays, each (n_i, p)
        center : bool
            If True, subtract the per-context mean used during fit (if available),
            otherwise subtract the mean of the given X.
        return_list : bool
            If True, return a list [Z_1, ..., Z_k]. 
            If False, return a concatenated array Z and an array of context labels.

        Returns
        -------
        Zs : list of arrays or (Z, ctx)
        """
        if self.components_ is None:
            raise RuntimeError("Call fit() before transform().")
        X_list = self._validate_df_list(X_list)

        A = self.components_              # (p, r)
        pinv_AT = np.linalg.pinv(A.T)     # (p, r)

        Z_list: List[np.ndarray] = []
        ctx_list: List[np.ndarray] = []

        for i, X in enumerate(X_list):
            if X.ndim != 2:
                raise ValueError(f"X_list[{i}] must be 2D, got shape {X.shape}.")
            if X.shape[1] != A.shape[0]:
                raise ValueError(
                    f"X_list[{i}] has p={X.shape[1]} features, but A has p={A.shape[0]}."
                )

            Xc = X
            if center:
                # If you stored means_ in fit, use them; otherwise center by current mean
                if hasattr(self, "means_") and self.means_ is not None and i < len(self.means_):
                    Xc = X - self.means_[i]
                else:
                    Xc = X - X.mean(axis=0, keepdims=True)

            Z = Xc @ pinv_AT  # (n_i, r)
            Z_list.append(Z)

            if not return_list:
                ctx_list.append(np.full((X.shape[0],), i, dtype=int))

        if return_list:
            return Z_list
        else:
            Z_concat = np.vstack(Z_list)          # (sum n_i, r)
            ctx_concat = np.concatenate(ctx_list)  # (sum n_i,)
            return Z_concat, ctx_concat
    
    
    def fit_transform(self, X_list: Sequence[np.ndarray], center: bool = True, return_list: bool = True):
        self.fit(X_list)
        return self.transform(X_list, center=center, return_list=return_list)

    def get_params(self, deep: bool = True):
        # sklearn compatibility (optional)
        return {
            "n_components": self.n_components,
            "rank_range": self.rank_range,
            "n_seed_pairs": self.n_seed_pairs,
            "cos_sim_threshold": self.cos_sim_threshold,
            "random_state": self.random_state,
            "nonnegative_loadings": self.nonnegative_loadings,
        }

    def set_params(self, **params):
        # sklearn compatibility (optional)
        for k, v in params.items():
            setattr(self, k, v)
        return self
