# MCPCA (Multi-Context Principal Component Analysis)

This repository implements **Multi-Context Principal Component Analysis (MCPCA)**, a method for analyzing multiple datasets (“contexts”) that share the same features. MCPCA decomposes context-dependent covariance structure into components that are shared across subsets of contexts.


## Installation

From the repository root, install the package into your current Python environment (Python >= 3.10):

```bash
pip install .
```

This installs the package as mcpca so it can be imported directly.

For development (recommended):

```bash
pip install -e .
```
---


## What MCPCA learns

Given $k$ datasets sharing $p$ features, MCPCA learns the following objects.

---

#### Shared components (MCPCs): matrix `A`

The matrix
$$
A \in \mathbb{R}^{p \times r}
$$
contains the **multi-context principal components (MCPCs)**. Each column of `A`
is a feature-space direction shared across all contexts.

Unlike standard PCA:
- components are learned jointly across contexts
- components are not required to be orthogonal

#### Context loadings: matrix `B`

The matrix
$$
B \in \mathbb{R}^{k \times r}
$$
encodes how strongly each shared component appears in each context

Interpretation:
- rows correspond to contexts
- columns correspond to shared components
- `B[i, j]` measures the strength of component `j` in context `i`


After fitting the model, the components are available as:

```python
A = model.components_  # shared components (A)
B = model.loadings_    # context loadings (B)
```

## Basic usage

The main sklearn-style entry point is `mcpca.MCPCA`. You pass in a list of context datasets, each a numpy array, fit the model, then optionally transform the data and visualize context loadings.

A synthetic data illustration is available at tests/test.py .

### 1. Prepare your data

The data in each context should be a 2D NumPy array of shape $(N_i, p)$ with the **same number of features** $p$ across contexts:

```python
import numpy as np
from mcpca import MCPCA

# Example: two contexts
X_list = [X1, X2, X3]
```

### 2. Fit the model and get A, B

```python
model = MCPCA(
	n_components=None,      # or an integer rank r
	rank_range=[1, 2, 3, 4],
	n_seed_pairs=5,
	cos_sim_threshold=0.9,
	random_state=0,
)

model.fit(X_list)

# Shared components A (multi-context PCs)
A = model.components_   # shape (p_features, r)

# Context loadings B
B = model.loadings_     # shape (k_contexts, r)
```

Here:

- `A` contains the shared directions (MCPCs) across contexts.
- `B` tells you how strongly each context loads onto each component.

If `n_components=None`, the rank is chosen automatically from `rank_range` via `choose_rank` in `MCPCA_core`.

### 3. Transform data (scores per sample)

Use `transform` to project each sample in each context into the MCPC space:

```python
Z_list = model.transform(X_list)

Z1, Z2, Z3 = Z_list  # each has shape (N_i, r)
```

Setting `return_list=False` instead concatenates all Z_i vertically and returns the concatenated array and context indices:

```python
Z_all, ctx = model.transform(X_list, return_list=False)
```

You can also call `fit_transform` in one step:

```python
Z_list = model.fit_transform(X_list)
```

### 4. Visualization

To visualize context loadings during fitting:
```python
model.fit(X_list, plot_B=True)
```

When `plot_B=True`, the algorithm will produce a heatmap of the context loadings matrix `B`, which helps visualize how different contexts load onto the shared components.

If you prefer, you can also take `B = model.loadings_` and create custom plots using Matplotlib or Seaborn.

Similarly, setting `plot_A = True` visualizes the shared component matrix `A`. For custom plots, access `A` via model.components_ .

### 5. Covariance matrix mode


MCPCA can also be applied directly to covariance matrices rather than raw data. This mode is useful when only second-order statistics are available or when raw data cannot be stored or shared.

Suppose `M_list` is a list of covariance matrices, one per context, each of
shape `(p, p)`:

```python
import numpy as np
from mcpca import MCPCA_decompose

T = np.stack(M_list, axis=2)  # shape (p, p, k)
A, B = MCPCA_decompose(T, r) 
```

Automatic rank selection from a specific rank range is also supported.

```python
T = np.stack(M_list,axis = 2)
rank_range = [1,2,3,4]
r = choose_rank(T, rank_range = rank_range)
A,B = mcpca.MCPCA_decompose(T,r)
```



## Citation

If you use MCPCA in your research, please cite the associated paper:

```bibtex
@article{mcpca2025,
  title={Multi-Context Principal Component Analysis},
  author={Kexin Wang, Salil Bhate, Jo\~ao M. Pereira, Joe Kileel, Matylda Figlerowicz, Anna Seigal},
  journal = {},
  year={2025}
}
