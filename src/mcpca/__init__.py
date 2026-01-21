"""mcpca package aggregator.

Provides a single import point that exposes existing modules:
- MCPCA_core
- SPM21
- myutils
- compiler_options

"""

from importlib import import_module

try:
    # Prefer reading version from package metadata when installed
    from importlib.metadata import version, PackageNotFoundError  # type: ignore
    try:
        __version__ = version("mcpca")
    except PackageNotFoundError:
        __version__ = "0.1.0"
except Exception:
    __version__ = "0.1.0"

# Re-export existing top-level modules as submodules of this package
MCPCA_module = import_module(".MCPCA_core",__name__)
SPM21 = import_module(".SPM21",__name__)
_MCPCA = import_module(".MCPCA_class",__name__)
MCPCA = getattr(_MCPCA, "MCPCA")
myutils = import_module(".myutils",__name__)
compiler_options = import_module(".compiler_options",__name__)

# Convenience top-level re-exports for common functions
# Users can call: mcpca.choose_rank(...) and mcpca.MCPCA_decompose(...)
choose_rank = getattr(MCPCA_module, "choose_rank")
cov_tensor = getattr(MCPCA_module, "cov_tensor")
similarity_measures = getattr(MCPCA_module,"similarity_measures")
MCPCA_decompose = getattr(MCPCA_module, "MCPCA_decompose")

__all__ = [
    "MCPCA_decompose",  # function
    "MCPCA",  # class
    "choose_rank",
    "cov_tensor",
    "similarity_measures",
    "SPM21",
    "myutils",
    "compiler_options",
    "__version__",
]
