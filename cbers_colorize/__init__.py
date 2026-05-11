from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("cbers-colorization")
except PackageNotFoundError:
    # fallback para ambiente "editable" / execução direta sem pacote instalado
    __version__ = "0.1.0"

__all__ = ["__version__"]