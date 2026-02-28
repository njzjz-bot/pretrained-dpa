"""A Pretrained DPA (please revise this docstring)."""

from __future__ import annotations

from ._version import __version__
from .deepmd_backend import PretrainedBackend

__email__ = "jinzhe.zeng@ustc.edu.cn"

__all__ = [
    "PretrainedBackend",
    "__version__",
]
