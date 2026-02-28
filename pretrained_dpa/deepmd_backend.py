"""DeepMD backend plugin entrypoint for ``*.pretrained`` model aliases."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from deepmd.backend.backend import Backend

if TYPE_CHECKING:
    from argparse import Namespace

    from deepmd.infer.deep_eval import DeepEvalBackend
    from deepmd.utils.neighbor_stat import NeighborStat


def parse_pretrained_alias(model_file: str) -> str:
    """Extract model name from ``*.pretrained`` alias string."""
    alias = Path(model_file).name
    suffix = ".pretrained"
    if not alias.endswith(suffix):
        msg = f"Invalid pretrained alias: {model_file}"
        raise ValueError(msg)

    model_name = alias[: -len(suffix)]
    if not model_name:
        msg = f"Invalid pretrained alias: {model_file}"
        raise ValueError(msg)

    return model_name


@Backend.register("pretrained")
class PretrainedBackend(Backend):
    """Backend that resolves pretrained aliases and delegates to actual backend."""

    name = "Pretrained alias backend"
    features: ClassVar[Backend.Feature] = Backend.Feature.DEEP_EVAL
    suffixes: ClassVar[list[str]] = [".pretrained"]

    def is_available(self) -> bool:
        return True

    @property
    def entry_point_hook(self) -> Callable[[Namespace], None]:
        msg = "Entry point is not supported by pretrained backend"
        raise NotImplementedError(msg)

    @property
    def deep_eval(self) -> type[DeepEvalBackend]:
        module = import_module("pretrained_dpa.deepmd_backend_impl")
        return module.PretrainedDeepEvalBackend

    @property
    def neighbor_stat(self) -> type[NeighborStat]:
        msg = "Neighbor stat is not supported by pretrained backend"
        raise NotImplementedError(msg)

    @property
    def serialize_hook(self) -> Callable[[str], dict]:
        msg = "Serialize hook is not supported by pretrained backend"
        raise NotImplementedError(msg)

    @property
    def deserialize_hook(self) -> Callable[[str, dict], None]:
        msg = "Deserialize hook is not supported by pretrained backend"
        raise NotImplementedError(msg)
