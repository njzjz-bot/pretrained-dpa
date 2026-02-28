"""DeepMD backend plugin that supports ``*.pretrained`` model aliases."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from deepmd.backend.backend import Backend  # type: ignore[import-not-found]
from deepmd.infer.deep_eval import DeepEvalBackend  # type: ignore[import-not-found]

from .cli import resolve_model_path

if TYPE_CHECKING:
    from argparse import Namespace

    from deepmd.infer.deep_eval import DeepEval  # type: ignore[import-not-found]
    from deepmd.utils.neighbor_stat import (
        NeighborStat,  # type: ignore[import-not-found]
    )


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


class _PretrainedDeepEvalBackend(DeepEvalBackend):
    """Resolve alias then delegate to the actual backend DeepEval implementation."""

    def __init__(
        self,
        model_file: str,
        output_def: object,
        *args: object,
        auto_batch_size: object = True,
        neighbor_list: object | None = None,
        **kwargs: object,
    ) -> None:
        model_name = parse_pretrained_alias(model_file)
        resolved = str(resolve_model_path(model_name))

        self._backend = DeepEvalBackend(
            resolved,
            output_def,
            *args,
            auto_batch_size=auto_batch_size,
            neighbor_list=neighbor_list,
            **kwargs,
        )

    def eval(
        self,
        coords: object,
        cells: object,
        atom_types: object,
        atomic: bool = False,
        fparam: object | None = None,
        aparam: object | None = None,
        **kwargs: object,
    ) -> object:
        """Delegate eval to resolved backend instance."""
        return self._backend.eval(
            coords,
            cells,
            atom_types,
            atomic,
            fparam=fparam,
            aparam=aparam,
            **kwargs,
        )

    def get_rcut(self) -> float:
        return self._backend.get_rcut()

    def get_ntypes(self) -> int:
        return self._backend.get_ntypes()

    def get_type_map(self) -> list[str]:
        return self._backend.get_type_map()

    def get_dim_fparam(self) -> int:
        return self._backend.get_dim_fparam()

    def has_default_fparam(self) -> bool:
        return self._backend.has_default_fparam()

    def get_dim_aparam(self) -> int:
        return self._backend.get_dim_aparam()

    @property
    def model_type(self) -> type[DeepEval]:
        return self._backend.model_type

    def get_sel_type(self) -> list[int]:
        return self._backend.get_sel_type()

    def get_numb_dos(self) -> int:
        return self._backend.get_numb_dos()

    def get_has_efield(self) -> bool:
        return self._backend.get_has_efield()

    def get_has_spin(self) -> bool:
        return self._backend.get_has_spin()

    def get_has_hessian(self) -> bool:
        return self._backend.get_has_hessian()

    def get_var_name(self) -> str:
        return self._backend.get_var_name()

    def get_ntypes_spin(self) -> int:
        return self._backend.get_ntypes_spin()

    def get_model(self) -> Any:
        return self._backend.get_model()


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
        return _PretrainedDeepEvalBackend

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
