"""Concrete DeepEval backend adapter for pretrained alias routing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from deepmd.infer.deep_eval import DeepEvalBackend

from .cli import resolve_model_path
from .deepmd_backend import parse_pretrained_alias

if TYPE_CHECKING:
    from deepmd.infer.deep_eval import DeepEval


class PretrainedDeepEvalBackend(DeepEvalBackend):
    """Resolve alias then delegate to the actual backend implementation."""

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
