"""DeepMD backend plugin that supports ``*.pretrained`` model aliases."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from .cli import resolve_model_path

try:
    from deepmd.backend.backend import Backend  # type: ignore[import-not-found]
    from deepmd.infer.deep_eval import DeepEvalBackend  # type: ignore[import-not-found]

    _HAS_DEEPMD = True
except ModuleNotFoundError:  # pragma: no cover
    _HAS_DEEPMD = False

    class DeepEvalBackend:  # type: ignore[no-redef]
        """Fallback placeholder when deepmd-kit is not installed."""

    class _DummyFeature:
        DEEP_EVAL = 0

    class Backend:  # type: ignore[no-redef]
        """Fallback placeholder when deepmd-kit is not installed."""

        Feature = _DummyFeature

        @staticmethod
        def register(_key: str) -> object:
            """No-op register decorator for environments without deepmd-kit."""

            def _decorator(cls: type) -> type:
                return cls

            return _decorator

        @staticmethod
        def detect_backend_by_model(_filename: str) -> type:
            """Raise a clear error when deepmd-kit is unavailable."""
            msg = "deepmd-kit is required to use pretrained backend"
            raise ModuleNotFoundError(msg)


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

        backend_cls = Backend.detect_backend_by_model(resolved)
        self._backend = backend_cls().deep_eval(
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
    def model_type(self) -> object:
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

    def get_model(self) -> object:
        return self._backend.get_model()


@Backend.register("pretrained")
class PretrainedBackend(Backend):
    """Backend that resolves pretrained aliases and delegates to actual backend."""

    name = "Pretrained alias backend"
    features: ClassVar[object] = Backend.Feature.DEEP_EVAL
    suffixes: ClassVar[list[str]] = [".pretrained"]

    def is_available(self) -> bool:
        return _HAS_DEEPMD

    @property
    def entry_point_hook(self) -> object:
        msg = "Entry point is not supported by pretrained backend"
        raise NotImplementedError(msg)

    @property
    def deep_eval(self) -> type[DeepEvalBackend]:
        return _PretrainedDeepEvalBackend

    @property
    def neighbor_stat(self) -> object:
        msg = "Neighbor stat is not supported by pretrained backend"
        raise NotImplementedError(msg)

    @property
    def serialize_hook(self) -> object:
        msg = "Serialize hook is not supported by pretrained backend"
        raise NotImplementedError(msg)

    @property
    def deserialize_hook(self) -> object:
        msg = "Deserialize hook is not supported by pretrained backend"
        raise NotImplementedError(msg)
