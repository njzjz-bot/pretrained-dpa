"""Tests for DeepMD pretrained alias backend integration helpers."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import ClassVar

import pytest


def _install_fake_deepmd_modules() -> None:
    """Install minimal fake deepmd modules for unit tests."""

    class FakeBackendBase:
        """Minimal backend registry to satisfy plugin decorator usage."""

        class Feature:
            DEEP_EVAL = 1

        _registry: ClassVar[dict[str, type]] = {}

        @classmethod
        def register(cls, key: str):
            def decorator(target_cls: type) -> type:
                cls._registry[key] = target_cls
                return target_cls

            return decorator

    class FakeDeepEvalBackendBase:
        """Minimal DeepEvalBackend type for subclassing in tests."""

    deepmd_mod = types.ModuleType("deepmd")
    backend_pkg = types.ModuleType("deepmd.backend")
    backend_mod = types.ModuleType("deepmd.backend.backend")
    infer_pkg = types.ModuleType("deepmd.infer")
    infer_mod = types.ModuleType("deepmd.infer.deep_eval")
    utils_pkg = types.ModuleType("deepmd.utils")
    neighbor_mod = types.ModuleType("deepmd.utils.neighbor_stat")

    backend_mod.Backend = FakeBackendBase  # type: ignore[attr-defined]
    infer_mod.DeepEvalBackend = FakeDeepEvalBackendBase  # type: ignore[attr-defined]
    infer_mod.DeepEval = object  # type: ignore[attr-defined]
    neighbor_mod.NeighborStat = object  # type: ignore[attr-defined]

    sys.modules["deepmd"] = deepmd_mod
    sys.modules["deepmd.backend"] = backend_pkg
    sys.modules["deepmd.backend.backend"] = backend_mod
    sys.modules["deepmd.infer"] = infer_pkg
    sys.modules["deepmd.infer.deep_eval"] = infer_mod
    sys.modules["deepmd.utils"] = utils_pkg
    sys.modules["deepmd.utils.neighbor_stat"] = neighbor_mod


_install_fake_deepmd_modules()
deepmd_backend = importlib.import_module("pretrained_dpa.deepmd_backend")


def test_model_name_from_alias() -> None:
    """Valid alias should parse model name correctly."""
    assert deepmd_backend.parse_pretrained_alias("DPA-3.2-5M.pretrained") == "DPA-3.2-5M"
    assert deepmd_backend.parse_pretrained_alias("relative/DPA-3.2-5M.pretrained") == "DPA-3.2-5M"


@pytest.mark.parametrize(
    "alias",
    [
        "DPA-3.2-5M.pt",
        "DPA-3.2-5M",
        ".pretrained",
    ],
)
def test_model_name_from_alias_rejects_invalid(alias: str) -> None:
    """Invalid alias strings should raise ValueError."""
    with pytest.raises(ValueError, match="Invalid pretrained alias"):
        deepmd_backend.parse_pretrained_alias(alias)


def test_pretrained_deep_eval_backend_resolves_and_delegates(monkeypatch, tmp_path) -> None:
    """Backend should resolve alias and delegate all backend calls."""
    resolved_model = tmp_path / "DPA-3.2-5M.pt"
    resolved_model.write_bytes(b"ok")

    calls: dict[str, object] = {}

    class FakeDeepEvalImpl:
        def __init__(
            self,
            model_file: str,
            output_def: object,
            *args: object,
            auto_batch_size: object = True,
            neighbor_list: object = None,
            **kwargs: object,
        ) -> None:
            calls["model_file"] = model_file
            calls["output_def"] = output_def
            calls["args"] = args
            calls["auto_batch_size"] = auto_batch_size
            calls["neighbor_list"] = neighbor_list
            calls["kwargs"] = kwargs

        def eval(self, *_args: object, **_kwargs: object) -> str:
            return "ok"

        def get_rcut(self) -> float:
            return 1.0

        def get_ntypes(self) -> int:
            return 2

        def get_type_map(self) -> list[str]:
            return ["H", "O"]

        def get_dim_fparam(self) -> int:
            return 3

        def has_default_fparam(self) -> bool:
            return True

        def get_dim_aparam(self) -> int:
            return 4

        @property
        def model_type(self):
            return object

        def get_sel_type(self) -> list[int]:
            return [0, 1]

        def get_numb_dos(self) -> int:
            return 5

        def get_has_efield(self) -> bool:
            return True

        def get_has_spin(self) -> bool:
            return True

        def get_has_hessian(self) -> bool:
            return True

        def get_var_name(self) -> str:
            return "energy"

        def get_ntypes_spin(self) -> int:
            return 1

        def get_model(self) -> object:
            return {"model": "x"}

    def fake_delegate(
        model_file: str,
        output_def: object,
        *args: object,
        auto_batch_size: object = True,
        neighbor_list: object = None,
        **kwargs: object,
    ) -> FakeDeepEvalImpl:
        return FakeDeepEvalImpl(
            model_file,
            output_def,
            *args,
            auto_batch_size=auto_batch_size,
            neighbor_list=neighbor_list,
            **kwargs,
        )

    monkeypatch.setattr(
        deepmd_backend,
        "resolve_model_path",
        lambda name: Path(resolved_model) if name == "DPA-3.2-5M" else Path("bad"),
    )
    monkeypatch.setattr(deepmd_backend, "_delegate_deep_eval", fake_delegate)

    backend_cls = deepmd_backend._get_pretrained_deep_eval_backend()
    backend = backend_cls(
        "DPA-3.2-5M.pretrained",
        {"od": "x"},
        "arg1",
        auto_batch_size=123,
        neighbor_list="nl",
        extra="kw",
    )

    assert calls["model_file"] == str(resolved_model)
    assert calls["output_def"] == {"od": "x"}
    assert calls["args"] == ("arg1",)
    assert calls["auto_batch_size"] == 123
    assert calls["neighbor_list"] == "nl"
    assert calls["kwargs"] == {"extra": "kw"}

    assert backend.eval("coords", "cells", "types") == "ok"
    assert backend.get_rcut() == 1.0
    assert backend.get_ntypes() == 2
    assert backend.get_type_map() == ["H", "O"]
    assert backend.get_dim_fparam() == 3
    assert backend.has_default_fparam() is True
    assert backend.get_dim_aparam() == 4
    assert backend.model_type is object
    assert backend.get_sel_type() == [0, 1]
    assert backend.get_numb_dos() == 5
    assert backend.get_has_efield() is True
    assert backend.get_has_spin() is True
    assert backend.get_has_hessian() is True
    assert backend.get_var_name() == "energy"
    assert backend.get_ntypes_spin() == 1
    assert backend.get_model() == {"model": "x"}


def test_pretrained_backend_interface_and_not_implemented() -> None:
    """Pretrained backend should expose deep_eval and reject unsupported hooks."""
    backend = deepmd_backend.PretrainedBackend()

    assert backend.is_available() is True
    assert backend.suffixes == [".pretrained"]
    assert backend.deep_eval is deepmd_backend._get_pretrained_deep_eval_backend()

    with pytest.raises(NotImplementedError, match="Entry point"):
        _ = backend.entry_point_hook

    with pytest.raises(NotImplementedError, match="Neighbor stat"):
        _ = backend.neighbor_stat

    with pytest.raises(NotImplementedError, match="Serialize hook"):
        _ = backend.serialize_hook

    with pytest.raises(NotImplementedError, match="Deserialize hook"):
        _ = backend.deserialize_hook
