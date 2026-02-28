"""Tests for DeepMD pretrained alias backend integration helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from pretrained_dpa import deepmd_backend


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
    """Backend should resolve alias and instantiate underlying backend deep_eval."""
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
            return 0

        def has_default_fparam(self) -> bool:
            return False

        def get_dim_aparam(self) -> int:
            return 0

        @property
        def model_type(self):
            return object

        def get_sel_type(self) -> list[int]:
            return []

        def get_numb_dos(self) -> int:
            return 0

        def get_has_efield(self) -> bool:
            return False

        def get_has_spin(self) -> bool:
            return False

        def get_has_hessian(self) -> bool:
            return False

        def get_var_name(self) -> str:
            return "energy"

        def get_ntypes_spin(self) -> int:
            return 0

        def get_model(self) -> object:
            return object()

    class FakeBackendCls:
        def __call__(self):
            return self

        @property
        def deep_eval(self):
            return FakeDeepEvalImpl

    monkeypatch.setattr(
        deepmd_backend,
        "resolve_model_path",
        lambda name: Path(resolved_model) if name == "DPA-3.2-5M" else Path("bad"),
    )
    monkeypatch.setattr(
        deepmd_backend.Backend,
        "detect_backend_by_model",
        lambda _filename: FakeBackendCls,
    )

    backend = deepmd_backend._PretrainedDeepEvalBackend(
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

    assert backend.get_rcut() == 1.0
    assert backend.eval("coords", "cells", "types") == "ok"
