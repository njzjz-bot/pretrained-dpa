"""Integration smoke tests with real deepmd-kit import chain."""

from __future__ import annotations

import importlib
import sys

import pytest


def _clear_deepmd_modules() -> None:
    """Clear preloaded deepmd modules so test runs against real installation."""
    for name in list(sys.modules):
        if name == "deepmd" or name.startswith("deepmd."):
            sys.modules.pop(name, None)


def test_import_deeppot_smoke() -> None:
    """Importing DeepPot should not fail via plugin entrypoint loading."""
    _clear_deepmd_modules()
    pytest.importorskip("deepmd")

    infer_module = importlib.import_module("deepmd.infer")
    deep_pot = getattr(infer_module, "DeepPot", None)
    if deep_pot is None:
        deep_pot_module = importlib.import_module("deepmd.infer.deep_pot")
        deep_pot = getattr(deep_pot_module, "DeepPot", None)

    assert deep_pot is not None
