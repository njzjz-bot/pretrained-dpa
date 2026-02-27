"""Test version."""

from __future__ import annotations

from importlib.metadata import version

from pretrained_dpa import __version__


def test_version() -> None:
    """Test version."""
    assert version("pretrained-dpa") == __version__
