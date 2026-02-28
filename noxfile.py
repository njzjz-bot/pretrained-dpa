"""Nox configuration file."""

from __future__ import annotations

import nox


@nox.session
def tests(session: nox.Session) -> None:
    """Run full test suite (unit + DeepMD integration) with pytest."""
    session.install("-e.[test]")
    session.run(
        "pytest",
        "--cov",
        "--cov-config",
        "pyproject.toml",
        "--cov-report",
        "term",
        "--cov-report",
        "xml",
    )
