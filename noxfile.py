"""Nox configuration file."""

from __future__ import annotations

import nox


@nox.session
def tests(session: nox.Session) -> None:
    """Run unit test suite with pytest."""
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
        "-k",
        "not deepmd_integration",
    )


@nox.session(name="tests-deepmd")
def tests_deepmd(session: nox.Session) -> None:
    """Run DeepMD integration smoke tests with real dependency stack."""
    session.install("-e.[test,test-deepmd]")
    session.run("pytest", "tests/test_deepmd_integration.py", "-q")
