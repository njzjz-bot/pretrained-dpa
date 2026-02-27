"""Tests for CLI model download behavior."""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

from pretrained_dpa import cli

if TYPE_CHECKING:
    from pathlib import Path


MODEL_NAME = "DPA-3.2-5M"
MODEL_URL = "https://example.com/DPA-3.2-5M.pt"
MODEL_FILENAME = "DPA-3.2-5M.pt"


def _model_map_with_hash(sha256: str) -> dict[str, dict[str, str]]:
    """Create a predictable model mapping for tests."""
    return {
        MODEL_NAME: {
            "url": MODEL_URL,
            "filename": MODEL_FILENAME,
            "sha256": sha256,
        },
    }


def test_configure_logging_sets_info_level() -> None:
    """Logging configuration should set root level to INFO."""
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level

    try:
        cli.configure_logging()
        assert root.level == logging.INFO
    finally:
        root.handlers = original_handlers
        root.setLevel(original_level)


def test_download_unknown_model(caplog, monkeypatch) -> None:
    """Unknown model name should return code 2 and show available models."""
    monkeypatch.setattr(cli, "_load_model_map", lambda: _model_map_with_hash("0" * 64))

    with caplog.at_level(logging.ERROR):
        code = cli.download_model("NOT-EXIST")

    assert code == 2
    assert "Unknown model: NOT-EXIST" in caplog.text
    assert MODEL_NAME in caplog.text


def test_download_existing_model_skips_download(monkeypatch, tmp_path, caplog) -> None:
    """If model file exists and hash matches, downloader should not run."""
    model_dir = tmp_path / "cache"
    model_file = model_dir / MODEL_FILENAME
    payload = b"already here"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file.write_bytes(payload)

    monkeypatch.setattr(cli, "DEFAULT_CACHE_DIR", model_dir)
    monkeypatch.setattr(
        cli,
        "_load_model_map",
        lambda: _model_map_with_hash(hashlib.sha256(payload).hexdigest()),
    )

    with caplog.at_level(logging.INFO):
        code = cli.download_model(MODEL_NAME)

    assert code == 0
    assert "already exists" in caplog.text
    assert str(model_file) in caplog.text


def test_download_model_success(monkeypatch, tmp_path, caplog) -> None:
    """Download command should write target file and report path."""
    model_dir = tmp_path / "cache"
    model_file = model_dir / MODEL_FILENAME
    payload = b"fake-model"

    monkeypatch.setattr(cli, "DEFAULT_CACHE_DIR", model_dir)
    monkeypatch.setattr(
        cli,
        "_load_model_map",
        lambda: _model_map_with_hash(hashlib.sha256(payload).hexdigest()),
    )

    def fake_download(url: str, destination: Path) -> None:
        """Write predictable bytes as a fake download."""
        assert MODEL_FILENAME in url
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)

    monkeypatch.setattr(cli, "_download_file", fake_download)

    with caplog.at_level(logging.INFO):
        code = cli.download_model(MODEL_NAME)

    assert code == 0
    assert model_file.exists()
    assert f"Downloaded '{MODEL_NAME}' to:" in caplog.text
    assert str(model_file) in caplog.text


def test_download_model_bad_hash_is_removed(monkeypatch, tmp_path, caplog) -> None:
    """Downloaded file with wrong hash should be removed and return error."""
    model_dir = tmp_path / "cache"
    model_file = model_dir / MODEL_FILENAME
    monkeypatch.setattr(cli, "DEFAULT_CACHE_DIR", model_dir)
    monkeypatch.setattr(cli, "_load_model_map", lambda: _model_map_with_hash("0" * 64))

    def fake_download(_url: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"corrupted")

    monkeypatch.setattr(cli, "_download_file", fake_download)

    with caplog.at_level(logging.ERROR):
        code = cli.download_model(MODEL_NAME)

    assert code == 1
    assert "SHA256 verification failed" in caplog.text
    assert "Expected:" in caplog.text
    assert "Actual:" in caplog.text
    assert not model_file.exists()


def test_download_existing_model_bad_hash_triggers_redownload(monkeypatch, tmp_path, caplog) -> None:
    """Bad cached file should be removed and replaced by downloaded content."""
    model_dir = tmp_path / "cache"
    model_file = model_dir / MODEL_FILENAME
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file.write_bytes(b"bad-cache")

    good_payload = b"good-model"
    monkeypatch.setattr(cli, "DEFAULT_CACHE_DIR", model_dir)
    monkeypatch.setattr(
        cli,
        "_load_model_map",
        lambda: _model_map_with_hash(hashlib.sha256(good_payload).hexdigest()),
    )

    def fake_download(_url: str, destination: Path) -> None:
        destination.write_bytes(good_payload)

    monkeypatch.setattr(cli, "_download_file", fake_download)

    with caplog.at_level(logging.INFO):
        code = cli.download_model(MODEL_NAME)

    assert code == 0
    assert "failed SHA256 check, re-downloading" in caplog.text
    assert model_file.read_bytes() == good_payload


def test_main_download_success(monkeypatch, tmp_path) -> None:
    """Main should return zero when download subcommand succeeds."""
    model_dir = tmp_path / "cache"
    payload = b"ok"

    monkeypatch.setattr(cli, "DEFAULT_CACHE_DIR", model_dir)
    monkeypatch.setattr(
        cli,
        "_load_model_map",
        lambda: _model_map_with_hash(hashlib.sha256(payload).hexdigest()),
    )

    def fake_download(_url: str, destination: Path) -> None:
        """Write a file without network activity."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)

    monkeypatch.setattr(cli, "_download_file", fake_download)

    assert cli.main(["download", MODEL_NAME]) == 0
