"""Tests for CLI model download behavior."""

from __future__ import annotations

import hashlib
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


def test_download_unknown_model(capsys, monkeypatch) -> None:
    """Unknown model name should return code 2 and show available models."""
    monkeypatch.setattr(cli, "_load_model_map", lambda: _model_map_with_hash("0" * 64))

    code = cli.download_model("NOT-EXIST")

    captured = capsys.readouterr()
    assert code == 2
    assert "Unknown model: NOT-EXIST" in captured.err
    assert MODEL_NAME in captured.err


def test_download_existing_model_skips_download(monkeypatch, tmp_path, capsys) -> None:
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

    code = cli.download_model(MODEL_NAME)

    captured = capsys.readouterr()
    assert code == 0
    assert "already exists" in captured.out
    assert str(model_file) in captured.out


def test_download_model_success(monkeypatch, tmp_path, capsys) -> None:
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

    code = cli.download_model(MODEL_NAME)

    captured = capsys.readouterr()
    assert code == 0
    assert model_file.exists()
    assert f"Downloaded '{MODEL_NAME}' to:" in captured.out
    assert str(model_file) in captured.out


def test_download_model_bad_hash_is_removed(monkeypatch, tmp_path, capsys) -> None:
    """Downloaded file with wrong hash should be removed and return error."""
    model_dir = tmp_path / "cache"
    model_file = model_dir / MODEL_FILENAME
    monkeypatch.setattr(cli, "DEFAULT_CACHE_DIR", model_dir)
    monkeypatch.setattr(cli, "_load_model_map", lambda: _model_map_with_hash("0" * 64))

    def fake_download(_url: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"corrupted")

    monkeypatch.setattr(cli, "_download_file", fake_download)

    code = cli.download_model(MODEL_NAME)
    captured = capsys.readouterr()

    assert code == 1
    assert "SHA256 verification failed" in captured.err
    assert not model_file.exists()


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
