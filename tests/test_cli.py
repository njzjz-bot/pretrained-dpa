"""Tests for CLI model download behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pretrained_dpa import cli

if TYPE_CHECKING:
    from pathlib import Path


def test_download_unknown_model(capsys) -> None:
    """Unknown model name should return code 2 and show available models."""
    code = cli.download_model("NOT-EXIST")

    captured = capsys.readouterr()
    assert code == 2
    assert "Unknown model: NOT-EXIST" in captured.err
    assert "DPA-3.2-5M" in captured.err


def test_download_existing_model_skips_download(monkeypatch, tmp_path, capsys) -> None:
    """If model file exists, downloader should not run and path is reported."""
    model_dir = tmp_path / "cache"
    model_file = model_dir / "DPA-3.2-5M.pt"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file.write_bytes(b"already here")

    monkeypatch.setattr(cli, "DEFAULT_CACHE_DIR", model_dir)

    code = cli.download_model("DPA-3.2-5M")

    captured = capsys.readouterr()
    assert code == 0
    assert "already exists" in captured.out
    assert str(model_file) in captured.out


def test_download_model_success(monkeypatch, tmp_path, capsys) -> None:
    """Download command should write target file and report path."""
    model_dir = tmp_path / "cache"
    model_file = model_dir / "DPA-3.2-5M.pt"
    monkeypatch.setattr(cli, "DEFAULT_CACHE_DIR", model_dir)

    def fake_download(url: str, destination: Path) -> None:
        """Write predictable bytes as a fake download."""
        assert "DPA-3.2-5M.pt" in url
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"fake-model")

    monkeypatch.setattr(cli, "_download_file", fake_download)

    code = cli.download_model("DPA-3.2-5M")

    captured = capsys.readouterr()
    assert code == 0
    assert model_file.exists()
    assert "Downloaded 'DPA-3.2-5M' to:" in captured.out
    assert str(model_file) in captured.out


def test_main_download_success(monkeypatch, tmp_path) -> None:
    """Main should return zero when download subcommand succeeds."""
    model_dir = tmp_path / "cache"
    monkeypatch.setattr(cli, "DEFAULT_CACHE_DIR", model_dir)

    def fake_download(_url: str, destination: Path) -> None:
        """Write a file without network activity."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"ok")

    monkeypatch.setattr(cli, "_download_file", fake_download)

    assert cli.main(["download", "DPA-3.2-5M"]) == 0
