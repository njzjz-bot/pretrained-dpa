"""Tests for CLI model download behavior."""

from __future__ import annotations

import hashlib
import io
import logging
import urllib.error

from pretrained_dpa import cli

MODEL_NAME = "DPA-3.2-5M"
MODEL_URL = "https://example.com/DPA-3.2-5M.pt"
MODEL_FILENAME = "DPA-3.2-5M.pt"
HF_MODEL_URL = "https://huggingface.co/deepmodelingcommunity/DPA-3.2-5M/resolve/main/DPA-3.2-5M.pt?download=true"


class ResponseOK:
    """Minimal context-manager response object for successful downloads."""

    def __init__(self, payload: bytes) -> None:
        self._stream = io.BytesIO(payload)

    def read(self, size: int = -1) -> bytes:
        """Read bytes like a file object."""
        return self._stream.read(size)

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, *_args: object) -> None:
        """Exit context manager without swallowing errors."""


class ResponseFail:
    """Minimal context-manager response object that fails while reading."""

    def read(self, _size: int = -1) -> bytes:
        """Raise to simulate broken connection."""
        msg = "boom"
        raise OSError(msg)

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, *_args: object) -> None:
        """Exit context manager without swallowing errors."""


def _model_map_with_hash(sha256: str, *, url: str = MODEL_URL) -> dict[str, dict[str, str]]:
    """Create a predictable model mapping for tests."""
    return {
        MODEL_NAME: {
            "url": url,
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


def test_download_unknown_model_uses_packaged_map(caplog) -> None:
    """Unknown model should fail and list available packaged models."""
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
    monkeypatch.setattr(cli, "_select_download_url", lambda _url: MODEL_URL)

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
    monkeypatch.setattr(cli, "_select_download_url", lambda _url: MODEL_URL)

    def fake_urlopen(url: str, timeout: int = 120) -> ResponseOK:
        """Return deterministic payload without network access."""
        assert MODEL_FILENAME in url
        assert timeout == 120
        return ResponseOK(payload)

    monkeypatch.setattr(cli.urllib.request, "urlopen", fake_urlopen)

    with caplog.at_level(logging.INFO):
        code = cli.download_model(MODEL_NAME)

    assert code == 0
    assert model_file.exists()
    assert model_file.read_bytes() == payload
    assert f"Downloaded '{MODEL_NAME}' to:" in caplog.text
    assert str(model_file) in caplog.text


def test_download_model_bad_hash_is_removed(monkeypatch, tmp_path, caplog) -> None:
    """Downloaded file with wrong hash should be removed and return error."""
    model_dir = tmp_path / "cache"
    model_file = model_dir / MODEL_FILENAME
    monkeypatch.setattr(cli, "DEFAULT_CACHE_DIR", model_dir)
    monkeypatch.setattr(cli, "_load_model_map", lambda: _model_map_with_hash("0" * 64))
    monkeypatch.setattr(cli, "_select_download_url", lambda _url: MODEL_URL)

    def fake_urlopen(_url: str, timeout: int = 120) -> ResponseOK:
        """Return deterministic payload without network access."""
        assert timeout == 120
        return ResponseOK(b"corrupted")

    monkeypatch.setattr(cli.urllib.request, "urlopen", fake_urlopen)

    with caplog.at_level(logging.ERROR):
        code = cli.download_model(MODEL_NAME)

    assert code == 1
    assert "SHA256 verification failed" in caplog.text
    assert "Expected:" in caplog.text
    assert "Actual:" in caplog.text
    assert not model_file.exists()


def test_download_existing_bad_hash_triggers_redownload(monkeypatch, tmp_path, caplog) -> None:
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
    monkeypatch.setattr(cli, "_select_download_url", lambda _url: MODEL_URL)

    def fake_urlopen(_url: str, timeout: int = 120) -> ResponseOK:
        """Return deterministic payload without network access."""
        assert timeout == 120
        return ResponseOK(good_payload)

    monkeypatch.setattr(cli.urllib.request, "urlopen", fake_urlopen)

    with caplog.at_level(logging.INFO):
        code = cli.download_model(MODEL_NAME)

    assert code == 0
    assert "failed SHA256 check, re-downloading" in caplog.text
    assert model_file.read_bytes() == good_payload


def test_download_network_error_returns_one(monkeypatch, caplog, tmp_path) -> None:
    """Failure during stream copy should return 1 and clean temporary files."""
    model_dir = tmp_path / "cache"
    output_path = model_dir / MODEL_FILENAME
    part_path = output_path.with_suffix(".pt.part")

    payload = b"ok"
    monkeypatch.setattr(cli, "DEFAULT_CACHE_DIR", model_dir)
    monkeypatch.setattr(
        cli,
        "_load_model_map",
        lambda: _model_map_with_hash(hashlib.sha256(payload).hexdigest()),
    )
    monkeypatch.setattr(cli, "_select_download_url", lambda _url: MODEL_URL)

    def fake_urlopen(_url: str, timeout: int = 120) -> ResponseFail:
        """Return failing response object."""
        assert timeout == 120
        return ResponseFail()

    monkeypatch.setattr(cli.urllib.request, "urlopen", fake_urlopen)

    with caplog.at_level(logging.ERROR):
        code = cli.download_model(MODEL_NAME)

    assert code == 1
    assert "Failed to download" in caplog.text
    assert not output_path.exists()
    assert not part_path.exists()


def test_download_uses_mirror_in_cn(monkeypatch, tmp_path, caplog) -> None:
    """Download should use hf-mirror when country API reports CN."""
    model_dir = tmp_path / "cache"
    payload = b"mirror-model"
    mirror_url = HF_MODEL_URL.replace(cli.HF_ORIGIN, cli.HF_MIRROR, 1)

    monkeypatch.setattr(cli, "DEFAULT_CACHE_DIR", model_dir)
    monkeypatch.setattr(
        cli,
        "_load_model_map",
        lambda: _model_map_with_hash(hashlib.sha256(payload).hexdigest(), url=HF_MODEL_URL),
    )

    def fake_urlopen(url: str, timeout: int = 120) -> ResponseOK:
        """Return country first, then model payload from selected URL."""
        if url == cli.COUNTRY_API_URL:
            assert timeout == 5
            return ResponseOK(b"CN\n")

        assert url == mirror_url
        assert timeout == 120
        return ResponseOK(payload)

    monkeypatch.setattr(cli.urllib.request, "urlopen", fake_urlopen)

    with caplog.at_level(logging.INFO):
        code = cli.download_model(MODEL_NAME)

    assert code == 0
    assert "using mirror" in caplog.text


def test_download_country_check_failure_falls_back_to_origin(monkeypatch, tmp_path) -> None:
    """If country API fails, download should fall back to huggingface origin URL."""
    model_dir = tmp_path / "cache"
    payload = b"origin-model"

    monkeypatch.setattr(cli, "DEFAULT_CACHE_DIR", model_dir)
    monkeypatch.setattr(
        cli,
        "_load_model_map",
        lambda: _model_map_with_hash(hashlib.sha256(payload).hexdigest(), url=HF_MODEL_URL),
    )

    def fake_urlopen(url: str, timeout: int = 120) -> ResponseOK:
        """Raise on country API and serve payload from origin URL."""
        if url == cli.COUNTRY_API_URL:
            msg = "country-unreachable"
            raise urllib.error.URLError(msg)

        assert url == HF_MODEL_URL
        assert timeout == 120
        return ResponseOK(payload)

    monkeypatch.setattr(cli.urllib.request, "urlopen", fake_urlopen)

    assert cli.download_model(MODEL_NAME) == 0


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
    monkeypatch.setattr(cli, "_select_download_url", lambda _url: MODEL_URL)

    def fake_urlopen(_url: str, timeout: int = 120) -> ResponseOK:
        """Return deterministic payload without network access."""
        assert timeout == 120
        return ResponseOK(payload)

    monkeypatch.setattr(cli.urllib.request, "urlopen", fake_urlopen)

    assert cli.main(["download", MODEL_NAME]) == 0
