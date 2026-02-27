"""Command line interface for pretrained_dpa."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import urllib.error
import urllib.request
from importlib.resources import files
from pathlib import Path

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "pretrained-dpa" / "models"
LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure basic logging for CLI output."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)


def _load_model_map() -> dict[str, dict[str, str]]:
    """Load model metadata from packaged JSON."""
    data_path = files("pretrained_dpa").joinpath("models.json")
    with data_path.open("r", encoding="utf-8") as f:
        data: dict[str, dict[str, str]] = json.load(f)
    return data


def _download_file(url: str, destination: Path) -> None:
    """Download URL content into destination atomically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".part")

    try:
        with urllib.request.urlopen(url, timeout=120) as response, tmp_path.open("wb") as out_file:  # noqa: S310
            shutil.copyfileobj(response, out_file)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    tmp_path.replace(destination)


def _sha256sum(path: Path) -> str:
    """Calculate SHA256 checksum of a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_model(model_name: str) -> int:
    """Download a named pretrained model if it is not already cached."""
    model_map = _load_model_map()
    model_info = model_map.get(model_name)
    if model_info is None:
        available = ", ".join(sorted(model_map))
        LOGGER.error("Unknown model: %s", model_name)
        LOGGER.error("Available models: %s", available)
        return 2

    filename = model_info["filename"]
    url = model_info["url"]
    expected_sha256 = model_info["sha256"]
    output_path = DEFAULT_CACHE_DIR / filename

    if output_path.exists():
        actual_sha256 = _sha256sum(output_path)
        if actual_sha256 == expected_sha256:
            LOGGER.info("Model '%s' already exists at:", model_name)
            LOGGER.info("%s", output_path)
            return 0

        LOGGER.warning("Cached file for '%s' failed SHA256 check, re-downloading...", model_name)
        output_path.unlink(missing_ok=True)

    LOGGER.info("Downloading '%s'...", model_name)
    try:
        _download_file(url, output_path)
    except (urllib.error.URLError, OSError):
        LOGGER.exception("Failed to download '%s'", model_name)
        return 1

    actual_sha256 = _sha256sum(output_path)
    if actual_sha256 != expected_sha256:
        output_path.unlink(missing_ok=True)
        LOGGER.error("Downloaded '%s' but SHA256 verification failed.", model_name)
        LOGGER.error("Expected: %s", expected_sha256)
        LOGGER.error("Actual:   %s", actual_sha256)
        return 1

    LOGGER.info("Downloaded '%s' to:", model_name)
    LOGGER.info("%s", output_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="pretrained-dpa",
        description="Utilities for pretrained DPA model files.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download", help="Download a pretrained model")
    download_parser.add_argument("model_name", help="Model name, e.g. DPA-3.2-5M")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return exit code."""
    configure_logging()

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "download":
        return download_model(args.model_name)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
