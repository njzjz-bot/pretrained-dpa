"""Command line interface for pretrained_dpa."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.error
import urllib.request
from importlib.resources import files
from pathlib import Path

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "pretrained-dpa" / "models"


def _echo(message: str, *, error: bool = False) -> None:
    """Write a message to stdout or stderr."""
    stream = sys.stderr if error else sys.stdout
    stream.write(f"{message}\n")


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

    with urllib.request.urlopen(url, timeout=120) as response, tmp_path.open("wb") as out_file:  # noqa: S310
        shutil.copyfileobj(response, out_file)

    tmp_path.replace(destination)


def download_model(model_name: str) -> int:
    """Download a named pretrained model if it is not already cached."""
    model_map = _load_model_map()
    model_info = model_map.get(model_name)
    if model_info is None:
        available = ", ".join(sorted(model_map))
        _echo(f"Unknown model: {model_name}", error=True)
        _echo(f"Available models: {available}", error=True)
        return 2

    filename = model_info["filename"]
    url = model_info["url"]
    output_path = DEFAULT_CACHE_DIR / filename

    if output_path.exists():
        _echo(f"Model '{model_name}' already exists at:")
        _echo(str(output_path))
        return 0

    _echo(f"Downloading '{model_name}'...")
    try:
        _download_file(url, output_path)
    except (urllib.error.URLError, OSError) as exc:
        _echo(f"Failed to download '{model_name}': {exc}", error=True)
        return 1

    _echo(f"Downloaded '{model_name}' to:")
    _echo(str(output_path))
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
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "download":
        return download_model(args.model_name)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
