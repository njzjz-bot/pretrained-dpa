# Pretrained DPA

<!-- [![PyPI - Version](https://img.shields.io/pypi/v/pretrained-dpa)](https://pypi.org/p/pretrained-dpa) -->

`pretrained-dpa` provides CLI tools for managing pretrained DPA model files.

## CLI

### Download a model

```bash
pretrained-dpa download DPA-3.2-5M
```

This command will download the model file from Hugging Face to:

```text
~/.cache/pretrained-dpa/models/DPA-3.2-5M.pt
```

The CLI validates SHA256 against metadata in `pretrained_dpa/models.json`.
If an existing cached file fails verification, it is deleted and re-downloaded.
If a fresh download fails checksum verification, it is deleted and the command exits with an error.

When running in China (`https://ipinfo.io/country` returns `CN`), Hugging Face URLs are automatically rewritten to the mirror:

```text
https://hf-mirror.com
```
