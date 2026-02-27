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

If the file already exists, it will not be downloaded again, and the CLI will print the existing location.
