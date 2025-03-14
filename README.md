# andylib

A library for image processing functions.

## Installation

```bash
pip install andylib
```

## Features

- Read TIFF image files

## Usage

```python
import numpy as np
from andylib import read_tiff

# Read a TIFF image
image = read_tiff('path/to/your/image.tiff')
print(f"Image shape: {image.shape}")
```

## Development

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/andylib.git
cd andylib
```

2. Create a virtual environment and install development dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

### Testing

Run tests with pytest:

```bash
pytest
```

### Code Formatting and Linting

This project uses pre-commit hooks to ensure code quality. After installing the development dependencies, set up the pre-commit hooks:

```bash
pre-commit install
```

This will automatically format your code with Black and check it with Flake8 before each commit. You can also run the hooks manually:

```bash
pre-commit run --all-files
```

### Building the package

```bash
python -m build
```

## CI/CD

This project uses GitHub Actions for:

- Running tests on multiple Python versions
- Linting the code
- Building and publishing the package to PyPI when a new version tag is pushed

## License

MIT License
