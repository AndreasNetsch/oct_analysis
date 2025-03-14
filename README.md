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

### Creating Releases

The CI/CD pipeline is configured to automatically build and publish the package to PyPI when a new version tag is pushed to the repository. This process ensures that only properly versioned, tagged releases get published.

To create and publish a new release:

1. Update the version number in `setup.py`
2. Commit your changes:
   ```bash
   git add setup.py
   git commit -m "Bump version to x.y.z"
   ```
3. Create a new version tag (tag name must start with "v"):
   ```bash
   git tag vx.y.z
   ```
4. Push the tag to GitHub:
   ```bash
   git push origin vx.y.z
   ```

Once the tag is pushed, GitHub Actions will:

1. Run all tests on multiple Python versions
2. If tests pass, build the package
3. Publish the package to PyPI using the configured PyPI API token

Note: Make sure you've added a `PYPI_API_TOKEN` secret to your GitHub repository settings under "Settings > Secrets and Variables > Actions" before triggering a release.

## License

MIT License
