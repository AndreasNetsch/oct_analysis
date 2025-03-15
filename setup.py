from setuptools import setup, find_packages
from pathlib import Path

# Read the long description from README.md
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="oct_analysis",
    version="0.1.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=["numpy", "opencv-python"],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "flake8>=4.0.0",
            "black>=22.0.0",
            "build",
            "twine",
            "pre-commit>=3.0.0",
        ],
    },
    author="Andreas Netsch",
    author_email="your.email@example.com",
    description="A library for image processing functions.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/oct_analysis",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
