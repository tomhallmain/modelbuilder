"""
Setup script for Model Builder package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README if available
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()

# Read version from package
version = "0.1.0"
try:
    with open(Path(__file__).parent / "mb" / "__init__.py", "r") as f:
        for line in f:
            if line.startswith("__version__"):
                version = line.split("=")[1].strip().strip('"').strip("'")
                break
except Exception:
    pass

setup(
    name="modelbuilder",
    version=version,
    description="A unified CLI application for building machine learning models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Model Builder Team",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.8",
    install_requires=[
        "Pillow>=9.0.0",
        "numpy>=1.25.2,<2.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "pytorch": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
        ],
        "keras": [
            "tensorflow>=2.10.0",
        ],
        "all": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "tensorflow>=2.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mb=mb.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
