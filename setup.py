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
    package_data={"mb": ["config/*.yaml"]},
    python_requires=">=3.8",
    install_requires=[
        "Pillow>=9.0.0",
        "numpy>=1.25.2,<2.0",
        "imageio>=2.31.0",
        "imageio-ffmpeg>=0.4.9",
        "pyyaml>=6.0",
        "cryptography>=41.0.0",
        "keyring>=24.0.0",
        "PySide6>=6.5.0",
    ],
    extras_require={
        "pytorch": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            # Used by ``mb convert``/``mb export bundle``/``mb info`` SafeTensors support.
            "safetensors>=0.4.0",
        ],
        "keras": [
            "tensorflow>=2.10.0",
        ],
        "onnx": [
            "onnx>=1.12.0",
            # torch.onnx.export (PyTorch 2.x) imports onnxscript internally
            "onnxscript>=0.1.0",
        ],
        # Image-generation LoRA training (mb train --model-type image_generation_lora).
        # PyTorch only; install "pytorch" alongside this.
        "lora": [
            "diffusers>=0.30.0",  # Flux support
            "peft>=0.10.0",
            "transformers>=4.30.0",
            "safetensors>=0.4.0",
        ],
        "post-quantum": [
            "liboqs-python @ git+https://github.com/open-quantum-safe/liboqs-python.git",
        ],
        # Optional platform integrations for keyring / encryptor (not required for basic keyring).
        "keyring-extras": [
            "pyobjc-framework-Cocoa; sys_platform == 'darwin'",
            "dbus-python; sys_platform == 'linux'",
        ],
        "all": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "tensorflow>=2.10.0",
            "onnx>=1.12.0",
            "onnxscript>=0.1.0",
            "safetensors>=0.4.0",
            "diffusers>=0.30.0",  # Flux support
            "peft>=0.10.0",
            "transformers>=4.30.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mb=mb.cli:main",
        ],
        "gui_scripts": [
            "mb-gui=ui.app:main",
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
