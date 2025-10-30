#!/usr/bin/env python3
"""
Setup script for nmf-sound-localizer package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
def read_requirements():
    """Read requirements from requirements.txt"""
    requirements_path = this_directory / "requirements.txt"
    if requirements_path.exists():
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [
                line.strip() for line in f 
                if line.strip() and not line.startswith('#') and not line.startswith('-')
            ]
    return [
        'torch>=1.9.0',
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'matplotlib>=3.3.0',
        'tqdm>=4.60.0',
    ]

setup(
    name="nmf-sound-localizer",
    version="1.0.0",
    author="Speech Processing Lab",
    author_email="contact@speechlab.example",
    description="A modular toolkit for NMF-based sound source localization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/speechlab/nmf-sound-localizer",
    project_urls={
        "Bug Reports": "https://github.com/speechlab/nmf-sound-localizer/issues",
        "Source": "https://github.com/speechlab/nmf-sound-localizer",
        "Documentation": "https://nmf-sound-localizer.readthedocs.io/",
        "Changelog": "https://github.com/speechlab/nmf-sound-localizer/blob/main/CHANGELOG.md",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Multimedia :: Sound/Audio :: Sound Synthesis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Environment :: Console",
        "Environment :: GPU :: NVIDIA CUDA",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "isort>=5.10.0",
            "mypy>=1.0.0",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
            "sphinx-autodoc-typehints>=1.19.0",
        ],
        "viz": [
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
            "ipython>=8.0.0",
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
        ],
        "gpu": [
            "torch[cuda]>=1.9.0",
        ],
        "all": [
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
            "ipython>=8.0.0",
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "nmf-localizer=nmf_localizer.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "nmf_localizer": [
            "*.md",
            "configs/*.yaml",
            "configs/*.json",
        ],
    },
    keywords=[
        "audio processing",
        "sound localization", 
        "source separation",
        "non-negative matrix factorization",
        "nmf",
        "signal processing",
        "acoustic analysis",
        "spatial audio",
        "machine learning",
        "pytorch",
        "scientific computing",
    ],
    zip_safe=False,
    test_suite="tests",
)