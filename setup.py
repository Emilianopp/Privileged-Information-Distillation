from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="privileged-distillation",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="PyTorch library for privileged information distillation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/privileged-distillation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "examples": [
            "matplotlib>=3.5.0",
            "tqdm>=4.64.0",
        ],
    },
    keywords="pytorch machine-learning reinforcement-learning knowledge-distillation privileged-information",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/privileged-distillation/issues",
        "Source": "https://github.com/yourusername/privileged-distillation",
    },
)
