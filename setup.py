from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="char-rnn-pytorch",
    version="2.0.0",
    author="Modernized from Andrej Karpathy's char-rnn",
    description="Character-level RNN for text generation with PyTorch and MPS support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/char-rnn-pytorch",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "PyYAML>=6.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.64.0",
        "tensorboard>=2.10.0",
        "click>=8.0.0",
        "rich>=12.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "char-rnn-train=scripts.train:main",
            "char-rnn-sample=scripts.sample:main",
        ],
    },
)
