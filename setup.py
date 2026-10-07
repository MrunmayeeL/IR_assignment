from setuptools import setup, find_packages

setup(
    name="lexis",
    version="1.0.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "pyyaml>=6.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "psutil>=5.9.0",
        "matplotlib>=3.7.0",
    ],
    entry_points={
        "console_scripts": [
            "lexis=lexis.cli:main",
        ],
    },
    python_requires=">=3.8",
)
