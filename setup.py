from setuptools import setup, find_packages

setup(
    name="minicv",
    version="1.0.0",
    description="A minimal OpenCV-like image processing library (CSE480 Spring 2026)",
    packages=find_packages(include=["minicv", "minicv.*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "pandas>=1.5",
        "matplotlib>=3.6",
    ],
    extras_require={
        "ml": ["mrmr-selection>=0.2.6"],
        "dev": ["pytest>=7.0"],
    },
)
