from setuptools import setup, find_packages

setup(
    name="minicv",
    version="1.0.0",
    description=(
        "A minimal OpenCV-like image-processing library built from scratch "
        "using NumPy, Matplotlib, Pandas, and the Python standard library."
    ),
    packages=find_packages(include=["minicv", "minicv.*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "matplotlib>=3.6",
        "pandas>=1.5",
    ],
    extras_require={
        "dev": ["pytest>=7.0"],
    },
)
