from setuptools import setup, find_packages

REQUIREMENTS = [
    "numpy>=1.20",
    "scipy>=1.7",
    "matplotlib>=3.0",
]

setup(
    name="invertiblewavelets",
    version="0.1.0",
    description="Invertible wavelet transform toolkit",
    author="Dr. Alex P. Hoffmann",
    author_email="alex.p.hoffmann@nasa.gov",
    url="https://github.com/yourusername/invertiblewavelets",
    install_requires=REQUIREMENTS,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)