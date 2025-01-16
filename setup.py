from setuptools import setup, find_packages

setup(
    name="titan_converted",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "fairscale>=0.4.0",
        "pytest>=7.0.0",
        "sentencepiece>=0.1.99",
    ],
)
