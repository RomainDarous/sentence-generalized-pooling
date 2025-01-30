# setup.py
from setuptools import setup, find_packages

setup(
    name="sentence-generalized-pooling",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.7.0",
        "sentence-transformers>=2.0.0"
    ],
    author="Romain Darous",
    author_email="romaindarous.pro@gmail.com",
    description="A library implementing generalized pooling for sentence transformers",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/RomainDarous/sentence-generalized-pooling",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)