# setup.py
from setuptools import setup, find_packages

setup(
    name="sentence-generalized-pooling",
    version="0.2.8",
    packages=find_packages(),
    install_requires=[
        "torch>=1.7.0",
        "sentence-transformers>=2.0.0"
    ],
    author="Romain Darous",
    author_email="romain.darous@telecom-paris.fr",
    description="A library implementing generalized pooling for sentence transformers",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sentence-generalized-pooling",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="MIT",
    python_requires=">=3.6",
)