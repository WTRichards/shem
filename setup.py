#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="shem",
    description="A GPU-accelerated SHeM simulations library based on PyTorch.",
    version="0.1.0",
    author="William Thomas Richards",
    license="MIT",
    packages=find_packages(include=['shem']),
    install_requires=[],
)
