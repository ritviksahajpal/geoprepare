#!/usr/bin/env python
"""Minimal setup.py for backwards compatibility.

This file is only needed for:
- Editable installs (`pip install -e .`) with pip < 21.3
- Legacy tooling that doesn't support pyproject.toml

All configuration is in pyproject.toml. This file simply defers to setuptools.
For modern pip (>= 21.3), you can use `pip install -e .` without this file.

To install in development mode:
    pip install -e .
    pip install -e ".[dev]"      # with dev dependencies
    pip install -e ".[docs]"     # with docs dependencies
    pip install -e ".[dev,docs]" # with all optional dependencies
"""

from setuptools import setup

if __name__ == "__main__":
    setup()