#!/usr/bin/env python3
"""MGA Data Preparation Script (wrapper for mga-data CLI).

For direct invocation without package install.
Use 'mga-data' after 'uv pip install -e .' for the installed CLI.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mga.cli.prepare_data import main

if __name__ == "__main__":
    main()
