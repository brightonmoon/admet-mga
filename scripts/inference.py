#!/usr/bin/env python3
"""MGA Inference Script (wrapper for mga-infer CLI).

For direct invocation without package install.
Use 'mga-infer' after 'uv pip install -e .' for the installed CLI.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mga.cli.inference import main

if __name__ == "__main__":
    main()
