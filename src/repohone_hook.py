#!/usr/bin/env python3
"""Hook entry point. Kept tiny so the host's per-hook timeout covers real work."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from repohone.cli import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["hook"]))
