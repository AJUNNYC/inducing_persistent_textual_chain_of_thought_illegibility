#!/usr/bin/env python3
"""Revalidate restored CPU collection inputs under one clean consumer source freeze."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from operations.collection_inputs import main

if __name__ == '__main__':
    raise SystemExit(main())
