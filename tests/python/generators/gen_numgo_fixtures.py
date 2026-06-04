#!/usr/bin/env python3
"""
Standalone fixture generator for numgo cross-validation tests.

Invoked via go:generate from lib/numgo/doc.go:
    //go:generate python3 ../../tests/python/generators/gen_numgo_fixtures.py --output ../../tests/fixtures

Produces tests/fixtures/numgo/fixtures.json with numpy inputs + expected outputs.
"""

import argparse
import json
import os
import sys

# Ensure generators package is importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from generators.gen_numgo import generate


def main():
    parser = argparse.ArgumentParser(description="Generate numgo cross-validation fixtures from numpy")
    parser.add_argument("--output", type=str, required=True, help="Output directory for fixtures")
    args = parser.parse_args()

    cases = generate()
    data = {
        "generator": "numgo",
        "numpy_version": np.__version__,
        "test_cases": cases,
    }

    def _round_floats(obj):
        """Round floats to 12 significant digits for cross-platform reproducibility."""
        if isinstance(obj, float):
            return float(f"{obj:.12g}")
        if isinstance(obj, dict):
            return {k: _round_floats(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_round_floats(x) for x in obj]
        return obj

    out_dir = os.path.join(args.output, "numgo")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "fixtures.json")
    with open(out_file, "w") as f:
        json.dump(_round_floats(data), f, indent=2, sort_keys=True, default=str)

    print(f"numgo: wrote {len(cases)} test cases to {out_file}")


if __name__ == "__main__":
    main()
