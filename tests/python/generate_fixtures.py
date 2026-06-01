#!/usr/bin/env python3
"""
pgmgo test fixture generator.

Invokes pgmpy to produce expected outputs for cross-validation testing.
Each generator module in generators/ produces fixtures for a specific
pgmgo package.

Usage:
    python generate_fixtures.py --package <package> --output <dir>
    python generate_fixtures.py --all --output <dir>

Fixture format (JSON):
    {
        "generator": "models.bayesian_network",
        "pgmpy_version": "1.1.2",
        "generated_at": "2026-06-01T00:00:00Z",
        "test_cases": [
            {
                "name": "test_add_node",
                "input": { ... },
                "expected": { ... }
            }
        ]
    }
"""

import argparse
import importlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pgmpy


def discover_generators(generators_dir: Path) -> dict[str, str]:
    """Discover all generator modules in the generators/ directory."""
    generators = {}
    for f in sorted(generators_dir.glob("gen_*.py")):
        module_name = f.stem
        package_name = module_name.removeprefix("gen_")
        generators[package_name] = module_name
    return generators


def run_generator(module_name: str, package_name: str) -> dict:
    """Run a single generator module and return fixture data."""
    mod = importlib.import_module(f"generators.{module_name}")
    test_cases = mod.generate()

    return {
        "generator": package_name,
        "pgmpy_version": pgmpy.__version__,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "test_cases": test_cases,
    }


def write_fixture(data: dict, output_dir: Path, package_name: str) -> Path:
    """Write fixture data to a JSON file."""
    pkg_dir = output_dir / package_name
    pkg_dir.mkdir(parents=True, exist_ok=True)
    output_file = pkg_dir / "fixtures.json"
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Generate pgmgo test fixtures from pgmpy")
    parser.add_argument("--package", type=str, help="Package to generate fixtures for")
    parser.add_argument("--all", action="store_true", help="Generate fixtures for all packages")
    parser.add_argument("--output", type=str, required=True, help="Output directory for fixtures")
    parser.add_argument("--list", action="store_true", help="List available generators")
    args = parser.parse_args()

    generators_dir = Path(__file__).parent / "generators"
    generators = discover_generators(generators_dir)

    if args.list:
        for name in generators:
            print(name)
        return

    if not args.package and not args.all:
        parser.error("Either --package or --all is required")

    output_dir = Path(args.output)
    packages = list(generators.keys()) if args.all else [args.package]

    for pkg in packages:
        if pkg not in generators:
            print(f"ERROR: no generator found for package '{pkg}'", file=sys.stderr)
            print(f"Available: {', '.join(generators.keys())}", file=sys.stderr)
            sys.exit(1)

        print(f"Generating fixtures for {pkg}...")
        data = run_generator(generators[pkg], pkg)
        out_file = write_fixture(data, output_dir, pkg)
        print(f"  -> {out_file} ({len(data['test_cases'])} test cases)")

    print("Done.")


if __name__ == "__main__":
    main()
