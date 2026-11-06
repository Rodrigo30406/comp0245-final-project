"""Run the test suite with src/ importable, without installing the package."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if __name__ == "__main__":
    argv = [sys.argv[0], "discover", "-s", str(ROOT / "tests"), *sys.argv[1:]]
    unittest.main(module=None, argv=argv)
