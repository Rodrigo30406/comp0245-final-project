"""Gate A1, full suite: unittest discovery of tests/test_*.py with the test artefacts redirected to this folder.

Several existing tests write their fixtures and evidence into results/ folders of earlier steps (results/step6/smoke,
results/step7/smoke, results/tmp_policies), and test_ee_features compares two files with git HEAD. Track A may not
write outside results/step9/ and runs no git command, so this loader rebinds those destinations, in memory only, to
results/step9/smoke/full_suite/ and skips that single git test; no test source or production module is changed.

Usage (from the task folder):  python -m unittest results.step9.gate_full_suite
"""
import os
from pathlib import Path
import sys
import tempfile
import unittest

TASK = Path(__file__).resolve().parents[2]
OUT = TASK / "results/step9/smoke/full_suite"


def load_tests(loader, tests, pattern):
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(TASK / "tests"))
    suite = loader.discover(str(TASK / "tests"), pattern="test_*.py", top_level_dir=str(TASK / "tests"))
    for name in ("test_chunk", "test_pd_layer"):                 # evidence files of step 6
        if name in sys.modules:
            sys.modules[name].SMOKE = OUT
    if "test_step7" in sys.modules:                              # step-7 fixtures
        sys.modules["test_step7"].temporary_fixture = lambda: tempfile.TemporaryDirectory(prefix="step7_", dir=OUT)
    if "test_policies" in sys.modules:                           # test models (results/tmp_policies)
        mod = sys.modules["test_policies"]
        mod.TMP_DIR = str(OUT / "models")
        mod.TEST_MODELS = {k: os.path.join(mod.TMP_DIR, os.path.basename(v)) for k, v in mod.TEST_MODELS.items()}
        for other in ("test_ee_features", "test_hybrid"):        # they imported TMP_DIR / TEST_MODELS by value
            if other in sys.modules:
                sys.modules[other].TMP_DIR = mod.TMP_DIR
                if hasattr(sys.modules[other], "TEST_MODELS"):
                    sys.modules[other].TEST_MODELS = mod.TEST_MODELS
    if "test_ee_features" in sys.modules:
        mod = sys.modules["test_ee_features"]
        mod.EE_TEST_MODEL = str(OUT / "models" / "test_ego_novw_ee_mlp.npz")
        cls = mod.TestOldModelsUnchanged
        cls.test_bitwise_against_head = unittest.skip("Track A runs no git command (git show HEAD)")(
            cls.test_bitwise_against_head)
    return suite
