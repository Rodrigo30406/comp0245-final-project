"""Exercise every sequential gate of the frozen experiment F decision rule."""
import os
import sys
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "step5"))
from F_mechanisms import dev_pass, hard_pass, verdict
import step5_pipeline as pipeline


class TestFGates(unittest.TestCase):
    def test_failed_gates_prevent_control_reads_and_evaluation(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(pipeline, "HERE", Path(directory)):
            for evidence in ({}, {"dev": {"condition_1": True, "condition_2": False}},
                             {"dev": {"condition_1": True, "condition_2": True},
                              "hard": {"condition_3": False}}):
                (Path(directory) / "F_mechanisms.json").write_text(json.dumps(evidence))
                sets = ("hard", "confirm") if not dev_pass(evidence) else ("confirm",)
                for set_name in sets:
                    with patch.object(pipeline.ar, "load_run") as load_run, \
                         patch.object(pipeline, "command") as command, \
                         patch.object(pipeline, "model_checks") as models:
                        with self.assertRaisesRegex(RuntimeError, "gate failed"):
                            pipeline.stalled_controls(set_name)
                        with self.assertRaisesRegex(RuntimeError, "gate failed"):
                            pipeline.evaluations(set_name)
                        with self.assertRaisesRegex(RuntimeError, "gate failed"):
                            pipeline.evaluate(Path("unused.npz"), set_name, Path(directory) / "unused")
                        load_run.assert_not_called()
                        command.assert_not_called()
                        models.assert_not_called()

    def test_gated_f5_reruns_only_stalled_controls_before_new_models(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(pipeline, "HERE", Path(directory)):
            evidence = {"dev": {"condition_1": True, "condition_2": True},
                        "hard": {"condition_3": True}}
            (Path(directory) / "F_mechanisms.json").write_text(json.dumps(evidence))
            for set_name, seed in (("hard", 0), ("confirm", 100)):
                run = SimpleNamespace(meta={"no_early_stop": False},
                                      rows=[{"episode": 3, "stop_reason": "stalled"},
                                            {"episode": 4, "stop_reason": "reached"}])
                calls = []
                with patch.object(pipeline, "SEEDS", {set_name: [seed]}), \
                     patch.object(pipeline, "model_checks"), \
                     patch.object(pipeline.ar, "load_run", return_value=run), \
                     patch.object(pipeline, "evaluate", side_effect=lambda *a, **k: calls.append((a, k))), \
                     patch.object(pipeline, "command") as command, \
                     patch.object(pipeline.ar, "ALLOW_CONFIRM", False):
                    pipeline.evaluations(set_name)
                self.assertEqual(len(calls), 2)
                control, new = calls
                self.assertEqual(control[0][0].name, f"ego-novw-trig_s{seed}.npz")
                self.assertEqual(control[0][3], [3])
                self.assertEqual(control[0][2], Path(directory) / "nostall" /
                                 f"ego-novw-trig_s{seed}_{set_name}")
                self.assertEqual(new[0][0].name, f"ego-novw-trig-dag1_s{seed}.npz")
                self.assertEqual("--allow-confirm" in command.call_args.args[0], set_name == "confirm")
            self.assertTrue((Path(directory) / "confirm_opened.json").exists())

    def test_sequential_decisions(self):
        j = {"dev": None, "hard": None, "confirm": None}
        self.assertEqual(verdict(j), "PENDING DEV")
        for c1, c2 in ((False, False), (False, True), (True, False)):
            j["dev"] = {"condition_1": c1, "condition_2": c2}
            self.assertFalse(dev_pass(j))
            self.assertFalse(hard_pass(j))
            self.assertEqual(verdict(j), "KEEP C3")
        j["dev"] = {"condition_1": True, "condition_2": True}
        self.assertEqual(verdict(j), "NEXT HARD")
        j["hard"] = {"condition_3": False}
        self.assertFalse(hard_pass(j))
        self.assertEqual(verdict(j), "KEEP C3")
        j["hard"] = {"condition_3": True}
        self.assertTrue(hard_pass(j))
        self.assertEqual(verdict(j), "NEXT CONFIRM")
        j["confirm"] = {"condition_4": False}
        self.assertEqual(verdict(j), "KEEP C3")
        j["confirm"] = {"condition_4": True}
        self.assertEqual(verdict(j), "ADOPT dag1")


if __name__ == "__main__":
    unittest.main()
