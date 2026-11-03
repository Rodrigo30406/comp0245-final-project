"""Exercise step-6 orchestration and frozen decision boundaries without datasets."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "results" / "step6"))
import gp_common as common
from gp_mechanisms import apply_conditions
import step6_pipeline as pipeline


def passed_gates():
    return {"dev": {"condition_1": True, "condition_2": True},
            "hard": {"condition_3": True}}


def metric(diff=0., lo=-.01, hi=.01, welch_lo=-.01, welch_hi=.01):
    return {"diff": diff, "lo": lo, "hi": hi,
            "welch_lo": welch_lo, "welch_hi": welch_hi}


def dev_metrics():
    return {"success": metric(diff=.1, lo=.01, welch_lo=.01),
            "risk": metric(diff=.03), "extended_not_reached": metric(),
            "circling": metric(diff=-.01), "held_10s": metric(diff=.01),
            "crash": metric(diff=-.01)}


class TestGPPipelineGates(unittest.TestCase):
    def test_failed_gates_prevent_reads_commands_and_model_checks(self):
        cases = (({}, ("hard", "confirm")),
                 ({"dev": {"condition_1": False, "condition_2": True}}, ("hard", "confirm")),
                 ({"dev": {"condition_1": True, "condition_2": False}}, ("hard", "confirm")),
                 ({"dev": {"condition_1": True, "condition_2": True}}, ("confirm",)),
                 ({"dev": {"condition_1": True, "condition_2": True},
                   "hard": {"condition_3": False}}, ("confirm",)))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(pipeline, "HERE", root), \
                 patch.object(pipeline, "TASK", root), \
                 patch.object(pipeline, "check_freeze"), \
                 patch.object(pipeline.ar, "load_run") as read, \
                 patch.object(pipeline, "command") as command, \
                 patch.object(pipeline, "model_checks") as models:
                for experiment in ("G", "P"):
                    for evidence, sets in cases:
                        (root / f"{experiment}_mechanisms.json").write_text(json.dumps(evidence))
                        for set_name in sets:
                            with self.subTest(experiment=experiment, evidence=evidence, set_name=set_name):
                                with self.assertRaisesRegex(RuntimeError, "门槛"):
                                    pipeline.stalled_controls(experiment, set_name)
                                with self.assertRaisesRegex(RuntimeError, "门槛"):
                                    pipeline.evaluations(experiment, set_name, allow_confirm=True)
                                with self.assertRaisesRegex(RuntimeError, "门槛"):
                                    pipeline.evaluate(experiment, root / "unused.npz", set_name,
                                                      root / "unused", episodes=[0])
                read.assert_not_called()
                command.assert_not_called()
                models.assert_not_called()

    def test_confirm_requires_explicit_allow_even_after_both_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "G_mechanisms.json").write_text(json.dumps(passed_gates()))
            with patch.object(pipeline, "HERE", root), \
                 patch.object(pipeline, "TASK", root), \
                 patch.object(pipeline, "model_checks"), \
                 patch.object(pipeline, "stalled_controls") as controls, \
                 patch.object(pipeline.ar, "load_run") as read, \
                 patch.object(pipeline, "command") as command, \
                 patch.object(pipeline.ar, "ALLOW_CONFIRM", False):
                with self.assertRaisesRegex(RuntimeError, "--allow-confirm"):
                    pipeline.evaluations("G", "confirm")
                self.assertFalse(pipeline.ar.ALLOW_CONFIRM)
                self.assertFalse((root / "G_confirm_opened.json").exists())
                controls.assert_not_called()
                read.assert_not_called()
                command.assert_not_called()

    def test_direct_confirm_evaluation_requires_opening_token(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "P_mechanisms.json").write_text(json.dumps(passed_gates()))
            with patch.object(pipeline, "HERE", root), \
                 patch.object(pipeline, "TASK", root), \
                 patch.object(pipeline, "check_freeze"), \
                 patch.object(pipeline.ar, "load_run") as read, \
                 patch.object(pipeline, "command") as command, \
                 patch.object(pipeline.ar, "ALLOW_CONFIRM", False):
                with self.assertRaisesRegex(RuntimeError, "--allow-confirm"):
                    pipeline.evaluate("P", root / "model.npz", "confirm", root / "run", episodes=[0])
                self.assertFalse(pipeline.ar.ALLOW_CONFIRM)
                read.assert_not_called()
                command.assert_not_called()


class TestGPMemoryScheduling(unittest.TestCase):
    def test_memory_reserve_floor_requested_jobs_and_five_job_cap(self):
        reserve = 2 * pipeline.GB_MIB
        peak = 1024.
        self.assertEqual(pipeline.adaptive_jobs(reserve + peak, peak), 1)
        self.assertEqual(pipeline.adaptive_jobs(reserve + 4 * peak - 1, peak), 3)
        self.assertEqual(pipeline.adaptive_jobs(reserve + 100 * peak, peak, requested=100), 5)
        self.assertEqual(pipeline.adaptive_jobs(reserve + 100 * peak, peak, requested=2), 2)
        self.assertEqual(pipeline.adaptive_jobs(reserve + pipeline.LIMIT_MIB,
                                                pipeline.LIMIT_MIB), 1)
        with self.assertRaisesRegex(RuntimeError, "2 GiB"):
            pipeline.adaptive_jobs(reserve + peak - 1, peak)
        with self.assertRaises(RuntimeError):
            pipeline.adaptive_jobs(reserve + peak, peak, requested=0)

    def test_invalid_or_over_limit_peak_never_launches_a_job(self):
        for peak in (0., -1., float("nan"), float("inf"), pipeline.LIMIT_MIB + .01):
            with self.subTest(peak=peak), self.assertRaisesRegex(RuntimeError, "6 GiB"):
                pipeline.adaptive_jobs(1_000_000, peak)


class TestGPResume(unittest.TestCase):
    def test_completed_run_is_validated_and_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "results" / "complete"
            folder.mkdir(parents=True)
            summary = folder / "summary.txt"
            summary.write_text("existing result\n")
            model = root / "trained" / "model.npz"
            with patch.object(pipeline, "TASK", root), \
                 patch.object(pipeline, "check_freeze"), \
                 patch.object(pipeline, "event") as event, \
                 patch.object(pipeline, "command") as command, \
                 patch.object(pipeline, "validate_run", return_value=(None, {})) as validate:
                pipeline.evaluate("G", model, "dev", folder, episodes=[2, 7], save_every=10)
                validate.assert_called_once_with(folder, [2, 7], "trained/model.npz", True)
                command.assert_not_called()
                self.assertEqual(event.call_args.args[:3], ("G", "run", "skip_complete"))
            self.assertEqual(summary.read_text(), "existing result\n")

    def test_invalid_completed_run_is_preserved_and_not_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "results" / "invalid"
            folder.mkdir(parents=True)
            (folder / "summary.txt").write_text("invalid evidence\n")
            with patch.object(pipeline, "TASK", root), \
                 patch.object(pipeline, "check_freeze"), \
                 patch.object(pipeline, "event"), \
                 patch.object(pipeline, "command") as command, \
                 patch.object(pipeline, "validate_run", side_effect=RuntimeError("coverage mismatch")):
                with self.assertRaisesRegex(RuntimeError, "coverage mismatch"):
                    pipeline.evaluate("G", root / "model.npz", "dev", folder, episodes=[0])
                command.assert_not_called()
            self.assertTrue((folder / "summary.txt").exists())

    def test_interrupted_run_without_summary_is_deleted_before_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "results" / "interrupted"
            folder.mkdir(parents=True)
            (folder / "partial.csv").write_text("partial\n")

            def rerun(args, log):
                self.assertFalse(folder.exists(), "incomplete output must be removed before evaluator starts")
                self.assertEqual(args[args.index("--episodes") + 1], "2,7")
                self.assertIn("--no-early-stop", args)
                self.assertIn("--save-traj", args)
                self.assertEqual(args[args.index("--traj-every") + 1], 10)
                folder.mkdir(parents=True)
                (folder / "summary.txt").write_text("complete\n")

            with patch.object(pipeline, "TASK", root), \
                 patch.object(pipeline, "check_freeze"), \
                 patch.object(pipeline, "event") as event, \
                 patch.object(pipeline, "command", side_effect=rerun) as command, \
                 patch.object(pipeline, "validate_run", return_value=(None, {})) as validate:
                pipeline.evaluate("P", root / "model.npz", "dev", folder,
                                  episodes=[2, 7], save_every=10)
                command.assert_called_once()
                validate.assert_called_once_with(folder, [2, 7], "model.npz", True)
                self.assertEqual([call.args[2] for call in event.call_args_list],
                                 ["interrupted_directory_deleted", "start", "end"])
            self.assertFalse((folder / "partial.csv").exists())
            self.assertEqual((folder / "summary.txt").read_text(), "complete\n")

    def test_stage_rejects_changed_implementation_before_resuming(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            here = root / "results" / "step6"
            here.mkdir(parents=True)
            source = here / "implementation.py"
            source.write_text("SETTING = 1\n")
            shell = here / "run_expG.sh"
            shell.write_text("#!/bin/sh\n")
            shared = [root / name for name in ("train_policy.py", "bc_features.py",
                                               "policies/bc_policy.py", "policies/pd_layer.py")]
            for path in shared:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("SETTING = 1\n")
            fn = Mock()
            with patch.object(pipeline, "TASK", root), \
                 patch.object(pipeline, "HERE", here), \
                 patch.object(pipeline, "CODE_DIR", here), \
                 patch.object(pipeline, "SMOKE_ROOT", None), \
                 patch.object(pipeline, "check_freeze") as freeze, \
                 patch.object(pipeline, "status"), \
                 patch.object(pipeline, "event") as event:
                pipeline.stage("G", "cache", fn)
                marker = here / "completed" / "G_cache.json"
                original = marker.read_bytes()
                pipeline.stage("G", "cache", fn)
                fn.assert_called_once_with()
                self.assertEqual(event.call_args.args, ("G", "cache", "skip_complete"))
                for path in (source, shell, *shared):
                    with self.subTest(changed=path.name):
                        original_source = path.read_bytes()
                        path.write_bytes(original_source + b"# changed\n")
                        with self.assertRaisesRegex(RuntimeError, "哈希发生变化"):
                            pipeline.stage("G", "cache", fn)
                        path.write_bytes(original_source)
                with patch.object(pipeline, "SMOKE_ROOT", root):
                    with self.assertRaisesRegex(RuntimeError, "哈希发生变化"):
                        pipeline.stage("G", "cache", fn)
                fn.assert_called_once_with()
                self.assertEqual(freeze.call_count, 9)
                self.assertEqual(marker.read_bytes(), original)

    def test_failed_freeze_prevents_stage_execution_and_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            here = Path(directory)
            fn = Mock()
            with patch.object(pipeline, "HERE", here), \
                 patch.object(pipeline, "check_freeze", side_effect=RuntimeError("freeze changed")), \
                 patch.object(pipeline, "status") as status:
                with self.assertRaisesRegex(RuntimeError, "freeze changed"):
                    pipeline.stage("P", "dev", fn)
                fn.assert_not_called()
                status.assert_not_called()
            self.assertFalse((here / "completed" / "P_dev.json").exists())

    def test_freeze_checks_plan_frozen_record_and_report_section(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            here = root / "results" / "step6"
            here.mkdir(parents=True)
            header = "STEP 6 FROZEN SECTION\n"
            section = header + "preregistered thresholds\n"
            plan = root / "STEP6_PLAN_PROMPT.md"
            frozen = here / "prereg_GP_frozen.txt"
            report = root / "report.txt"
            plan.write_text("synthetic plan\n")
            frozen.write_text("metadata\n" + section)
            report.write_text("step 5 results\n" + section)
            manifest = {"plan_sha256": common.sha(plan), "frozen_sha256": common.sha(frozen),
                        "section_sha256": common.hashlib.sha256(section.encode()).hexdigest(),
                        "report_header": header}
            (here / "prereg_GP_frozen.json").write_text(json.dumps(manifest))
            with patch.object(common, "SOURCE_TASK", root), \
                 patch.object(common, "SOURCE_HERE", here), patch("builtins.print"):
                self.assertEqual(common.check_freeze(), manifest)
                for path in (plan, frozen, report):
                    with self.subTest(path=path.name):
                        original = path.read_bytes()
                        path.write_bytes(original + b"changed\n")
                        if path == report:
                            # Content following the frozen section is allowed;
                            # changing a threshold inside the section is not.
                            path.write_text(report.read_text().replace("thresholds", "modified"))
                        with self.assertRaisesRegex(RuntimeError, "发生变化"):
                            common.check_freeze()
                        path.write_bytes(original)


class TestGPDecisionBoundaries(unittest.TestCase):
    def test_pilot_uses_smallest_delta_in_inclusive_five_pp_band(self):
        choice = common.pilot_choice({9: [.70, .70], 27: [.72, .72], 81: [.75, .75]}, [.5, .5])
        self.assertEqual(choice["delta_star"], 9)
        choice = common.pilot_choice({9: [.699999, .699999], 27: [.72, .72], 81: [.75, .75]}, [.5, .5])
        self.assertEqual(choice["delta_star"], 27)
        with self.assertRaisesRegex(ValueError, "9, 27 and 81"):
            common.pilot_choice({9: [.5], 27: [.5]}, [.5])

    def test_pilot_stops_strictly_below_ten_pp_loss(self):
        for rate, should_continue in ((.4, True), (.399999, False)):
            with self.subTest(rate=rate):
                choice = common.pilot_choice({9: [rate] * 2, 27: [rate] * 2, 81: [rate] * 2}, [.5, .5])
                self.assertEqual(choice["continue"], should_continue)

    def test_dev_requires_strictly_positive_both_cis_and_inclusive_risk_cap(self):
        for key in ("lo", "welch_lo"):
            for value, expected in ((0., False), (-1e-8, False), (1e-8, True)):
                with self.subTest(key=key, value=value):
                    out = dev_metrics()
                    out["success"][key] = value
                    self.assertEqual(apply_conditions(out, "dev", "G")["condition_1"], expected)
        for diff, expected in ((.03, True), (.030001, False)):
            with self.subTest(risk=diff):
                out = dev_metrics()
                out["risk"]["diff"] = diff
                self.assertEqual(apply_conditions(out, "dev", "P")["condition_2"], expected)

    def test_mirror_worse_requires_both_upper_bounds_strictly_negative(self):
        for upper, welch_upper, expected in ((-.001, -.001, True), (0., -.001, False),
                                             (-.001, 0., False), (.001, -.001, False)):
            with self.subTest(upper=upper, welch_upper=welch_upper):
                out = dev_metrics()
                out["success"].update(hi=upper, welch_hi=welch_upper)
                self.assertEqual(apply_conditions(out, "dev", "G")["mirror_worse"], expected)

    def test_mechanism_equivalence_band_is_inclusive_in_both_cis(self):
        for experiment, name, metric_name in (("G", "MG3", "extended_not_reached"),
                                               ("P", "MP3", "circling")):
            out = dev_metrics()
            out[metric_name].update(lo=-.07, hi=.07, welch_lo=-.07, welch_hi=.07)
            self.assertTrue(apply_conditions(copy.deepcopy(out), "dev", experiment)["mechanism_predictions"][name])
            for key, value in (("lo", -.070001), ("hi", .070001),
                               ("welch_lo", -.070001), ("welch_hi", .070001)):
                with self.subTest(experiment=experiment, key=key):
                    changed = copy.deepcopy(out)
                    changed[metric_name][key] = value
                    self.assertFalse(apply_conditions(changed, "dev", experiment)["mechanism_predictions"][name])

    def test_hard_and_confirm_positive_boundaries_follow_frozen_rule(self):
        for diff, expected in ((0., False), (-1e-8, False), (1e-8, True)):
            with self.subTest(set_name="hard", diff=diff):
                out = {"success": metric(diff=diff)}
                self.assertEqual(apply_conditions(out, "hard", "G")["condition_3"], expected)
        for diff, lo, expected in ((.01, .001, True), (.01, 0., False), (0., .001, False),
                                   (.01, -.001, False)):
            with self.subTest(set_name="confirm", diff=diff, lo=lo):
                out = {"success": metric(diff=diff, lo=lo, welch_lo=-.1)}
                self.assertEqual(apply_conditions(out, "confirm", "P")["condition_4"], expected)


if __name__ == "__main__":
    unittest.main(failfast=True)
