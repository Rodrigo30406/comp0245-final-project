"""Step 7 formula, DEV replay and orchestration checks; never opens data/test.

Run from the real module entry point with PYTHONDONTWRITEBYTECODE=1.  The DEV
replay reads only existing rollout results and existing no-stall replacements.
All temporary fixtures stay below results/step7/smoke.
"""
import ast
import hashlib
import json
import math
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy import stats

TASK = Path(__file__).resolve().parents[1]
HERE = TASK / "results" / "step7"
sys.path.insert(0, str(TASK / "results" / "step6"))
sys.path.insert(0, str(HERE))
import gp_common as gc
import PC_mechanisms as pc
import pc_pipeline as pipeline
import pc_freeze as freezing


def metric(diff=.1, lo=.01, hi=.2, welch_lo=.01, welch_hi=.2):
    return {"diff": diff, "lo": lo, "hi": hi,
            "welch_lo": welch_lo, "welch_hi": welch_hi}


def decision_inputs():
    return ({"success": metric()}, {"success": metric()},
            {"success": metric(), "risk": {"diff": .03}})


def temporary_fixture():
    smoke = HERE / "smoke"
    smoke.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="unittest_", dir=smoke)


def pipeline_fixture(directory):
    root = Path(directory)
    here = root / "results" / "step7"
    here.mkdir(parents=True)
    model = root / "trained" / "ego-novw-trig-pd_s0.npz"
    model.parent.mkdir()
    model.write_bytes(b"synthetic model fixture, never evaluated")
    folder = root / "results" / f"{model.stem}_hard"
    return root, here, model, folder


class TestCombinedStatistics(unittest.TestCase):
    def test_weighted_bootstrap_and_four_variance_welch_known_answer(self):
        # HARD SE^2 = 1 + 1; CONFIRM SE^2 = 4 + 4.  Each arm has two
        # seeds, so the four variance terms have one degree of freedom each.
        hard = {"a": 1., "b": 2., "diff": 1.,
                "draws": np.array([0., 1., 2., 3., 4.]),
                "per_seed_a": np.array([0., 2.]),
                "per_seed_b": np.array([1., 3.])}
        confirm = {"a": 2., "b": 4., "diff": 2.,
                   "draws": np.array([4., 3., 2., 1., 0.]),
                   "per_seed_a": np.array([0., 4.]),
                   "per_seed_b": np.array([2., 6.])}
        got = pc.combine_statistics(hard, confirm, n_hard=200, n_confirm=300)
        np.testing.assert_allclose(got["draws"], [2.4, 2.2, 2., 1.8, 1.6], atol=0, rtol=1e-15)
        self.assertAlmostEqual(got["a"], 1.6, places=14)
        self.assertAlmostEqual(got["b"], 3.2, places=14)
        self.assertAlmostEqual(got["diff"], 1.6, places=14)
        self.assertAlmostEqual(got["lo"], 1.62, places=14)
        self.assertAlmostEqual(got["hi"], 2.38, places=14)
        np.testing.assert_allclose(got["welch_variance_terms"], [.16, .16, 1.44, 1.44],
                                   atol=0, rtol=1e-14)
        self.assertAlmostEqual(got["welch_se"], math.sqrt(3.2), places=14)
        self.assertAlmostEqual(got["welch_df"], 100 / 41, places=14)
        q = stats.t.ppf(.975, 100 / 41)
        self.assertAlmostEqual(got["welch_lo"], 1.6 - q * math.sqrt(3.2), places=13)
        self.assertAlmostEqual(got["welch_hi"], 1.6 + q * math.sqrt(3.2), places=13)

    def test_zero_seed_variance_gives_degenerate_combined_interval(self):
        def constant(a, b):
            return {"a": a, "b": b, "diff": b - a,
                    "draws": np.full(8, b - a),
                    "per_seed_a": np.full(5, a), "per_seed_b": np.full(5, b)}
        got = pc.combine_statistics(constant(0., 1.), constant(0., 1.))
        for key in ("diff", "lo", "hi", "welch_lo", "welch_hi"):
            self.assertEqual(got[key], 1.)
        self.assertEqual(got["welch_se"], 0.)

    def test_matrix_path_reuses_original_paired_bootstrap_draws_exactly(self):
        A = np.array([[0., 1., 0., 1., 1., 0.], [1., 0., 1., 0., 0., 1.]])
        B = np.array([[1., 1., 0., 1., 0., 1.], [0., 1., 1., 1., 1., 0.]])
        got = pc.matrix_difference(A, B)
        original = gc.ar.paired_diff(A, B)
        np.testing.assert_array_equal(got["draws"], original["draws"])
        for key, old_key in (("a", "mean_a"), ("b", "mean_b"), ("diff", "diff"),
                             ("lo", "lo"), ("hi", "hi"), ("p_boot", "p")):
            self.assertEqual(got[key], original[old_key])


class TestDEVBitwiseReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # No prepare/reference creation is needed for success: reuse the exact
        # archived C3 stalled-row substitutions and raw P DEV rollout rows.
        cls.A, cls.notes = gc.merged_old(gc.raw_runs(gc.OLD, "dev", gc.SEEDS["dev"]))
        cls.B = gc.raw_runs(gc.MODELS["P"], "dev", gc.SEEDS["dev"])
        cls.episodes, _ = gc.ar.paired_episodes(cls.A + cls.B)
        cls.YA = gc.ar.metric_matrix(cls.A, cls.episodes, "success")
        cls.YB = gc.ar.metric_matrix(cls.B, cls.episodes, "success")
        cls.archived = json.loads((TASK / "results" / "step6" / "P_mechanisms.json").read_text())["dev"]

    def test_dev_975_matches_archived_step6_success_exactly(self):
        got = pc.single_set(self.A, self.B, metric="success", episodes=self.episodes, level=.975)
        self.assertEqual(len(self.episodes), self.archived["n_episodes"])
        self.assertEqual(self.notes, self.archived["replacement_notes"])
        for key, expected in self.archived["success"].items():
            with self.subTest(field=key):
                self.assertEqual(gc.ar._clean(got[key]), expected)

    def test_dev_95_matches_step6_original_function_exactly(self):
        boot = gc.ar.Bootstrap([len(self.A), len(self.B)], len(self.episodes),
                               gc.ar.N_BOOT, gc.ar.BOOT_SEED)
        expected = gc.difference(self.YA, self.YB, level=.95, boot=boot)
        got = pc.single_set(self.A, self.B, metric="success", episodes=self.episodes,
                            level=.95, boot=boot)
        for key, value in expected.items():
            with self.subTest(field=key):
                self.assertEqual(gc.ar._clean(got[key]), gc.ar._clean(value))
        original_draws = gc.ar.paired_diff(self.YA, self.YB, boot=boot)["draws"]
        np.testing.assert_array_equal(got["draws"], original_draws)


class TestPCComparisonSeedCompatibility(unittest.TestCase):
    def test_amend1_keeps_primary_statistical_functions_identical(self):
        before = ast.parse((HERE / "PC_mechanisms.before_amend1.py.txt").read_text())
        after = ast.parse((HERE / "PC_mechanisms.py").read_text())
        primary = ("matrix_difference", "single_set", "combine_statistics", "conditions")
        def functions(module):
            return {node.name: ast.dump(node, include_attributes=False)
                    for node in module.body if isinstance(node, ast.FunctionDef)}
        old_functions, new_functions = functions(before), functions(after)
        for name in primary:
            with self.subTest(function=name):
                self.assertEqual(old_functions[name], new_functions[name])

    @unittest.skipUnless((HERE / "smoke" / "confirmed").exists(),
                         "needs the step-7 smoke fixtures (not in git; results/step7/pc_smoke.py)")
    def test_actual_complete_comparison_accepts_independent_stream_seed(self):
        # The real descriptive comparison performs seed + 1 + k in its
        # stratification path.  A SeedSequence cannot be passed through there.
        # Reuse synthetic rollouts only; write every output in a fresh fixture.
        root = HERE / "smoke" / "confirmed"
        with temporary_fixture() as directory:
            output = Path(directory) / "comparison"
            output.mkdir()
            seed_h = np.random.SeedSequence(gc.ar.BOOT_SEED).spawn(2)[0]
            got, lines = pc._compare_set(root, output, "hard", seed_h)
            self.assertIsInstance(got["compare_boot_seed"], int)
            expected_seed = int(np.random.SeedSequence(gc.ar.BOOT_SEED).spawn(2)[0]
                                .generate_state(1, dtype=np.uint32)[0])
            self.assertEqual(got["compare_boot_seed"], expected_seed)
            self.assertTrue((output / "PC_compare_hard.txt").is_file())
            self.assertTrue((output / "PC_compare_hard.json").is_file())
            self.assertTrue(lines)
            # The original preregistered calculation is still evaluated with
            # exactly the original independent HARD stream and original helper.
            A, _ = pc.merged_controls(pc._raw_runs(root, pc.OLD, "hard", pc.SEEDS["hard"]), root)
            B = pc._raw_runs(root, pc.NEW, "hard", pc.SEEDS["hard"])
            episodes = list(range(200))
            seed_original = np.random.SeedSequence(gc.ar.BOOT_SEED).spawn(2)[0]
            boot = gc.ar.Bootstrap([5, 5], 200, gc.ar.N_BOOT, seed_original)
            YA, YB = (gc.ar.metric_matrix(runs, episodes, "success") for runs in (A, B))
            original = gc.difference(YA, YB, level=.95, boot=boot)
            for key, value in original.items():
                with self.subTest(primary_field=key):
                    self.assertEqual(gc.ar._clean(got["success"][key]), gc.ar._clean(value))
            np.testing.assert_array_equal(got["success"]["draws"], gc.ar.paired_diff(YA, YB, boot=boot)["draws"])


class TestPCDecisionBoundaries(unittest.TestCase):
    def test_all_three_conditions_confirm_at_inclusive_risk_limit(self):
        got = pc.conditions(*decision_inputs())
        for key in ("condition_1", "condition_2", "condition_3", "confirmed"):
            self.assertTrue(got[key])
        self.assertFalse(got["mirror_worse"])
        self.assertEqual(got["verdict"], "P 在留出集上得到确认")
        self.assertIn("助教", got["main_model"])
        self.assertIn("C3", got["main_model"])

    def test_both_lower_bounds_must_be_strictly_positive(self):
        for key in ("lo", "welch_lo"):
            for value in (0., -1e-8):
                with self.subTest(bound=key, value=value):
                    hard, confirm, combined = decision_inputs()
                    combined["success"][key] = value
                    got = pc.conditions(hard, confirm, combined)
                    self.assertFalse(got["condition_1"])
                    self.assertFalse(got["confirmed"])
                    self.assertEqual(got["verdict"], "没有得到确认")
                    self.assertEqual(got["main_model"], "C3")

    def test_each_set_point_estimate_must_be_strictly_positive(self):
        for which in (0, 1):
            for value in (0., -1e-8):
                with self.subTest(set_index=which, value=value):
                    data = decision_inputs()
                    data[which]["success"]["diff"] = value
                    got = pc.conditions(*data)
                    self.assertFalse(got["condition_2"])
                    self.assertFalse(got["confirmed"])

    def test_risk_above_three_pp_fails(self):
        for value in (.030001, np.nextafter(.03, np.inf)):
            with self.subTest(risk=value):
                hard, confirm, combined = decision_inputs()
                combined["risk"]["diff"] = value
                got = pc.conditions(hard, confirm, combined)
                self.assertFalse(got["condition_3"])
                self.assertFalse(got["confirmed"])

    def test_mirror_requires_both_upper_bounds_strictly_negative(self):
        for hi, t_hi, expected in ((-.001, -.001, True), (0., -.001, False),
                                    (-.001, 0., False), (.001, -.001, False)):
            with self.subTest(hi=hi, welch_hi=t_hi):
                hard, confirm, combined = decision_inputs()
                combined["success"].update(diff=-.1, lo=-.2, hi=hi,
                                           welch_lo=-.2, welch_hi=t_hi)
                got = pc.conditions(hard, confirm, combined)
                self.assertEqual(got["mirror_worse"], expected)
                self.assertFalse(got["confirmed"])
                self.assertEqual(got["main_model"], "C3")


class TestPCResumeAndAuthorization(unittest.TestCase):
    def test_completed_run_is_validated_and_skipped_unchanged(self):
        with temporary_fixture() as directory:
            root, here, model, folder = pipeline_fixture(directory)
            folder.mkdir()
            summary = folder / "summary.txt"
            summary.write_text("existing completed result\n")
            with patch.object(pipeline, "TASK", root), patch.object(pipeline, "HERE", here), \
                 patch.object(pipeline, "check_freeze") as freeze, \
                 patch.object(pipeline, "event") as event, \
                 patch.object(pipeline, "command") as command, \
                 patch.object(pipeline, "validate_run", return_value=(None, {})) as validate:
                pipeline.evaluate(model, "hard", folder, [2, 7], state_dir=here)
                freeze.assert_called_once_with()
                validate.assert_called_once_with(folder, [2, 7], f"trained/{model.name}")
                command.assert_not_called()
                self.assertEqual(event.call_args.args[1:3], ("run", "skip_complete"))
            self.assertEqual(summary.read_text(), "existing completed result\n")
            self.assertFalse((here / "owned_runs.json").exists())

    def test_invalid_completed_run_is_preserved_without_rerun(self):
        with temporary_fixture() as directory:
            root, here, model, folder = pipeline_fixture(directory)
            folder.mkdir()
            summary = folder / "summary.txt"
            summary.write_text("invalid existing result\n")
            with patch.object(pipeline, "TASK", root), patch.object(pipeline, "HERE", here), \
                 patch.object(pipeline, "check_freeze"), patch.object(pipeline, "event"), \
                 patch.object(pipeline, "command") as command, \
                 patch.object(pipeline, "validate_run", side_effect=RuntimeError("coverage mismatch")):
                with self.assertRaisesRegex(RuntimeError, "coverage mismatch"):
                    pipeline.evaluate(model, "hard", folder, [2, 7], state_dir=here)
                command.assert_not_called()
            self.assertEqual(summary.read_text(), "invalid existing result\n")

    def test_unowned_interrupted_directory_is_preserved(self):
        with temporary_fixture() as directory:
            root, here, model, folder = pipeline_fixture(directory)
            folder.mkdir()
            partial = folder / "partial.csv"
            partial.write_text("pre-existing incomplete output\n")
            with patch.object(pipeline, "TASK", root), patch.object(pipeline, "HERE", here), \
                 patch.object(pipeline, "check_freeze"), patch.object(pipeline, "event"), \
                 patch.object(pipeline, "command") as command:
                with self.assertRaisesRegex(RuntimeError, "refusing to delete"):
                    pipeline.evaluate(model, "hard", folder, [2, 7], state_dir=here)
                command.assert_not_called()
            self.assertEqual(partial.read_text(), "pre-existing incomplete output\n")
            self.assertFalse((here / "owned_runs.json").exists())

    def test_new_run_registered_before_launch_and_owned_interruption_resumes(self):
        with temporary_fixture() as directory:
            root, here, model, folder = pipeline_fixture(directory)
            calls = []

            def evaluate_fixture(args, log):
                registry = json.loads((here / "owned_runs.json").read_text())
                key = str(folder.relative_to(root))
                self.assertEqual(registry[key]["episodes"], [2, 7])
                self.assertEqual(registry[key]["model_sha256"], gc.sha(model))
                self.assertFalse(folder.exists())
                self.assertIn("--no-early-stop", args)
                self.assertEqual(args[args.index("--episodes") + 1], "2,7")
                self.assertEqual(args[args.index("--out-dir") + 1], key)
                self.assertTrue(Path(log).is_relative_to(here))
                folder.mkdir()
                calls.append(args)
                if len(calls) == 1:
                    (folder / "partial.csv").write_text("interrupted fixture\n")
                    raise RuntimeError("simulated evaluator interruption")
                (folder / "summary.txt").write_text("complete fixture\n")

            with patch.object(pipeline, "TASK", root), patch.object(pipeline, "HERE", here), \
                 patch.object(pipeline, "check_freeze"), patch.object(pipeline, "event") as event, \
                 patch.object(pipeline, "command", side_effect=evaluate_fixture), \
                 patch.object(pipeline, "validate_run", return_value=(None, {})) as validate:
                with self.assertRaisesRegex(RuntimeError, "simulated evaluator interruption"):
                    pipeline.evaluate(model, "hard", folder, [7, 2], state_dir=here)
                self.assertTrue((folder / "partial.csv").exists())
                pipeline.evaluate(model, "hard", folder, [2, 7], state_dir=here)
                validate.assert_called_once_with(folder, [2, 7], f"trained/{model.name}")
                statuses = [call.args[2] for call in event.call_args_list]
                self.assertEqual(statuses, ["start", "interrupted_directory_deleted", "start", "end"])
            self.assertFalse((folder / "partial.csv").exists())
            self.assertEqual((folder / "summary.txt").read_text(), "complete fixture\n")

    def test_freeze_failure_prevents_any_stage_or_evaluator_work(self):
        with temporary_fixture() as directory:
            root, here, model, folder = pipeline_fixture(directory)
            with patch.object(pipeline, "TASK", root), patch.object(pipeline, "HERE", here), \
                 patch.object(pipeline, "check_freeze", side_effect=RuntimeError("freeze changed")), \
                 patch.object(pipeline, "event") as event, \
                 patch.object(pipeline, "evaluations") as evaluations, \
                 patch.object(pipeline, "command") as command:
                with self.assertRaisesRegex(RuntimeError, "freeze changed"):
                    pipeline.run_stage("hard")
                with self.assertRaisesRegex(RuntimeError, "freeze changed"):
                    pipeline.evaluate(model, "hard", folder, [0], state_dir=here)
                event.assert_not_called()
                evaluations.assert_not_called()
                command.assert_not_called()
            self.assertFalse(folder.exists())
            self.assertFalse((here / "completed").exists())

    def test_confirm_requires_explicit_flag_and_prior_opening_token(self):
        with temporary_fixture() as directory:
            root, here, model, _ = pipeline_fixture(directory)
            folder = root / "results" / f"{model.stem}_confirm"
            with patch.object(pipeline, "TASK", root), patch.object(pipeline, "HERE", here), \
                 patch.object(pipeline, "check_freeze", return_value={}), \
                 patch.object(pipeline.ar, "ALLOW_CONFIRM", False), \
                 patch.object(pipeline.ar, "load_run") as load, \
                 patch.object(pipeline, "command") as command:
                with self.assertRaisesRegex(RuntimeError, "--allow-confirm"):
                    pipeline.evaluate(model, "confirm", folder, [0], state_dir=here)
                with self.assertRaisesRegex(RuntimeError, "opened"):
                    pipeline.evaluate(model, "confirm", folder, [0], state_dir=here, allow_confirm=True)
                self.assertFalse(pipeline.ar.ALLOW_CONFIRM)
                self.assertFalse((here / "PC_confirm_opened.json").exists())
                load.assert_not_called()
                command.assert_not_called()

    def test_confirm_opening_is_one_persistent_frozen_token(self):
        with temporary_fixture() as directory:
            root, here, _, _ = pipeline_fixture(directory)
            frozen = {"plan_sha256": "plan-hash", "frozen_sha256": "frozen-hash"}
            with patch.object(pipeline, "TASK", root), patch.object(pipeline, "HERE", here), \
                 patch.object(pipeline, "check_freeze", return_value=frozen), \
                 patch.object(pipeline, "event") as event, \
                 patch.object(pipeline.ar, "ALLOW_CONFIRM", False):
                first = pipeline.confirm_token(here, True, create=True)
                token = here / "PC_confirm_opened.json"
                original = token.read_bytes()
                self.assertEqual(pipeline.confirm_token(here, True, create=True), first)
                self.assertEqual(token.read_bytes(), original)
                self.assertTrue(pipeline.ar.ALLOW_CONFIRM)
                event.assert_called_once()
                with patch.object(pipeline, "check_freeze", return_value={**frozen, "frozen_sha256": "changed"}):
                    with self.assertRaisesRegex(RuntimeError, "freeze mismatch"):
                        pipeline.confirm_token(here, True, create=True)
                self.assertEqual(token.read_bytes(), original)

    def test_confirm_has_no_hard_result_gate(self):
        with patch.object(pipeline, "confirm_token") as token, \
             patch.object(pipeline, "evaluate") as evaluate:
            pipeline.evaluations("confirm", allow_confirm=True)
            token.assert_called_once_with(pipeline.HERE, True, create=True)
            self.assertEqual(evaluate.call_count, 5)
            for call, seed in zip(evaluate.call_args_list, range(100, 105)):
                self.assertEqual(call.args[0].name, f"ego-novw-trig-pd_s{seed}.npz")
                self.assertEqual(call.args[1], "confirm")
                self.assertEqual(call.args[3], list(range(300)))
                self.assertTrue(call.kwargs["allow_confirm"])

    def test_controls_reuse_existing_nostall_with_no_step6_gate(self):
        with temporary_fixture() as directory:
            root, here, _, _ = pipeline_fixture(directory)
            name = "ego-novw-trig_s0_hard"
            candidate = root / "results" / "step5" / "nostall" / name
            candidate.mkdir(parents=True)
            (candidate / "summary.txt").write_text("existing replacement fixture\n")
            rows = [{"episode": e, "stop_reason": "stalled" if e == 2 else "timeout"}
                    for e in range(3)]
            run = gc.ar.Run(str(root / "results" / name), rows, ["episode", "stop_reason"], {},
                            {"warnings": [], "no_early_stop": False,
                             "policy_arg": "trained/ego-novw-trig_s0.npz"})
            with patch.object(pipeline, "TASK", root), patch.object(pipeline, "HERE", here), \
                 patch.object(pipeline, "SEEDS", {"hard": [0]}), \
                 patch.object(pipeline, "COUNTS", {"hard": 3}), \
                 patch.object(pipeline.ar, "load_run", return_value=run), \
                 patch.object(pipeline, "validate_run", return_value=(None, {})) as validate, \
                 patch.object(pipeline, "evaluate") as evaluate:
                notes = pipeline.stalled_controls("hard")
                validate.assert_called_once_with(candidate, [2], "trained/ego-novw-trig_s0.npz")
                evaluate.assert_not_called()
                self.assertEqual(notes[0]["source"], str(candidate.relative_to(root)))
                self.assertEqual(notes[0]["stalled"], 1)


class TestPCFreeze(unittest.TestCase):
    def test_freeze_preserves_original_report_and_checks_frozen_hashes(self):
        with temporary_fixture() as directory:
            root = Path(directory)
            output = root / "results" / "step7"
            output.mkdir(parents=True)
            plan = root / "STEP7_PLAN_PROMPT.md"
            plan.write_text("synthetic plan\n" + freezing.SECTION_HEADING
                            + "\n\n- synthetic preregistered threshold\n\n## 4. Deliverables\n")
            report = root / "report.txt"
            original_report = ("some earlier [References] wording\nprior results\n"
                               + freezing.REFERENCE_MARKER + "\nsource list\n").encode()
            report.write_bytes(original_report)
            for relative in (*freezing.IMPLEMENTATION_PATHS, *freezing._frozen_input_paths()):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"# synthetic implementation fixture\n")
            freezing.capture_baseline(repo=root)
            frozen = freezing.freeze(repo=root)
            self.assertEqual(report.read_bytes(), original_report)
            self.assertEqual(freezing.check_freeze(repo=root), frozen)
            self.assertEqual(freezing.freeze(repo=root), frozen)
            section = freezing._section(plan.read_bytes())
            report_copy = (output / "report_PC_frozen.txt").read_bytes()
            expected = original_report.replace(freezing.REFERENCE_MARKER.encode(),
                                               b"\n" + freezing._report_section(section)
                                               + b"\n" + freezing.REFERENCE_MARKER.encode(), 1)
            self.assertEqual(report_copy, expected)
            self.assertIn(section, (output / "prereg_PC_frozen.txt").read_bytes())
            changed_paths = (plan, report, output / "prereg_PC_frozen.txt",
                             output / "prereg_PC_time.txt", root / freezing.IMPLEMENTATION_PATHS[0],
                             root / freezing.FROZEN_SOURCE_INPUTS[0])
            for path in changed_paths:
                with self.subTest(changed=path.name):
                    original = path.read_bytes()
                    path.write_bytes(original + b"changed\n")
                    try:
                        with self.assertRaisesRegex(RuntimeError, "freeze verification failed"):
                            freezing.check_freeze(repo=root)
                    finally:
                        path.write_bytes(original)
            self.assertEqual(freezing.check_freeze(repo=root), frozen)

    def test_amendment_accepts_audited_code_change_and_rejects_changed_anchors(self):
        with temporary_fixture() as directory:
            root = Path(directory)
            output = root / "results" / "step7"
            output.mkdir(parents=True)
            plan = root / "STEP7_PLAN_PROMPT.md"
            plan.write_text("synthetic plan\n" + freezing.SECTION_HEADING
                            + "\n\n- synthetic preregistered threshold\n\n## 4. Deliverables\n")
            (root / "report.txt").write_text("synthetic report\n" + freezing.REFERENCE_MARKER + "\nrefs\n")
            for relative in (*freezing.IMPLEMENTATION_PATHS, *freezing._frozen_input_paths()):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"# synthetic original implementation\n")
            freezing.capture_baseline(repo=root)
            frozen = freezing.freeze(repo=root)
            anchored_files = ("prereg_PC_frozen.json", "prereg_PC_frozen.txt", "prereg_PC_time.txt",
                              "report_PC_section.txt", "report_PC_frozen.txt")
            original_artifacts = {name: (output / name).read_bytes() for name in anchored_files}
            source_name = "results/step7/PC_mechanisms.py"
            source = root / source_name
            backup = output / "PC_mechanisms.before_amend1.py.txt"
            backup.write_bytes(source.read_bytes())
            source.write_bytes(b"# synthetic repaired descriptive seed adapter\n")
            with self.assertRaisesRegex(RuntimeError, "implementation SHA256"):
                freezing.check_freeze(repo=root)
            before = frozen["implementation_sha256"]
            after = {name: freezing._file_sha(root / name) for name in before}
            amendment = {
                "schema_version": 1, "amendment_id": "PC-amend1",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "reason": "Synthetic descriptive seed adapter compatibility repair; primary rules unchanged",
                "anchors": {key: frozen[key] for key in ("frozen_sha256", "plan_sha256", "section3_sha256")},
                "before_implementation_sha256": before,
                "after_implementation_sha256": after,
                "backups": {source_name: {"path": str(backup.relative_to(root)), "sha256": before[source_name]}},
            }

            def save_record(record):
                raw = (json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
                (output / "implementation_PC_amend1.json").write_bytes(raw)
                (output / "implementation_PC_amend1.sha256").write_text(hashlib.sha256(raw).hexdigest() + "\n")

            save_record(amendment)
            effective = freezing.check_freeze(repo=root)
            self.assertEqual(effective["effective_implementation_sha256"], after)
            self.assertEqual(effective["implementation_sha256"], before)
            self.assertEqual(effective["implementation_amendment"]["changed_sources"], [source_name])
            changes = (
                ("anchor", lambda value: value["anchors"].__setitem__("section3_sha256", "0" * 64)),
                ("before map", lambda value: value["before_implementation_sha256"].__setitem__(source_name, "0" * 64)),
                ("after paths", lambda value: value["after_implementation_sha256"].pop(source_name)),
                ("backup path", lambda value: value["backups"][source_name].__setitem__("path", "../outside.txt")),
            )
            for name, mutate in changes:
                with self.subTest(invalid_amendment=name):
                    invalid = json.loads(json.dumps(amendment))
                    mutate(invalid)
                    save_record(invalid)
                    with self.assertRaisesRegex(RuntimeError, "amend1"):
                        freezing.check_freeze(repo=root)
            save_record(amendment)
            backup_bytes = backup.read_bytes()
            backup.write_bytes(backup_bytes + b"changed\n")
            with self.assertRaisesRegex(RuntimeError, "backup bytes"):
                freezing.check_freeze(repo=root)
            backup.write_bytes(backup_bytes)
            plan_bytes = plan.read_bytes()
            plan.write_bytes(plan_bytes.replace(b"threshold", b"different threshold"))
            with self.assertRaisesRegex(RuntimeError, "PC freeze verification failed"):
                freezing.check_freeze(repo=root)
            plan.write_bytes(plan_bytes)
            self.assertEqual(freezing.check_freeze(repo=root)["effective_implementation_sha256"], after)
            for name, original in original_artifacts.items():
                self.assertEqual((output / name).read_bytes(), original)


if __name__ == "__main__":
    unittest.main(failfast=True)
