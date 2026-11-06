"""Parallel episodes preserve data and recover from worker failure/interruption."""

import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robot_learning.data.storage import read_json, sha256, write_json


class ParallelCollectionTests(unittest.TestCase):
    def command(self, *args):
        return [sys.executable, str(ROOT / "scripts/collect_dataset.py"), *map(str, args)]

    def invoke(self, *args, expected=0):
        result = subprocess.run(self.command(*args), cwd=ROOT.parent, capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def plan(self, temporary, *, steps=5000):
        spec = temporary / "spec.json"
        write_json(spec, {"episodes": 2, "max_steps": steps})
        self.invoke("--config", spec, "--output-dir", temporary / "data", "--plan-only")
        return next((temporary / "data").iterdir())

    def wait_for_active_workers(self, directory, process):
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            self.assertIsNone(process.poll(), "Collector ended before workers could be observed")
            paths = list((directory / "executions").glob("*.json"))
            if paths:
                execution = read_json(paths[0])
                episodes = execution["episodes"]
                if len(episodes) == 2 and all((directory / "episodes" / f"{e['episode_id']}.json").exists() for e in episodes):
                    return execution
            time.sleep(.02)
        self.fail("Timed out waiting for two active episode workers")

    def test_parallel_arrays_match_sequential_and_resume_can_change_workers(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            spec = temporary / "spec.json"
            write_json(spec, {"episodes": 4, "max_steps": 500, "initial_base": {
                "mode": "uniform_xy", "x": [-1, 1], "y": [-1, 1], "yaw": [-.2, .2]}})
            for mode, workers in (("sequential", 1), ("parallel", 2)):
                self.invoke("--config", spec, "--output-dir", temporary / mode, "--workers", workers)
            sequential = next((temporary / "sequential").iterdir())
            parallel = next((temporary / "parallel").iterdir())
            self.assertEqual(read_json(sequential / "scenarios.json"), read_json(parallel / "scenarios.json"))
            self.assertEqual(read_json(sequential / "config.json"), read_json(parallel / "config.json"))
            for file in (parallel / "episodes").glob("*.npz"):
                with np.load(file) as actual, np.load(sequential / "episodes" / file.name) as expected:
                    for key in expected.files:
                        np.testing.assert_array_equal(actual[key], expected[key], err_msg=f"{file.name}: {key}")
            execution = read_json(next((parallel / "executions").glob("*.json")))
            self.assertEqual(execution["peak_active_workers"], 2)
            self.assertEqual(len(execution["episodes"]), 4)
            self.assertLess(execution["episodes"][1]["started_at"], execution["episodes"][0]["finished_at"])
            before = {p.name: sha256(p) for p in (parallel / "episodes").glob("*.npz")}
            self.invoke("--resume", parallel, "--workers", 3)
            self.assertEqual(before, {p.name: sha256(p) for p in (parallel / "episodes").glob("*.npz")})
            runs = [read_json(p) for p in (parallel / "executions").glob("*.json")]
            resumed = next(r for r in runs if r["workers"] == 3)
            self.assertEqual(resumed["episodes"], [])

    def test_worker_crash_does_not_lose_other_episode_and_can_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = self.plan(Path(temporary))
            process = subprocess.Popen(self.command("--resume", directory, "--workers", 2),
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
            try:
                execution = self.wait_for_active_workers(directory, process)
                os.kill(execution["episodes"][0]["pid"], signal.SIGKILL)
                stdout, stderr = process.communicate(timeout=90)
                self.assertEqual(process.returncode, 1, stdout + stderr)
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    process.communicate(timeout=15)
            summary = read_json(directory / "summary.json")
            self.assertEqual(summary["complete_episodes"], 1)
            self.assertEqual(summary["termination_counts"]["worker_error"], 1)
            good = next(e for e in summary["episodes"] if e["status"] == "complete")
            good_path = directory / "episodes" / good["data_file"]
            good_hash = sha256(good_path)
            self.invoke("--resume", directory, "--workers", 2)
            self.assertEqual(sha256(good_path), good_hash)
            self.assertEqual(read_json(directory / "summary.json")["complete_episodes"], 2)

    def test_ctrl_c_reaps_all_children_and_resume_completes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = self.plan(Path(temporary))
            process = subprocess.Popen(self.command("--resume", directory, "--workers", 2),
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
            try:
                execution = self.wait_for_active_workers(directory, process)
                process.send_signal(signal.SIGINT)
                stdout, stderr = process.communicate(timeout=20)
                self.assertEqual(process.returncode, 130, stdout + stderr)
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    process.communicate(timeout=15)
            self.assertFalse((directory / ".collection.lock").exists())
            self.assertEqual(read_json(directory / "dataset.json")["status"], "interrupted")
            for episode in execution["episodes"]:
                with self.assertRaises(ProcessLookupError):
                    os.kill(episode["pid"], 0)
            self.invoke("--resume", directory, "--workers", 2)
            self.assertEqual(read_json(directory / "summary.json")["complete_episodes"], 2)

    def test_invalid_workers_and_parallel_gui_are_rejected_before_dataset_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "data"
            for options in (("--workers", "0"), ("--workers", "-1"), ("--workers", "2", "--gui")):
                self.invoke("--config", ROOT / "experiment_specs/collection/smoke.json",
                            "--output-dir", output, *options, expected=2)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
