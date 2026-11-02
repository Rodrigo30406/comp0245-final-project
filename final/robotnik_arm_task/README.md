1) Make sure the folders "/models", "/configs" from RoboEnv are accessible (easiest is to copy them to this folder)
2) Use your RoboEnv environent (in whichever way you have been running the labs: pixi, uv, conda, etc)
3) Run robotnik_arm_task.py

Expected behaviour: pybullet 3D environment with 7dof arm mounted on a 4-wheeled platform. The robot moves to reach a target (red dot) with its end-effector. The simulation terminates when the target is reached, and plots of the robot behaviour are shown.

Key variables in robotnik_arm_task.py:

 - TARGET_EE_POS: reach target point in cartesian coordinates (x,y,z), represented by a red dot.

 - base_pos:     wheeled base position (x,y,z)
 - base_bearing: wheeled base bearing (angle in radians)
 - q_mes:        arm joint angles (radians)
 - qd_mes:       arm joint angle velocities (rad/s)

 - wheel_cmds:     angular velocity command for each wheel (rad/s)
 - torque_arm:     command torques for each arm joint
 - torque_fingers: torque for each finger joint of the end-effector grasper. In this work, they should just remain fixed.
 
 Assessment Goal:
 - ML model that produces (wheel_cmds, torque_arm) from measurements (base_pos,base_bearing,q_mes,qd_mes)

Step 5: tolerance curves and one round of DAgger (experiment F)
------------------------------------------------------------

The approved protocol is `STEP5_DAGGER_PLAN_PROMPT.md`. Its section 3 and the
approved F1 amendment 1 are frozen in `report.txt` and
`results/step5/prereg_F_frozen.txt`. Every stage checks their hashes.
Approved amendment 2 changes only descriptive F0 and its consistency rule.
The current plan SHA256 is
`554caf7ffea7364ac870b08eb85bbca80d06503d241a68d63e3e330066c8b43b`;
removing that paragraph reproduces the original frozen plan. The original
freeze records remain unchanged; the frozen section hash remains
`753a14e699d9c82421bf2c11589393f1a39f43bfe0b1dc66750b61744db6c06e`.
`amendment2_baseline.json` and `implementation_change2.json` record the change.

New code:

- `dagger_relabel.py`: sequential shadow-expert replay of full 1 kHz trajectories.
- `results/step5/tolerance_curve.py`: F0 curves from existing DEV, HARD and
  CONFIRM original rollout rows. DEV adds a separate C3 stalled replacement
  sensitivity curve (60.8% at 10 mm); HARD/CONFIRM use original rows only.
- `results/step5/build_dagger_cache.py`: F3 features, shadow labels and the
  lossless `results/step3/cache/dagger1/` side cache; descriptive `F_data.txt`.
- `results/step5/step5_pipeline.py`, `run_expF.sh`, `f_common.py`: resumable
  selection, pilot, simulations, training, integrity checks and sequential gates.
- `results/step5/F_mechanisms.py`: the frozen success criteria, M1-M4,
  descriptive secondary metrics with Holm correction and full comparisons.
- `tests/test_extra_rows.py`, `tests/test_f_gates.py`: training integration and
  sequential gate tests.

Modified code: `train_policy.py` adds the `ego-novw-trig-dag1` model and optional
`--extra-rows` for `train` and `train-many`. Extra rows are appended after the
trigger copies, with the original stride-9 standardisers and unchanged expert
validation. `tests/test_dagger_relabel.py` implements amendment 1, preserving
the first verification outputs. `report.txt` contains the frozen registration;
this README documents the execution.

Run from this directory in RoboEnv (all Python children inherit MPLCONFIGDIR):

```bash
nohup bash results/step5/run_expF.sh all >> results/step5/chainF.log 2>&1 < /dev/null &
step5_chain_pid=$!
wait "$step5_chain_pid"
```

The shell waits to retain the execution session. In another session, check
`results/step5/chainF.log` after `sleep 600`. Completed stages and complete runs
are verified and skipped. An interrupted run directory without `summary.txt`
is deleted before rerunning. `WORKERS=22` is the simulation default; `FJOBS=6`
is the training default, subject to the available-memory check. Training and
simulation do not overlap.

DEV uses ten seeds; HARD and CONFIRM each use five. HARD opens only after the
DEV success and risk gates pass; CONFIRM opens once after HARD passes.
`F_mechanisms.py --allow-confirm` is used only in that gated stage. A failed
gate completes the experiment with KEEP C3 and leaves later sets unopened.
F5 reruns C3 HARD/CONFIRM stalled episodes with `--no-early-stop` after each
set's gate passes, before its dag1 evaluations. F0 never requires those reruns.
`data/test` remains sealed.

Outputs: F1 verification JSON/logs; `tolerance_curve.{txt,json}` and
`fig_tolerance.{png,pdf}`; pilot diagnostics; `F2_selection.json`; 400 training
rollouts; the DAgger cache/manifest; `F_data.{txt,json}`;
`backward_compatibility.json`; 15 `trained/ego-novw-trig-dag1_s*.npz` models;
`model_checks.json`; gated evaluation runs; `F_MM_compare_<set>.{txt,json}`;
`F_mechanisms.{txt,json}`; stage timestamps and `CODEX_REPORT_F.md`. Errors
produce `chainF_blocked.json` with a resume command.

Execution status (2026-11-02, after AF_UNIX was enabled): frozen hashes and all
107 implementation/input hashes passed. The full suite passed (175 tests,
2 existing optional slow tests skipped), and the C3 seed 0 retraining arrays
are bit-identical to the original model (`backward_compatibility.json`).

Current execution status (amendment 2): the chain was found terminated at F2 at 04:14:25 UTC (exit code 143; exact termination time and cause unknown), and documented at 04:15:28 UTC. F0 and the pilot passed; F2 completed 240/400 episodes. Seed 3 has 34 partial rows and no summary, so resume will delete and rerun that incomplete directory. F3-F5 and F_mechanisms have not started; HARD/CONFIRM remain unopened. The exact error and resume commands are recorded in `results/step5/chainF_blocked.json` and the Chinese report. Later stages have not been run past this blocker.

Full amendment-2 tests: 177 tests passed, with 2 existing optional slow tests skipped. Final integrity checks cover 109 implementation/input files and 380 existing result files. See [the Chinese audit report](results/step5/CODEX_REPORT_F.md), `resume_final_integrity_amendment2.json`, and `chainF.log`; earlier session history is preserved.
