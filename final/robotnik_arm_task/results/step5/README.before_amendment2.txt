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

New code:

- `dagger_relabel.py`: sequential shadow-expert replay of full 1 kHz trajectories.
- `results/step5/tolerance_curve.py`: F0 curves from existing DEV, HARD and
  CONFIRM rollout results, with stalled C3 rows replaced as in experiment D.
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

The chain then stopped at F0. The published DEV C3 value (60.5% rounded)
comes from the original early-stop rows; replacing stalled rows as required
raises it to 60.8%, so the published-value assertion fails even though the
10/20/50 mm per-row metric checks pass. Read-only diagnosis also found that
the required existing C3 stalled reruns are missing for all five HARD and
all five CONFIRM seeds. F0 is restricted to existing results, and new
HARD/CONFIRM evaluations remain gated. No implementation code or frozen
rules were changed to bypass these conditions.

No chain is running. F2-F5 and F_mechanisms have not started, and there is no
experiment F verdict. Resolve the F0 baseline/rerun prerequisites before
using the resume command above; the completed compatibility stage will be
verified and skipped. See `results/step5/F0_published_baseline_diagnosis.json`,
`chainF_blocked.json`, and [the Chinese audit report](results/step5/CODEX_REPORT_F.md)
for exact evidence, missing runs, and commands. The report preserves both
earlier sessions as history. Start/end integrity evidence is saved in
`resume_integrity_20261102.json` and `resume_final_integrity_20261102.json`.
