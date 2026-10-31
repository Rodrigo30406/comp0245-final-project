# C0: offline evidence behind experiment C3 (2026-10-30)

Descriptive checks on the validation cache (`results/step3/cache/val`, 284 episodes, 1 kHz), the trained models and the
existing run folders. No closed-loop simulation. Run each script from `final/robotnik_arm_task/` with the RoboEnv pixi
Python; its output is the `.txt` file of the same name. They were written during the review of the experiment-C plan
and were seen before the experiment-C3 pre-registration.

| Script | What it shows |
|---|---|
| `C_evidence.py` | Labels near the arm trigger: "parked-looking" frames (ring < 0.2 m, facing, arm at hold) are 7.7% reach; with the base still 100%, but see below |
| `offline_trigger_stats.py` | The 8,802 "still" frames have the arm already moving (max qd >= 0.2 rad/s in 8,801): the base is never still before the arm moves; quiet-arm parked-looking frames are 0.2-2.1% reach |
| `probe_trigger.py` | Response score r of each model's predicted torque change along the label jump at nav_done (0 = hold, 1 = label); EM steady state; v/w switching |
| `probe_time.py` | r on recorded frames k0 .. k0+33 ms (ego and ego-novw both 0.00 at k0, about 1.0 after); "kick" control |
| `probe_jitter.py` | Arm qd during hold vs at k0; the base is still before the arm moves in 0 of 284 episodes |
| `k0labels.py`, `footprint.py` | Labels and state at k0 (v = w = 0, reach-start torque; ring 0.074 m, beta 0.032 rad); training rows near k0 with a quiet arm: 147 (0.008%) |
| `circling.py` | ego-novw MM classes cross-tabulated with the same seed x episode in the ME cell: 70% of circling and 90% of parked_no_arm succeed once the expert arm fires at the parking pose (reads the gitignored ME logs) |
| `check455.py` | ego / ego-novw parked and facing in their own closed loop, and how often the arm extended (ego DEV 4/55, HARD 0/70) |
