# Step 11 (2026-11-06): three pre-registered controls R / S / W

DEV only (HARD, CONFIRM and data/test not opened); no adoption rule, the main model stays C3. The pre-registration
(`prereg_step11_draft.md`) was frozen at 2026-11-06 01:43:40 UTC before any DEV run of a new model
(`prereg_step11_frozen.txt`: sha256 of the plan, code, scripts, split and caches) and is copied verbatim into
`report.txt`; the results are in `report.txt` ("第 11 步") and `report_step11_results.txt`.

| | change against C3 (one variable) | verdict under the frozen rule |
|---|---|---|
| R | the K = 16 jittered copies come from a random base-moving row (>= 1 s before the latch), not the onset row | H-R supported: 0.3 % success, -60.1 pp [-66.9, -53.7] vs C3 |
| S | a base net on the six target features (v, w) + an arm net on ego_novw (tau), arm with / without copies | H-S-COPY supported: 0.4 % without, 63.3 % with copies (+62.9 pp); = C3 (+2.8 pp), circling 12 % -> 0.3 % |
| W | hidden layers (256, 256, 256) instead of (128, 128), with / without copies | onset part supported (2.4 %); floor part "undetermined": 33.5 % vs 60.6 % (-27.1 pp) |

Files
- `step11.py`: `build-placebo-cache` (R's rows; cache in `cache/placebo`, regenerable, not in git), `train-r`
  (C3's `train_policy.run_train` path with the placebo row loader), `train-s --net base|arm-K0|arm-K16`,
  `train-w --variant K0|K16`. Adds the feature set `ego_target` (first six ego_novw columns) at import; no
  existing file is edited. Models: `trained/step11/{R,S-*,W-*}_s{0..4}.npz`.
- `../../policies/split_policy.py`: the two-network policy of S (`--policy-arg "base=...;arm=..."`).
- `run_step11_local.sh train|eval`: R and S, trained and evaluated on the local machine (run folders `R_s*_dev`,
  `S-K0_s*_dev`, `S-K16_s*_dev`).
- `run_step11_W.sh`: W on the UCL machine windcharger. Its simulator is deterministic but not bitwise equal to the
  local one (the expert: 0/300 episodes identical, same 99.3 % success), so W and its comparators (C3 and ego-novw
  s0-4 re-run there: `wc/wc_C3_s*`, `wc/wc_ego-novw_s*`) are all compared on that machine; C3 replicates there
  (60.6 % vs 60.5 %). `wc/` holds the run folders copied back (without trajectories and logs) and the version of
  `step11.py` used there (identical W code; sha in the freeze record).
- `step11_analysis.sh r|s|w`: the pre-registered read-outs (`analyze_rollouts.py compare`, two-level bootstrap and
  Welch) -> `analysis/*.txt|json`; `step11_cond.py`: P(arm never extended | parked); `analysis/welch_classes.txt`:
  Welch intervals of the failure-class shares. S needs `--allow-incomplete` only because
  `analyze_rollouts.is_hybrid()` flags any policy-arg containing `base=` (all S runs are 300/300, 0 harness errors).
- Tests: `tests/test_step11.py`.

Post-run refactor (2026-11-06 ~05:30, after all runs; no effect on any result): the full test suite showed that
registering `ego_target` in `bc_features.FEATURE_SETS` left `bc_features.build_features("ego_target")` on a wrong
branch (`test_training` iterates over every feature set). The registration is now `split_policy.register_ego_target()`,
which also patches `build_features` to return the first six ego_novw columns; training and the policy always sliced
those columns themselves, so the models and runs are unchanged. The freeze record's sha256 of `step11.py` and
`split_policy.py` refer to the versions before this refactor.
