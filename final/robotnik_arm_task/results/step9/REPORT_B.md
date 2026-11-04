# Step 9, Track B handover: learners (features, training, policies)

Date 2026-11-04. Spec: `STEP9_PLAN_PROMPT.md` sections 1, 4, 5, 7. TASK = this task folder; `PY` below =
`pixi run --manifest-path /home/w/comp0245/RoboEnv/pixi.toml python`. No git operation, no install, no existing file
edited; nothing written to `trained/step9/` or `results/step9/train_cache/` (the official locations stay empty for
Claude). Smoke artefacts (pilot2 split, cache, 2-epoch models, rollouts) are in the session scratchpad
(`/tmp/claude-1000/-home-w-comp0245/34757f37-87fe-417f-a316-33e71fa71488/scratchpad/b4full/`); section 5 gives the
commands that reproduce them.

## 1. What was built

| file | content |
| --- | --- |
| `obstacle_task/features.py` | Feature sets `arm20` (= bc_features `ego_novw`), `base70` (ego_novw[:6] + lidar 64), `single84` (arm20 + lidar 64), `base70_side` (base70 + side, for L2). One builder `build_features(set, signals, side=None)` for training and deployment; signals from `logged_signals` (training: `bc_features.measured_signals` of the recorded arrays + `lidar.scan_batch` of the logged base pose and the episode's obstacles) or `obs_signals` (deployment: same `measured_signals` + `obs["lidar"]`). Model file format `obstacle_mlp_v1` (`save_model` / `load_model`, checked names and dimensions) and the numpy `forward` (reuses `bc_features.forward`; classifier: logistic of the logit, = sklearn). |
| `obstacle_task/train.py` | CLI `split`, `build-cache`, `train`. `CONFIGS` = L0, L1base, L1arm, L1armK0, L2cls, L2base. `CachePart` (memory-mapped cache, checked against the split), `NetData` (one network's rows, standardisation, trigger copies, standardised chunks, validation loss), `trigger_copies` (C3 replica), `train_net` (epoch loop + early stopping), `run_train` (resumable, outputs + meta). |
| `obstacle_task/policies.py` | `SinglePolicy` (L0), `TwoNetPolicy` (L1, L1-K0), `SideLatchedPolicy` (L2), each `reset(target, info)` / `act(obs) -> (wheels (4,), tau (7,))`, (v, w) -> wheels by `collect_data.wheel_speeds` with `BaseControllerParams(nav_timeout=25.0)`. Factories `make_single`, `make_two_net`, `make_side_latched` and `make_config` ("L2,seed=3[,dir=...]", using `POLICY_MODELS`: L1-K0 = L1base + L1armK0, L2 = L2cls + L2base + L1arm). |
| `tests/test_obstacle_learning.py` | 39 unittest tests (gates B1-B3 and the rest, section 4); synthetic section-4 episodes in a temporary folder, plus B1 on `results/step9/smoke/pilot2` when that set is complete. |

sha256 at handover:
- features.py `106421a1adbf8dc11647597e6416845ad02f9bfd4540f4fe7256616019c96348`
- train.py `9f9305cfd863a9110e291cecb4a4e60777fedefd7f8b21b0989ceb7f863d2f73`
- policies.py `5b3a26c0121a2417cf202dc8ff588d1f5238aad3629d363e441a29f2ce45ac2f`
- test file `3f57fb60fdc1fc0e954064d7d7a58454f9b7ed7e5da754d6546fb896461b1ac6`

Shared files used unchanged: bc_features.py `e96bd3a2...`, train_policy.py `441df489...` (the step-6 values),
lidar.py `05fd2d86...`, scenario.py `f0ad5555...`. Every model's meta records the sha256 of train.py, features.py,
lidar.py, bc_features.py and train_policy.py, the split file and the cache manifest.

## 2. Design

**Split** (`trained/step9/data_split.json`, format `obstacle_split_v1`).
- Kept = index.csv `status == reached` and `collided` False.
- noisy, ambiguous, side, obstacles and target come from the set's `scenarios.json`. They must agree with
  index.csv (`ambiguous`, `side`, `noisy`, noise columns), or the split refuses.
- 85/15 by episode, per stratum in this order: clean unambiguous, clean ambiguous, noisy unambiguous, noisy
  ambiguous. Per stratum n_val = round(0.15 n), all drawn with one `default_rng(0)`; this is
  `train_policy.make_split`'s rule.
- Offsets come from `default_rng(0).integers(0, 9, n_train)`.
- The file stores per episode: file, n_steps, noisy, ambiguous, side, lateral offset, target, obstacles,
  nav_timed_out. It also records excluded episodes by reason, `n_kept_nav_timed_out`, and the sha256 of index.csv
  and scenarios.json.
- It refuses to overwrite without `--force`.

**Cache** (`results/step9/train_cache/`, format `obstacle_cache_v1`, one memory-mapped `.npy` per column and part,
manifest written last).
- Parts:
  - `train`: steps offset::9.
  - `val`: every 1 kHz step.
  - `trigger`: one row per training episode, its first nav_done step; the manifest records the step, time and
    segment of each.
- Columns (the ones needed by all configs): episode, step, t, base_xy, base_bearing, q_arm, qd_arm, target,
  label_v, label_w, label_tau, nav_done, side.
- Recorded dtypes are kept (float32; target float64). (v, w) is computed from the recorded clean wheel commands,
  as in train_policy.
- Each npz is checked against the split and the section-4 format: LOG_FIELDS dtypes, n_steps, log_every 1,
  status, collided, target, and obstacles equal to `meta["scenario"]`.
- `CachePart` refuses a cache built from another split, a subset (test or smoke) cache for an official model, and
  changed columns.

**Features at training time.** The lidar is not cached.
- Every chunk of rows is rebuilt from the cached signals by `features.build_features`. The arm columns are put back
  into the recorded 17-joint layout, as `train_policy.signals_from_cache` does.
- The lidar is computed by `lidar.scan_batch` per run of one episode's rows, in blocks of 2048 rows. The values are
  identical, and it is about 2x faster than one call on 65k rows because the temporaries stay in cache.
- On the same recorded step this is the same function on the same numbers as at deployment (gate B1: differences
  are exactly 0).

**Networks** (`CONFIGS`):

| config | network | inputs | outputs | rows |
| --- | --- | --- | --- | --- |
| L0 | 2x128 regressor | single84 | v, w, tau1..7 (equal weight) | all, K = 16 copies |
| L1base | 2x128 regressor | base70 | v, w | all, K = 16 |
| L1arm | 2x128 regressor | arm20 | tau1..7 | all, K = 16 |
| L1armK0 | 2x128 regressor | arm20 | tau1..7 | all, no copies |
| L2cls | MLPClassifier 2x64 | base70 | scenario side (+1 left / -1 right) | t < 1.0 s, no copies |
| L2base | 2x128 regressor | base70_side (the episode's scenario side) | v, w | all, K = 16 |

- All networks: ReLU, Adam lr 1e-3, alpha 1e-4, batch 200, float64, random_state = `np.random.RandomState(seed)`.
- Standardisation: inputs and regressor outputs use the mean and std (ddof 0; std < 1e-12 -> 1) of the original
  training rows, without the copies.
- Validation loss:
  - Regressors: standardised MSE over all validation rows and the network's outputs (C3's definition).
  - Classifier: mean binary log-loss over the validation rows with t < 1 s, p clipped to [eps, 1 - eps].
- Early stopping: `train_policy.EarlyStopping(patience=8, rel_tol=1e-3)`, at most 150 epochs; the best epoch's
  weights are kept.
- A final check recomputes the validation loss of the saved weights, which must equal the best value to 1e-12.

**Epoch loop.**
- One epoch is one pass over a fresh `default_rng([seed, 1]).permutation` of the training rows (cached rows plus
  copies).
- The rows go to `partial_fit` in chunks of 102,400 rows (a multiple of the batch size), with sklearn's shuffle off.
- sklearn keeps the Adam state between calls, and its per-call bookkeeping never touches the weights. So the
  batches and updates are exactly those of one `partial_fit` call over the permuted rows.
- The test proves this bit for bit for both the regressor and the classifier. Every epoch also asserts
  `t_ == epoch * n` and Adam steps `== epoch * ceil(n / 200)`.
- Everything runs inside `train_policy.single_thread()`, and the CLI sets the BLAS environment variables to 1.
- Training is deterministic per seed: the tests show identical weights and loss curves for the same seed.

**Trigger copies (C3, replicated).** `trigger_copies` follows `train_policy.trigger_copies` step for step.
- K copies of each trigger row, episode-major, with the recorded signals and labels.
- Each copy's arm qd gets `default_rng(0).standard_normal((n_ep * K, 7)) * sigma_hold`. sigma_hold is
  `train_policy.hold_qd_sigma`: the per-joint std of arm qd over the cached training rows with nav_done False (the
  C3 "auto" rule).
- The copies are standardised with the base scalers and appended after the cached rows.
- Every seed and config gets the same added rows; each network takes its own columns and outputs. So base70 and
  base70_side copies are K identical rows (no qd column), and L2base copies carry the episode's side.
- For arm20 the copies are bit-identical to `train_policy.trigger_copies(..., "ego_novw", ...)`, tested on
  synthetic data and on pilot2.

**Outputs.**
- `trained/step9/<config>_s<seed>.npz`, written by `features.save_model`: weights, scalers, feature set and names,
  output names, classes, meta.
- `<config>_s<seed>.json`: the same meta plus the model's sha256 and `complete: true`. It is written last, so a model
  counts as finished only when both files exist and the json says complete.
- `train` skips finished models; `--force` retrains.
- The meta records:
  - config, kind, feature set and names, outputs, seed, row rule;
  - split file and sha256, cache manifest sha256, data dir, stride;
  - row counts (base, copies, val) and episode counts;
  - the trigger block (K, k0 steps, sigma, row share, rule);
  - the standardisation and loss definitions and the hyperparameters;
  - epochs run, best epoch, val loss, stop reason, and the per-epoch history (train and val loss, fit and val
    seconds);
  - n_params, code sha256, versions, timings and peak RSS.
- Smoke guard: a non-default `--max-epochs`, `--split` or `--cache` needs a non-default `--out-dir`, so a smoke
  model never takes an official name.

**Policies.**
- Each step: `obs_signals(obs)` once, the features of every network from it, a numpy forward pass, and wheel
  speeds from `wheel_speeds`.
- Inputs: only `base_pos`, `base_bearing`, `q_mes`, `qd_mes`, `target` and `lidar`. Nothing from info and no clock.
  A missing or wrong-shape `obs["lidar"]` raises.
- Wrong network files are refused (kind, feature set and outputs are checked).
- SideLatchedPolicy:
  - `reset()` clears the latch.
  - The first `act()` computes p_left from the classifier on that observation's base70 features and sets
    `latched_side = +1 if p_left >= 0.5 else -1`.
  - The classifier runs once per episode (tested); the base network gets the latched side at every step.
  - `latched_side`, `p_left` and `episode_log()` expose the latch for logging.
- Track A's `evaluate.py` imports the file as `obstacle_task.policies`. It also works when loaded under another
  module name, as v1's loader would (tested).

## 3. Deviations from the plan, with reasons

1. **Chunked `partial_fit` instead of one call per epoch, with my own shuffle.** The permutation comes from
   `default_rng([seed, 1])` instead of the model's RandomState.
   - Reason: the lidar sets do not fit in 4 GB. At train2 scale (about 4.4M training rows and 6.9M validation
     rows), single84 needs about 3.0 GB for X_train plus 4.7 GB for X_val in float64.
   - Effect: the same algorithm, proven bitwise against a single call; only the permutation stream differs from
     train_policy's.
2. **Features are rebuilt for every chunk, every epoch (training and validation), never materialised.** Same reason:
   0.2-0.3 GB of private memory per fit instead of about 8 GB. The cost is the feature build, roughly a quarter to a
   third of each epoch; the section 6 timings include it.
3. **Standardisation moments come from a one-pass chunked (Chan) merge**, not numpy's full-matrix mean and std. The
   definition is the same, and it agrees with numpy to 1e-12 relative for the mean and 1e-10 for the std (tested).
   Reason: memory.
4. **Early-stopping tolerance.** The plan says patience 8; I kept C3's relative min-delta rule (1e-3, from
   `train_policy.EarlyStopping`), so "improvement" means what it meant in steps 3-7.
5. **The classifier stops on its own log-loss, not on an MSE** ("each network's own validation loss").
   - Its other hyperparameters (Adam 1e-3, alpha 1e-4, batch 200, ReLU) are the regressors'; the plan only names
     2x64.
   - Its inputs are standardised from its own t < 1 s training rows (recorded float32 `t`).
   - Inside the ambiguous band the label is a coin flip, so its loss floor is about 0.35, not 0.
6. **The L2 side input is standardised** like every other input column.
7. **The cache has no lidar or wheel-speed column.** The lidar is recomputed exactly, and no feature set uses wheel
   speeds. The trigger rows are a third part of the same `build-cache` run, not a separate command.
8. **Split filter.**
   - The default is exactly the plan's: reached and not collided.
   - A-to-B also dropped nav-timed-out episodes, so the split records `n_kept_nav_timed_out` (0 on pilot2 and
     pilot2n).
   - An opt-in `--exclude-nav-timeout` applies the A-to-B filter (see section 7).
9. **`train` accepts lists** (`--config L0,L1base` or `all`, `--seed 0-4`) and runs them in order in one process.
   Parallelism comes from launching several processes (section 5).
10. **Own model format** (`obstacle_mlp_v1`). `bc_features.load_model` checks names against
    `bc_features.FEATURE_SETS` and cannot hold the lidar sets. The forward pass reuses `bc_features.forward`.

## 4. Gate evidence

**Test runs**

| command | result |
| --- | --- |
| `PY -m unittest tests.test_obstacle_learning -v` | 39 tests OK in 3.4 s |
| `PY -m unittest discover -s tests -p 'test_obstacle*'` | 85 OK (shared 9 + Track A 37 + Track B 39) |
| full suite (`-p 'test_*.py'`) | 316 OK (2 skipped) in 53 s |

What the 39 tests cover:
- **Split:** determinism, filter (including the nav_timed_out option), strata, the draw rule, offsets, refusals,
  index/scenario agreement.
- **Cache:** rows are offset::9 or all steps; values equal the npz with dtypes kept; the trigger row is the first
  nav_done step; refusals.
- **Epoch loop:** chunked `partial_fit` = one call bit for bit; early stopping (patience 8, best weights kept);
  shuffle per epoch and per seed; single thread; hyperparameters.
- **`run_train`:** outputs and meta for all six configs; loss definitions recomputed; determinism; resumability;
  CLI.
- **Policies:**
  - act() shapes and the wheel conversion;
  - end-to-end equality of act() with the networks run on the training features;
  - side latching (threshold, once per episode, reset);
  - factories, harness-style import, lidar required.

**About the pilot2 data.** The pilot2 evidence uses the complete set Track A rebuilt around 01:35 (index.csv
sha256 `d2c292e1aef99c0c95b0c62b48410f5c6b2dbeb5820bd08719511166a9c7be61`). It has 100 clean episodes, all kept,
split 85/15 (strata: 44 train / 8 val unambiguous, 41 / 7 ambiguous). Track A deleted and rebuilt pilot2 during my
session; an earlier run on its first 16 episodes gave the same results.

**B1: training features equal deployment features (tolerance 1e-9).** "Deployment" means the harness observation of
the recorded step, with `obs["lidar"] = lidar.scan(base_pos[:2], base_bearing, obstacles)` exactly as
`evaluate.py` builds it.
- Synthetic data: 40 episodes with real scenarios. Compared: 300 shuffled training rows plus validation rows, the
  standardised training chunks, and the unjittered copies against the recorded trigger step.
- pilot2: 1500 train rows, 1500 val rows and all 85 trigger rows; the unit test also checks a subset cache.
- Result: the maximum difference is exactly 0.0 for all four sets (arm20, base70, single84, base70_side) and all
  parts. The lidar columns equal `lidar.scan_batch` of the logged pose bit for bit.

**B2: numpy forward equals sklearn (tolerance 1e-9).** On the pilot2 2-epoch models and all validation rows
(346,236; the classifier uses its 15,000 rows with t < 1 s), max |numpy − sklearn `predict` or
`predict_proba[:, 1]`| is exactly 0.0 for all six networks. The unit tests also pass at ≤ 1e-9. The classifier's
`classes_` are [-1, 1] with a logistic output.

**B3: trigger copies.** On pilot2:
- **Count:** 1360 = 16 × 85 for every config with copies, 0 for L1armK0 and L2cls, appended after the cached rows.
- **Jitter:** changes exactly qd1..qd7 in arm20 and single84, and nothing in base70 or base70_side.
- **Labels:** equal the recorded k0 row (v and w from wheel_cmds, plus the torques), read from the npz, for all
  85 × 16 copies.
- **Noise:** equals `default_rng(0).standard_normal((1360, 7)) * sigma_hold`. The arm20 copies equal C3's
  `train_policy.trigger_copies` bit for bit.
- **sigma_hold:** equals a direct recomputation from the npz files (rtol 1e-12, synthetic data). On pilot2 it is
  [0.0547, 0.029, 0.0199, 0.0227, 0.0326, 0.0003, 0.0091]; this is a clean set, so train2's noisy half will raise it.
- **Scalers:** identical with and without copies (base rows only).
- **Trigger rows:** every pilot2 k0 is in segment 1, at t = 17.5-25.4 s, with wheel command 0.

**B4: smoke run — pass.**
- Training: every config trained 2 epochs on pilot2 (`train --config all --seed 0 --max-epochs 2`): 64 s for all
  six, peak RSS 514 MiB.
- Evaluation: every policy ran pilot2 episodes 0-2 in `python -m obstacle_task.evaluate` (protocol v2.0) with no
  policy error.
- An L2 re-run against Track A's updated evaluate.py (01:52) gave identical outcomes, at 0.072 ms per step.

| policy | stop reasons | passed obstacle | on scenario side | docked_clean | policy ms (mean) |
| --- | --- | --- | --- | --- | --- |
| L0 | collision 1, tipped 2 | 1/3 | 1/1 | 0/3 | 0.16 |
| L1 | collision 1, tipped 2 | 2/3 | 2/2 | 0/3 | 0.24 |
| L1-K0 | collision 1, timeout 1, tipped 1 | 3/3 | 2/3 | 0/3 | 0.24 |
| L2 | collision 2, tipped 1 | 2/3 | 2/2 | 0/3 | 0.25 |

How to read the smoke outcomes:
- These failures are what 2-epoch networks trained on 85 episodes are expected to do. The machine was loaded at the
  time.
- Tipping is frequent (6 of 12). Offline, the smoke L0 tracks the labels on recorded validation steps about as well
  as L1arm, so this is closed-loop error growth, not a wiring bug.
- One contributing factor: with only 85 start headings, the forward lidar beams almost always read 4 m. When the
  obstacle is ahead at the start, their standardised values reach |z| ≈ 14.

## 5. Commands for Claude (from TASK)

```bash
PY="pixi run --manifest-path /home/w/comp0245/RoboEnv/pixi.toml python"

# 0. gates B1-B3 (+ B1 on pilot2 when it is complete); expected: 39 tests OK
$PY -m unittest tests.test_obstacle_learning -v

# 1. split, once, after data/obstacle/train2 (2000 episodes) is complete -> trained/step9/data_split.json
#    (prints kept / excluded counts, strata and "kept with nav_timed_out"; see section 7, item 2)
$PY obstacle_task/train.py split --data data/obstacle/train2

# 2. cache -> results/step9/train_cache (~1.7 GB, ~1 min with 12 workers)
$PY obstacle_task/train.py build-cache --workers 12

# 3. the 30 fits (seeds 0-4), longest first, 10 processes; resumable: run the same line again after an
#    interruption (finished models are skipped; logs are appended)
for c in L0 L1base L2base L1arm L1armK0 L2cls; do for s in 0 1 2 3 4; do echo "$c $s"; done; done | \
  xargs -P 10 -n 2 sh -c 'pixi run --manifest-path /home/w/comp0245/RoboEnv/pixi.toml python \
    obstacle_task/train.py train --config "$0" --seed "$1" >> "trained/step9/$0_s$1.log" 2>&1'
ls trained/step9/*_s[0-4].json | wc -l        # 30 when done; each json has "complete": true

# 4. (only if the H-MM primary holds) confirm2 networks, seeds 100-104: 20 fits
for c in L1base L2base L1arm L2cls; do for s in 100 101 102 103 104; do echo "$c $s"; done; done | \
  xargs -P 10 -n 2 sh -c 'pixi run --manifest-path /home/w/comp0245/RoboEnv/pixi.toml python \
    obstacle_task/train.py train --config "$0" --seed "$1" >> "trained/step9/$0_s$1.log" 2>&1'

# 5. evaluation (Track A's harness), e.g. dev2, one run per policy and seed
$PY -m obstacle_task.evaluate --policy obstacle_task/policies.py:make_config --policy-arg "L2,seed=0" \
    --eval-set data/obstacle/dev2 --workers 20 --out-dir results/step9/L2_s0_dev2
#    policies: L0, L1, L1-K0, L2; explicit files instead: make_two_net --policy-arg "base=...,arm=..."

# B4 reproduction (smoke dir of your choice; the guard enforces non-official names)
S=<smoke dir>
$PY obstacle_task/train.py split --data results/step9/smoke/pilot2 --out $S/split.json
$PY obstacle_task/train.py build-cache --split $S/split.json --cache $S/cache --workers 8
$PY obstacle_task/train.py train --config all --seed 0 --max-epochs 2 --split $S/split.json --cache $S/cache \
    --out-dir $S/models
for p in L0 L1 L1-K0 L2; do $PY -m obstacle_task.evaluate --policy obstacle_task/policies.py:make_config \
    --policy-arg "$p,seed=0,dir=$S/models" --eval-set results/step9/smoke/pilot2 --episodes 0-2 --workers 3 \
    --out-dir $S/eval_$p; done
```

The xargs launch line was dry-run with `echo`: arguments and log names map correctly. `trained/step9/` is created by
the split command.

## 6. Expected runtime and memory per fit

Measured with one epoch per network on a train2-sized cache, built by tiling pilot2's cache 20 times: 1700 train and
300 val episodes, 4,440,500 training rows + 27,200 copies, 6,924,720 validation rows, 1.6 GB. Single thread, idle
machine (load about 2).

| network | standardisation | fit per epoch | validation per epoch | epoch total | peak RSS (private) |
| --- | --- | --- | --- | --- | --- |
| L0 (single84, 9 outputs) | 8.8 s | 31.0 s | 22.9 s | 54 s | 1.92 GB (0.2 GB) |
| L1base (base70) | 6.5 s | 31.6 s | 17.2 s | 49 s | 1.60 GB (0.3 GB) |
| L2base (base70_side) | 6.6 s | about as L1base | about as L1base | about 49 s (95 s measured at load 18) | 1.61 GB (0.3 GB) |
| L1arm / L1armK0 (arm20) | 1.3 s | 16.8 s | 6.8 s | 24 s | 1.58 GB (0.2 GB) |
| L2cls (t < 1 s rows: 189k train / 300k val) | 0.3 s | 0.6 s | 0.5 s | about 1 s | 0.85 GB (0.1 GB) |

**Memory.** Peak RSS per fit is 1.6-1.9 GB, of which only 0.2-0.3 GB is private. The rest is memory-mapped cache
pages: file-backed, reclaimable, and one shared page cache for all parallel fits. Ten parallel fits need about 3 GB
of private memory plus about 1.7 GB of page cache, so the 4 GB-per-process rule holds with margin.

**Time per fit** = standardisation + epochs × epoch time + one more validation pass for the final check.

How many epochs each fit will run is unknown until the first real fit: at least 9 (best at epoch 1 plus patience 8)
and at most 150. For reference, C3 stopped at 11 epochs with patience 5. Projected wall time per fit on an idle
machine:

| network | 15 epochs | 25 epochs | 40 epochs |
| --- | --- | --- | --- |
| L0 | 14 min | 23 min | 36 min |
| L1base / L2base | 12 min | 21 min | 33 min |
| L1arm / L1armK0 | 6 min | 10 min | 16 min |
| L2cls | — | 1-3 min | — |

- **Under load:** epoch times roughly double, measured with about 20 of Track A's simulation workers running (load 18
  on 24 cores).
- **Total:** 30 fits at 25 epochs is about 7 h of CPU, so about 1-1.5 h wall with `-P 10`. The wall time cannot drop
  below the slowest L0 fit.
- **Cache build:** pilot2's 100 episodes took 3.1 s with 8 workers under load, so train2 should take about 1 minute.

## 7. For Claude to decide

1. **Logging the latched side (needed for M4).** `obstacle_task/evaluate.py` (Track A) has no hook for policy
   diagnostics, so `rollouts.csv` never records SideLatchedPolicy's `latched_side`, `p_left` or `episode_log()`.
   - Suggested fix: Track A or Claude adds the columns, e.g. `row.update(getattr(policy, "episode_log", dict)())` at
     the end of `run_policy_episode`.
   - Without it, M4 can only be rebuilt offline, by running the classifier on each episode's first observation.
2. **Nav-timed-out episodes.** The plan's filter keeps reached episodes whose navigation timed out; A-to-B dropped
   them, because their trigger row would be a stop forced by the timer rather than a dock. The split prints and
   stores the count. If train2 has any, either keep them (plan) or re-split with `--exclude-nav-timeout --force`
   before building the cache.
3. **Parallelism.** Choose `-P` against the other jobs (the dev2 and confirm2 collections). Memory is not the limit;
   CPU and memory bandwidth are.
4. **Tipping.** The smoke policies tipped in 6 of 12 episodes. Watch for this in the first real dev2 runs. Tipping
   falls under classify.py's `crash` class (tipped, invalid or policy error), not under collision.
