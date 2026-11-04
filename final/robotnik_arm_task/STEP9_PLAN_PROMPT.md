# Step 9 plan: obstacle task B1 ("fork in the road") + two-MLP factorisation (E1)

Written 2026-11-04 by Claude for the user (COMP0245 group assignment). The user approved the scope on
2026-11-04: B1 + E1 in full, with a 2D lidar, random side choice (E-rand), efficiency first; implementation
split between Codex (Track A, simulation side) and a Claude subagent (Track B, learning side); Claude (main)
owns the shared modules, runs every long job, audits both tracks and freezes the pre-registration.

TASK = `/home/w/comp0245/LAB_SESSION_COMP_0245_2026_PUBLIC/.claude/worktrees/eval-protocol/final/robotnik_arm_task`
(all paths below are relative to TASK). Python: `pixi run --manifest-path /home/w/comp0245/RoboEnv/pixi.toml python ...`
(default env: numpy, scipy, sklearn, pybullet, pinocchio; no torch). Tests: `unittest` (no pytest).

## 0. Why

The lecturer said the A-to-B task is simple and the team may define a harder problem, changing the expert
(obstacles, sensors). The professor suggested two MLPs (base / arm) for the arm-start copycat. B1 adds one
static obstacle between the start and the docking point; inside an ambiguous band the expert passes it on the
left or the right at random. MSE regression returns the conditional mean, so a policy trained on left and right
demonstrations should steer between them, into the obstacle: the textbook multimodal BC failure that the
unimodal A-to-B task cannot show. A mode-aware base head that commits to one side should remove it. The same
task tests the professor's split: a base network that sees the lidar and an arm network that does not.

## 1. Task definition (shared modules, DONE, read-only for both tracks)

`obstacle_task/scenario.py` and `obstacle_task/lidar.py` are implemented and tested
(`tests/test_obstacle_shared.py`, 9 tests). Do not edit them; if something must change, stop and report.

- `scenario.sample_scenario(seed, episode)` -> `Scenario` (dataclass; JSON via `save_scenarios`/`load_scenarios`):
  start pose (disc of 2 m, any yaw), target r 3.5-5.0 m from the start, any bearing, z 0.2-1.2 m;
  `dock_xy`/`dock_bearing`/`standoff` exactly as the A-to-B expert (`evaluate_policy.expert_base_goal`);
  one cylinder `obstacles=[{x, y, r 0.25-0.35, h 1.5}]` at a fraction 0.3-0.7 of start->dock, lateral offset
  `lateral_offset` d in [-0.4, 0.4] m (left normal positive); `ambiguous = |d| < 0.2`; `side` +1 = pass left,
  -1 = right (fair coin if ambiguous, else the shorter side -sign(d)); `waypoint_xy` = centre + side (r + 0.9) n.
  Rejection constraints keep the in-place turns (start, waypoint, dock) and both straight segments clear.
  Each episode uses its own RNG stream `[seed, episode, 9]`; `make_scenarios(seed, n, noise_frac)` adds the
  noisy flag from stream `[seed, episode, 10]`. Measured on 2000 draws: 1.22 attempts on average,
  start-dock length p5/p50/p95 2.97/3.70/4.28 m, 48% ambiguous, 50% left within the band.
- `lidar.scan(base_xy, yaw, obstacles)` -> 64 ranges (base frame, beam 0 forward, counter-clockwise,
  clipped at 4 m); `lidar.scan_batch(xy (N,2), yaw (N,), obstacles)` -> (N, 64). Exact ray-circle casting,
  ignores the robot body (idealised sensor at the base centre).

## 2. Expert v2 (Track A)

- Base: two segments driven by the UNCHANGED `robotnik_arm_control.base_controller`.
  Segment 0: goal = `waypoint_xy` (pass `dock_bearing` as the unused goal bearing). Segment 1: goal = `dock_xy`,
  goal bearing = `dock_bearing`, i.e. exactly the A-to-B navigation. At each step call the controller for the
  current segment; if a segment-0 call leaves its state in phase "orient" (it reached or passed the waypoint),
  discard that call's command, switch to segment 1 with a fresh `BaseNavState()` and call the controller for
  segment 1 in the same step. The orient phase is never executed at the waypoint.
- `BaseControllerParams(nav_timeout=45.0)` (A-to-B: 25 s); the timeout covers the whole navigation: if it fires
  in segment 0, switch to segment 1 with `done=True`.
- Arm: `robotnik_arm_control.arm_controller` unchanged; `reach_active = (segment == 1 and state1.done)`.
- The expert reads the privileged scenario (waypoint, dock); learned policies never see it.

## 3. Simulation additions (Track A)

- Spawn each obstacle right after the base reset: `createCollisionShape(GEOM_CYLINDER, radius=r, height=h)`,
  `createVisualShape(GEOM_CYLINDER, radius=r, length=h)`, `createMultiBody(baseMass=0, basePosition=[x, y, h/2])`.
  Robot-obstacle contacts are on even with `collision_enable 0` (it only toggles self-collision).
- Collision: at every step `getContactPoints(bodyA=robot, bodyB=obstacle)` non-empty -> the episode ends
  immediately (contact forces are not physical: 7.5 kN measured when pushing a wall). Record time and first link.
- Sensor: the harness puts `obs["lidar"] = lidar.scan(base_pos[:2], base_bearing, scenario.obstacles)` (64,)
  in every observation. It is the only obstacle information a learned policy receives.
- Learned policies get a sanitized `info` at reset (robot model, frame ids, joint limits, time step, start pose,
  initial joint angles; NO obstacles, waypoint, side or dock). The expert policy gets the scenario.

## 4. Protocol v2.0 (Track A implements, Claude runs)

Sets under `data/obstacle/<name>/` (each: `scenarios.json` via `scenario.save_scenarios`, `config.json`,
`index.csv`, `ep_XXXX.npz`), all recorded with the expert so its own outcome is known per episode:

| set | seed | episodes | noise_frac | role |
|---|---|---|---|---|
| train2 | 3 | 2000 | 0.5 | training / validation (split by episode) |
| pilot2 | 3000 | 100 | 0.0 | expert validation, smoke tests (also run once with noise_frac 1.0 as pilot2n) |
| dev2 | 300 | 300 | 0.0 | every decision |
| confirm2 | 302 | 300 | 0.0 | replication with fresh training seeds |
| test2 | 301 | 300 | 0.0 | sealed: built and opened only for the final frozen model |

- Collector (`obstacle_task/collect.py`): same per-step fields as `collect_data.LOG_FIELDS` (+ `segment` int8)
  and the same DART noise model and defaults as `collect_data.py` (v 0.02 m/s, w 0.05 rad/s, torque 2.0 N*m,
  held 0.3 s, arm noise fading within 0.15 m of the target, base noise only while navigating and fading within
  0.5 m of the CURRENT segment goal); labels are the clean expert commands. Differences from A-to-B, nothing
  else: obstacles, segments, nav_timeout 45 s, step cap 60 000 (60 s), collision ends the episode with status
  `collision`. Stop rule as `collect_data.py` (template check_stop + its reached fix). The npz `meta` JSON
  carries the full scenario dict, status, collided, collision_time/link, noise settings. `index.csv` adds
  `ambiguous, side, lateral_offset, obstacle_r, collided, segment_switch_time`. Resumable like collect_data.
- Harness (`obstacle_task/evaluate.py`, `PROTOCOL_VERSION = "v2.0"`): same rules and columns as
  `evaluate_policy.py` v1.1 except T_MAX 60 s, collision is an early stop and a failure, and the new columns
  below. success = EE < 10 mm at some moment within 60 s, no collision before it, normal end; the 1 s and 10 s
  hold windows as v1.1 (a collision inside them ends the window and fails the hold metrics, not success).
  New columns: `collision, collision_time, collision_link, min_clearance` (robot-obstacle distance from
  `getClosestPoints` every 50 ms), `passed_obstacle` (the base's coordinate along start->dock exceeded the
  obstacle centre's), `side_taken` (+1/-1: sign of the base's lateral position at that moment; 0 if never),
  `docked` (v1.1 `parked`: within 0.20 m of the standoff ring and still for 1 s), `docked_clean` (docked
  before any collision), `time_to_dock, ambiguous, scenario_side, lateral_offset, obstacle_r`.
  Reuse `evaluate_policy` helpers by import (metrics classes, setup pieces); never edit `evaluate_policy.py`.
- Failure classes (`obstacle_task/classify.py`), first match wins: `crash` (tipped / invalid / policy error),
  `collision`, `no_pass` (never passed the obstacle, no collision), then the A-to-B classes of
  `analyze_rollouts.classify_row` for episodes that passed (circling, left_or_overshot, parked_no_arm,
  extended_not_reached, approached_other, ...). Validate on the expert (all successes unclassified, failures
  explained) like step 3.

## 5. Learners (Track B)

Data: train2 episodes with expert status reached and no collision; split 85/15 by episode stratified by
(noisy, ambiguous), split seed 0; stride 9 with a per-episode offset (default_rng(0)), validation on all
1 kHz rows of the validation episodes, as `train_policy.py`. Lidar features recomputed offline with
`lidar.scan_batch` from the logged base pose and the episode's obstacles (deployment uses obs["lidar"]).
Trigger copies: K = 16 copies of each training episode's first nav_done row (reach onset), arm-qd jitter
exactly as C3 (read `train_policy.py`: trigger cache and automatic sigma; replicate and document).

Feature sets (one shared builder for training and deployment):
- `arm20`: the A-to-B `ego_novw` set (target in the base frame dx, dy, z, rho, cos/sin bearing, q 7, qd 7).
- `base70`: dx, dy, z, rho, cos/sin bearing (6) + lidar 64. No arm state, no wheel speeds, no clock.
- `single84`: `arm20` + lidar 64.

Configs (sklearn `MLPRegressor` 2x128 ReLU, Adam 1e-3, alpha 1e-4, batch 200, standardized inputs and
outputs, early stopping on each network's own validation MSE with patience 8, max 150 epochs, best weights kept;
numpy forward at deployment; seeds 0-4 on dev2, 100-104 for confirm2):
- L0 single: `single84` -> (v, w, tau1..7), copies on, 9 outputs equal weight (the C3 recipe + lidar).
- L1 two-net: base net `base70` -> (v, w); arm net `arm20` -> tau1..7; the same copy rows as L0 (each net gets
  the copied rows with its own outputs). The policy wrapper calls both.
- L1-K0: L1's base nets + arm nets trained WITHOUT copies (only the arm nets are retrained).
- L2 side-latched: L1's arm nets; base = (a) side classifier `MLPClassifier` 2x64 on `base70` rows with
  t < 1.0 s, label = scenario side; (b) side-conditioned base regressor on `base70` + side (+-1) -> (v, w), all
  rows, with copies. Deployment: at the first act() of an episode compute p(left) from the first observation,
  latch side = +1 if p >= 0.5 else -1 for the whole episode, feed it to (b).
Fits: 5 x (L0, L1 base, L1 arm, L1-K0 arm, L2 classifier, L2 base) = 30.

Policies (`obstacle_task/policies.py`): `reset(target, info)`, `act(obs) -> (wheel_cmds (4,), torque_arm (7,))`;
(v, w) -> wheels with `collect_data.wheel_speeds` and `BaseControllerParams(nav_timeout=25.0)` as `BCPolicy`.

## 6. Pre-registration DRAFT (Claude freezes it in report.txt before any learner runs on dev2)

Statistics as steps 3-7: episode = unit, all configs paired on the same dev2 episodes, 5 seeds per config,
two-level bootstrap (seed, then episode; 10 000) and Welch on per-seed rates, 95% CIs; Holm within the
secondary metrics of each hypothesis. Ambiguous band = `ambiguous` column (|d| < 0.2 m), fixed by the sampler.

H-MM (multimodality; primary). L1 (MSE base) averages left and right inside the band and drives into or stalls
in front of the obstacle; L2 (latched side) does not.
- Primary: docked_clean rate inside the band, L2 - L1: both 95% lower bounds > 0.
- M1: in the band, >= 60% of L1's failures are `collision` or `no_pass`.
- M2 (specificity): outside the band the docked_clean difference L2 - L1 has both CIs within +-7 pp.
- M3: within L1, docked_clean is lower inside the band than outside (paired by seed, point estimate).
- M4: L2 passes on its latched side in >= 95% of its passes (no mode hopping).
- Competing: C1 closed-loop symmetry breaking rescues MSE (L1 in band ~ outside); C2 perception is the
  bottleneck (L1 and L2 fail alike in and out of the band); C3 the side classifier or the conditioned regressor
  fails off-distribution (L2 worse outside the band).
- Confirmation (only if the primary holds): L1 vs L2 with seeds 100-104 on confirm2: point estimate > 0 and
  bootstrap 95% lower bound > 0 -> "confirmed".

H-FACT (professor's factorisation). L1 vs L0, both with copies: two-sided, reported with CIs; prediction
L1 >= L0 on overall docked_clean (the base net cannot use arm-state shortcuts) and no difference in
`parked_no_arm`. Secondary: success@10 mm, collision, circling.

H-COPY (professor's copycat claim). L1-K0 vs L1: prediction `parked_no_arm` (arm never starts after docking)
>= 50% for L1-K0 and <= 10% for L1, i.e. the split alone does not start the arm. If L1-K0 `parked_no_arm` is
< 20%, the split does solve it (the professor is right) and we report that.

Reported for every config and the expert: success@10/20/50 mm, docked_clean, collision, no_pass, failure
classes, held_10s, policy_ms. Figures: docked_clean and collision vs lateral offset (binned, per config with
seed CIs: the "bifurcation plot"); top-down trajectories of ambiguous episodes; failure-class bars.
test2: one run of the final frozen model (L2 if confirmed, else the best of L0/L1 on dev2) and the expert.

## 7. Ownership, rules, gates

Ownership (create/modify ONLY your files; read anything):
- Shared (Claude, done): `obstacle_task/__init__.py, scenario.py, lidar.py`, `tests/test_obstacle_shared.py`.
- Track A (Codex): `obstacle_task/sim.py, expert.py, collect.py, evaluate.py, classify.py`,
  `tests/test_obstacle_sim.py`, `results/step9/` scripts (set building, smoke runs, `run_step9_sets.sh`),
  `results/step9/REPORT_A.md` (handover).
- Track B (Claude subagent): `obstacle_task/features.py, train.py, policies.py`, `tests/test_obstacle_learning.py`,
  `results/step9/REPORT_B.md`.
Rules: never edit existing files outside `obstacle_task/`, `tests/test_obstacle_*`, `results/step9/`; never read
`data/test`; A-to-B data, models and results are read-only; no git operations; no package installs; jobs longer
than ~10 min are not run by the implementers (write resumable scripts, run smoke tests only; Claude runs the
long chains); keep code in the style of the existing modules (docstrings, constants at the top, numpy).

Gates Track A (report the evidence in REPORT_A.md):
- A1 `python -m unittest discover -s tests -p 'test_obstacle*'` passes; the full suite `-p 'test_*.py'` still passes.
- A2 no-obstacle equivalence: with zero obstacles and segment 0 skipped, the new collector reproduces
  `collect_data.py`'s recording of 3 A-to-B episodes bitwise (same seed, start, target, noise).
- A3 the harness's `obs["lidar"]` equals `lidar.scan_batch` on the logged poses (<= 1e-12).
- A4 expert on pilot2 (100 clean): success >= 95%, collisions 0, `side_taken == scenario_side` in every pass,
  docked_clean >= 97%. If not met: stop, report the failures with evidence; do not change scenario.py.
- A5 noisy expert on pilot2n (noise_frac 1.0, 100 episodes): collisions <= 3%, success >= 90%.
- A6 the harness with the expert reproduces the collector's recordings on 20 pilot2 episodes (as v1.1's check).
Gates Track B (REPORT_B.md):
- B1 training-time features == deployment features (<= 1e-9) on recorded steps, all three sets (use synthetic
  episodes in the npz format of section 4 until pilot2 exists, then pilot2).
- B2 numpy forward == sklearn predict / predict_proba (<= 1e-9).
- B3 copies: K per training episode, jitter only on the arm-qd columns, labels equal to the source row.
- B4 smoke (after Track A delivers): each config trains 2 epochs on pilot2 and runs 3 episodes in evaluate.py.

## 8. Chain and time (estimates, assistant implementation + compute)

| step | who | estimate |
|---|---|---|
| shared modules + this plan | Claude | done |
| Track A implementation + gates A1-A3, A6 | Codex | 6-10 h |
| Track B implementation + gates B1-B3 | subagent | 5-7 h (parallel with A) |
| pilot2 / pilot2n expert gates A4-A5 | Claude runs | 0.5 h |
| train2 (2000) + dev2 + confirm2 recordings | Claude runs | 1.5-2 h compute |
| caches + 30 fits | Claude runs | 2-3 h compute |
| freeze pre-registration (report.txt) | Claude | 0.5 h, before any dev2 learner run |
| dev2: 20 learner runs + expert | Claude runs | 4-6 h compute |
| analysis, figures, report.txt results | Claude | 3-4 h |
| confirm2 (10 runs, fresh seeds), test2 (final) | Claude runs | 2-3 h compute |
