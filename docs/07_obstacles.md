# 7. Obstacle avoidance: static and moving obstacles

The reaching recipe (egocentric inputs, ignition copies, chapters 2–6) extended to tasks with an obstacle in the way.
The policy gets one extra input: a **64-beam planar lidar scan** around the base (92 inputs in total, variant
`raw21_targetpolar_clock_lidar64_v1`, [02](02_inputs_and_outputs.md)). It never sees the obstacle's position: only what
the scan shows. Only the best model of each family is reported; full tables in [results.md](results.md) sections 2–3.

## 7.1 Static obstacle

**Task** (the group's obstacle task, `controlled_experiments/`): one cylinder between the start and a target 3.5–5 m
away. The expert drives around it, docks and reaches. Evaluation: 300 unseen scenes, 60 s each. Training: 2000 expert
episodes. Success = end effector within 1 cm of the target without touching the obstacle; *held 10 s* = still within
1.5 cm 10 s later.

**What we did to the data.** Cloning the plain recordings gave a policy that avoided the obstacle and reached, but left
the target 2–3 s later (held 10 s: 8 %). Three changes to how the demonstrations are recorded, the same ideas as in
reaching:

1. **Keep recording after the reach.** The original recordings stop the moment the hand first reaches the target, so the
   policy never sees what to do next. We keep the expert running 10 s more: the data now contain "stay on the target".
2. **Perturb the commands while recording** (DART, [01](01_data_collection.md)): small random pushes on the executed
   wheel and arm commands, while the recorded label stays the expert's clean command. The data show how to recover from
   the policy's own small errors.
3. **Keep perturbing the base while holding.** A cloned policy never stops its base exactly ([06](06_base_drift.md)),
   so during the hold its base creeps. The original noise was switched off once the base arrived, so the data never
   showed a moving base during the hold. With the perturbation kept on, the data show the arm compensating a moving
   base, which is what the policy needs.

The first two alone raise *held 10 s* only to 59 %; the third brings it to 77 %.

**Results** (300 unseen scenes, three training seeds, mean ± sd (min–max)):

| policy | success | held 10 s | docked without contact | contacts |
|---|---|---|---|---|
| expert | 100 | 100 | 100 | 0 |
| MLP-Ego-IC + lidar, original recordings | 54 ± 6 (48–61) | 8 ± 4 | 93 | 0 |
| **MLP-Ego-IC + lidar, improved recordings** | **80 ± 8 (71–87)** | **77 ± 9 (67–85)** | 90 | 1 |
| GRU-Ego + lidar, improved recordings | 55 ± 9 (45–60) | 38 ± 5 (32–42) | 99 | 0 |
| TF-Ego + lidar (transformer), original recordings | 12 ± 4 (7–15) | 0 | 97 | 0 |

**Reading.** Avoiding the obstacle is the easy part: every model docks without contact in 90–99 % of the scenes. As in
reaching, the hard part is the last centimetre and staying there, and it is solved by what the data show, not by a
bigger model: the MLP beats the memory models (the GRU docks better but reaches less precisely; the transformer, trained
on the original recordings only, docks but almost never reaches 1 cm).

**Limitations.** Of the ~20 % failures, about half never dock cleanly and in the rest the base docks but the hand
does not get within 1 cm. The seed spread is large (71–87 %). The transformer was not retrained on the improved recordings. The
dead band that helps parking in reaching hurts here (80 → 47 %): it removes the small final turn.

## 7.2 Moving obstacle (preliminary)

**Task** (ours): the reaching task with one cylinder (radius 0.25–0.35 m) crossing the robot's path at 0.10–0.25 m/s,
timed so that the two would meet. Success = reach within 1 cm and stay 10 s, without contact; 75 s per episode.
Test: 100 unseen scenes.

**Expert.** The course's expert has no avoidance (in these scenes it hits the cylinder 54 times out of 100), so we
wrote one (`scripts/obstacle/dynamic_obstacle.py`): every 50 ms it simulates a few candidate manoeuvres (keep going,
wait, slow down, back off, turn and drive around) with the cylinder moving at its current velocity, and executes the
cheapest according to one cost that combines reaching the dock and keeping clear of the obstacle (the idea of the
dynamic-window planners used in open-source navigation). Far from the cylinder its best candidate is the course
expert's own command, so it behaves like it. Two features of the simulator shaped it: the base slides sideways if it
turns while driving, so manoeuvres turn in place and then drive straight; and it keeps rotating after a turn, so a
yaw-rate correction was added. Training data: 1600 episodes recorded with the same perturbations as above, all
successful, no contacts.

**Results** (100 unseen scenes, one training seed):

| policy | success | contacts | did not reach |
|---|---|---|---|
| our avoidance expert | **100** | 0 | 0 |
| course expert (no avoidance) | 46 | 54 | 0 |
| MLP-Ego-IC + lidar | 34 | 26 | 35 (+5 reached, then left) |
| GRU-Ego + lidar | 0 | 33 | 66 |

**Reading.** The MLP learns part of the avoidance from a single scan: it halves the contacts of the course expert and
succeeds in 16 of the 54 scenes that need avoiding. What it does not learn is getting back on track after a detour: in
35 scenes the base wanders off (median 2.3 m from the dock at the end), far from the cylinder. The GRU fits the data
poorly and never finishes.

**Limitations and next steps.** One seed and no transformer: a preliminary result. A single scan cannot show the
cylinder's velocity (two scans or memory can). Because this expert can label any state, DAgger (relabelling the
states the policy visits) applies directly and targets exactly the recovery that fails.

## 7.3 Reproducing

Trained models: `weights/obstacle/` ([weights/README.md](../weights/README.md)). Tables from the stored results, without
simulating:

```bash
python3 scripts/obstacle/dev2_table.py --runs results/obstacle/static_dev2     # static table (per-episode rows)
# moving obstacle: results/obstacle/dynamic_test/<policy>/index.csv, one row per test scene
```

Collection, training and evaluation commands: [usage.md](usage.md) section 7. In the code and data names, the improved
static recordings are "set B" (`OBSTACLE_HOLD_WHEEL_NOISE=0.5`), the original ones "recorded" or train2.
