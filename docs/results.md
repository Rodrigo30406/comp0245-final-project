# Results: tables for the report

All numbers of the report in one place, for the three tasks: reaching (the assignment task), static obstacle and
moving obstacle. Every cell is computed from the evaluation directories of the experiment records (`rodrigo` branch,
`runs/evaluation/`, `runs/obstacle/`) with `scripts/analysis/seed_stats.py` and `scripts/obstacle/dev2_table.py`; nothing
is copied from a single seed unless marked. Model names: see the progression in [03_models.md](03_models.md).

## Metrics and conventions

| metric | definition |
|---|---|
| **stable arrival within 1 / 2 / 5 cm** (reaching, primary) | the end effector stays within 1 / 2 / 5 cm of the target for 0.5 s at some point of a 40 s episode (1 kHz control, no stop rule: the policy runs the whole 40 s). Counted over the 205 held-out targets of a bank |
| offline NRMSE | per validation episode, RMSE of each output divided by the training std of that output, averaged half over the two wheels and half over the seven torques; mean over episodes. 0.13 = 13 % of the spread of the demonstrated commands |
| seeds | five training seeds (0–4) unless marked; cells are **mean ± sd (min–max)** over the seeds; no cell reports the best seed alone |
| banks (reaching) | **nominal**: 205 held-out grid targets from the lecturer's start (origin, facing +x); **random start**: 205 held-out grid targets, start uniform in [−2, 2]² m with any heading. The expert reaches 205 / 205 and 200 / 205 of them |
| obstacle tasks | see sections 2 and 3 |

---

## 1. Reaching

### 1.1 Main table: the progression of models (five seeds, stable arrivals of 205)

| model | inputs | nominal 1 cm | nominal 2 cm | nominal 5 cm | random start 1 cm | random start 5 cm | offline NRMSE |
|---|---|---|---|---|---|---|---|
| expert | — | 205 | 205 | 205 | 200 | 200 | — |
| MLP-Base | 21 + clock | 1.6 ± 3.6 (0–8) | 4.6 ± 9.2 (0–21) | 27.4 ± 22.2 (16–67) | 0.8 ± 1.8 (0–4) | 23.2 ± 19.0 (12–57) | 0.159 |
| MLP-Ego | 28 (egocentric) | 2.0 ± 1.6 (0–4) | 7.8 ± 3.6 (2–11) | 39.0 ± 4.6 (31–42) | 1.4 ± 1.1 (0–3) | 38.2 ± 4.2 (31–41) | 0.120 |
| **MLP-Ego-IC** (ignition copies, K 144) | 28 | **189.0 ± 9.3 (175–199)** | 197.6 ± 3.0 (194–202) | 201.0 ± 2.0 (199–204) | **192.6 ± 8.7 (180–200)** | 202.0 ± 2.0 (199–204) | 0.138 |
| MLP-Ego-IC + DB (wheel dead band 0.45 rad/s) | 28 | 189.2 ± 9.3 (174–198) | 197.6 ± 5.2 (190–204) | 200.6 ± 2.7 (197–204) | 185.6 ± 12.6 (164–197) | 200.4 ± 1.8 (198–202) | 0.138 |
| MLP-Ego-Cat (categorical wheels + K 144) | 28 | 188.2 ± 11.3 (176–199) | 196.8 ± 6.2 (190–202) | 202.0 ± 1.4 (201–204) | 189.0 ± 9.8 (179–200) | 203.4 ± 1.1 (202–205) | 0.149 |
| GRU-Ego (memory, no copies) | 28 | 149.2 ± 23.9 (121–186) | 183.2 ± 24.1 (154–203) | 196.8 ± 12.2 (177–205) | 132.4 ± 24.5 (103–168) | 194.4 ± 13.6 (172–205) | 0.106 |
| TF-Ego (causal transformer, no copies) | 28 | 158.4 ± 43.2 (85–200) | 171.6 ± 41.6 (100–204) | 185.0 ± 27.9 (137–205) | 156.2 ± 49.1 (73–197) | 181.6 ± 31.9 (128–203) | 0.105 |

In percent of 205 (1 cm, nominal): MLP-Ego-IC 92 ± 5 %, +DB 92 ± 5 %, Cat 92 ± 6 %, GRU-Ego 73 ± 12 %, TF-Ego 77 ± 21 %,
MLP-Ego 1 %, MLP-Base 1 %.

Reading: the offline error is inversely related to closed-loop success (the best-fitting models, GRU and transformer,
arrive less often than the MLP with ignition copies; the no-DART control in 1.2 fits best of all and never arrives).
The two steps of the progression: the egocentric transforms lower the offline error (0.159 → 0.120) and bring the base
to the dock, but the arm never starts reaching (end effector stuck ~0.2 m from the target,
[figures/problems/4_arm_error.png](figures/problems/4_arm_error.png)); the ignition copies start the arm (2 → 189 at 1 cm).

### 1.2 Data controls (five seeds, MLP-Ego-IC recipe unless marked)

| training data | nominal 1 cm | nominal 5 cm | random start 1 cm | random start 5 cm | offline NRMSE |
|---|---|---|---|---|---|
| full mixture, DART (reference = MLP-Ego-IC) | 189.0 ± 9.3 | 201.0 ± 2.0 | 192.6 ± 8.7 | 202.0 ± 2.0 | 0.138 |
| same scenarios **without DART** | 0.0 ± 0.0 | 2.8 ± 1.5 | 0.0 ± 0.0 | 1.8 ± 1.6 | 0.095 |
| same scenarios without DART, MLP-Ego (no copies) | 0.6 ± 0.9 | 27.6 ± 17.2 | 0.2 ± 0.4 | 19.0 ± 8.1 | 0.085 |
| **nominal starts only** (721 episodes) | 152.4 ± 24.8 (115–178) | 196.8 ± 3.2 | 61.8 ± 14.6 (40–79) | 106.6 ± 20.9 | 0.182 |

Data size (MLP-Ego-IC, seed 0; random subsets of the 3 221 training episodes keeping the mixture's proportions):

| training episodes | 400 | 800 | 1 600 | 2 400 | 3 221 |
|---|---|---|---|---|---|
| nominal 1 cm | 142 | 148 | 177 | 195 | 199 |
| random start 1 cm | 137 | 127 | 174 | 187 | 200 |
| offline NRMSE | 0.209 | 0.189 | 0.142 | 0.142 | 0.139 |

Balanced coverage (one target per coarse grid block, every start pose of each target kept; seed 0):

| targets (episodes) | 128 (506) | 256 (1 011) | 512 (2 000) | 819 (3 221, full) |
|---|---|---|---|---|
| nominal 1 cm | 80 | 150 | 190 | 199 |
| random start 1 cm | 92 | 145 | 190 | 200 |
| offline NRMSE | 0.222 | 0.155 | 0.141 | 0.139 |

At equal episode count, fewer well-spread targets with four start poses each are worse than more scattered targets (506
balanced: 80; 400 random: 142): the curve is driven by the number of distinct targets. Both curves reach the full-set
level at ~500 targets / ~2 000 episodes (512 balanced: 190 / 190, inside the five-seed range of the full set, 175–199).

### 1.3 Generalisation beyond the training range (three seeds, seeds 0 / 1 / 2; stable arrivals of 205)

Training range: targets x, y ∈ [−2, 2] m, z ∈ [0.3, 0.8] m; starts x, y ∈ [−2, 2] m. Evaluation banks, never trained on:

| bank | expert | MLP-Ego-IC 1 cm / 5 cm | MLP-Ego-IC + DB 1 cm / 5 cm | MLP-Ego-Cat 1 cm / 5 cm | GRU-Ego 1 cm / 5 cm |
|---|---|---|---|---|---|
| far targets (max(\|x\|, \|y\|) 2.2–3 m) | 205 | 205 / 196 / 197; 205 / 205 / 205 | 202 / 198 / 205; 204 / 205 / 205 | 178 / 184 / 194; 205 / 204 / 205 | 111 / 61 / 118; 160 / 189 / 205 |
| target height outside the range (z 0.15–0.28 and 0.85–1.05 m) | 205 | 18 / 41 / 16; 157 / 77 / 129 | 22 / 35 / 17; 159 / 75 / 112 | 31 / 14 / 22; 113 / 110 / 151 | 16 / 6 / 14; 131 / 83 / 114 |
| far starts (x 2.3–3.5 m, y ±3.5 m, any heading) | 205 | 200 / 195 / 201; 200 / 204 / 205 | 196 / 198 / 204; 196 / 204 / 205 | 150 / 177 / 198; 175 / 202 / 205 | 85 / 59 / 103; 132 / 180 / 205 |

Reading: the MLP extrapolates in the plane (far targets and far starts at the in-range level, because the drive is
steered by the heading inputs and the stop by the target-in-base features, which are in range near the goal) but not
in height: outside the demonstrated z band the arm reaches the 5 cm shell and almost never 1 cm. The GRU degrades on
every bank (its memory integrates longer drives than any it was trained on).

### 1.4 Robustness to adversities (nominal bank, three seeds 0 / 1 / 2, stable arrivals of 205)

| adversity | expert | MLP-Ego-IC 1 cm | MLP-Ego-IC 5 cm | GRU-Ego 1 cm | GRU-Ego 5 cm |
|---|---|---|---|---|---|
| none | 205 | 199 / 193 / 175 | 202 / 204 / 199 | 138 / 121 / 151 | 177 / 193 / 205 |
| left wheels execute 95 % of the command | 182 | 199 / 194 / 180 | 203 / 203 / 199 | 133 / 107 / 150 | 175 / 190 / 205 |
| payload 0.1 kg on the last arm link | 204 | 201 / 199 / 188 | 203 / 203 / 204 | 141 / 109 / 157 | 177 / 190 / 205 |
| payload 0.25 kg | 205 | 191 / 195 / 153 | 203 / 205 / 204 | 121 / 80 / 119 | 187 / 190 / 205 |
| payload 0.5 kg | 205 | 50 / 36 / 37 | 202 / 200 / 199 | 69 / 18 / 64 | 191 / 190 / 203 |
| observation noise, 1/10 (2 mm, 0.002 rad base; 0.001 rad, 0.005 rad/s arm) | 205 (reads the true state) | 34 / 103 / 10 | 61 / 141 / 15 | 149 / 143 / 148 | 189 / 204 / 205 |
| observation noise, 1/2 | 205 | 10 / 17 / 4 | 26 / 28 / 7 | 199 / 184 / 177 | 204 / 202 / 204 |
| observation noise, full (2 cm, 0.02 rad base; 0.01 rad, 0.05 rad/s arm) | 205 | 7 / 8 / 5 | 16 / 15 / 11 | 127 / 149 / 119 | 195 / 187 / 198 |
| command delay 2 ms | 204 | 0 / 0 / 0 | 1 / 0 / 1 | 0 / 0 / 0 | 0 / 0 / 0 |
| command delay 5 ms | 129–186 | 0 | 0 | 0 | 0 |
| command delay 50 ms | 0 | 0 | 0 | 0 | 0 |

Reading: wheel calibration error and a payload up to 0.25 kg leave the MLP at its nominal level (with 0.5 kg it keeps
the 5 cm arrivals and loses the last centimetre: the median closest approach of the hand goes from 3.8 to 7.3 mm); observation noise is the axis where the GRU is
clearly more robust than the MLP (its memory filters the noise; the MLP's torques chatter and the arm is thrown);
neither learner tolerates any command delay (the expert itself fails at 5 ms and above).

### 1.5 Efficiency (median over five seeds)

| model | parameters | inference, one step (µs, CPU, NumPy) | best epoch | training time (min, one GPU) |
|---|---|---|---|---|
| MLP-Base | 20 617 | 36 | 38 | 47 |
| MLP-Ego / MLP-Ego-IC | 21 385 | 35 | 41 / 38 | 29 / 27 |
| MLP-Ego-Cat | 31 705 | 40 | 27 | 27 |
| GRU-Ego | 81 929 | 33 | 286 | 20 |
| TF-Ego | 81 993 | 31 | 236 | 22 |

All run far below the 1 ms control period. Training on 35 M rows (3 221 episodes at 1 kHz); the MLP's peak memory is
the dataset in float32.

---

## 2. Static obstacle (lidar)

**Task** (the group's obstacle task, protocol v2.0): one cylinder between the start and a target 3.5–5 m away, a 64-beam
planar lidar added to the observations (variant `raw21_targetpolar_clock_lidar64_v1`, 92 inputs); the expert drives
around it through a waypoint, docks and reaches. **Evaluation bank dev2**: 300 scenes (seed 300), 60 s episodes. Training
set train2: 2000 scenes (seed 3), collected with the hold (10 s after the first reach) and our DART noise (sets D / B / F,
[01_data_collection.md](01_data_collection.md) section 1.6). Three seeds per row unless marked.

| metric | definition |
|---|---|
| success | end effector within 10 mm of the target at some moment, no contact |
| held 10 s | after reaching, within 15 mm for 10 s |
| stable 1 cm | within 1 cm for 0.5 s |
| docked clean | base within 0.2 m of the docking ring and still for 1 s, before any contact |
| collisions | robot-cylinder contacts |

Percent of the 300 dev2 scenes, mean ± sd (min–max):

| model, training set | success | held 10 s | stable 1 cm | docked clean | collisions |
|---|---|---|---|---|---|
| expert | 100 | 100 | 100 | 100 | 0 |
| group's baseline L2 (side classifier + base and arm networks), seed 0 | 60.0 | 9 | 49 | 97 | 0 |
| MLP-Ego-IC + lidar, recorded set (stops at the reach) | 54 ± 6 (48–61) | 8 ± 4 | 45 ± 8 | 93 ± 5 | 0 |
| MLP-Ego-IC + lidar, hold, group's noise | 62 ± 14 (53–78) | 18 ± 17 | 55 ± 16 | 97 ± 2 | 3 ± 3 |
| MLP-Ego-IC + lidar, set D (hold, our DART) | 73 ± 2 (70–74) | 59 ± 3 | 71 ± 3 | 89 ± 5 | 1 ± 2 |
| **MLP-Ego-IC + lidar, set B (hold, our DART + base noise in the hold)** | **80 ± 8 (71–87)** | **77 ± 9 (67–85)** | 80 ± 8 | 90 ± 5 | 1 ± 0 |
| MLP-Ego-IC + lidar, set F (B + base noise faded near the dock) | 77 ± 26 (47–97) | 71 ± 26 (44–96) | 76 ± 26 | 93 ± 3 | 1 ± 1 |
| MLP-Ego-IC + lidar, set B + DB 0.45 | 47 ± 22 (26–71) | 45 ± 23 | 47 ± 22 | 89 ± 5 | 0 |
| GRU-Ego + lidar, recorded set | 13 ± 3 (9–15) | 1 ± 1 | 7 ± 2 | 97 ± 1 | 8 ± 8 |
| GRU-Ego + lidar, set B | 55 ± 9 (45–60) | 38 ± 5 (32–42) | 51 ± 8 | 99 ± 1 | 0 |
| TF-Ego + lidar, recorded set (700 training episodes, memory) | 12 ± 4 (7–15) | 0 | 8 ± 3 | 97 ± 2 | 0 |

Reading: the obstacle itself is solved (0–1 % contacts, docked in 90–99 %); the failures are the last centimetre and the
hold, as in reaching. The hold data alone do not fix the hold (18 %); keeping the base perturbed during the hold, so the
demonstrations show the arm compensating a moving base, does (8 → 77 % held). Set F has the best single run (seed 1:
290 / 300 = 97 %) and the widest spread; set B is the stable choice. The wheel dead band, which helps in reaching, hurts
here (80 → 47 %): it removes the small final turn and the base parks misaligned. Limitations and untried fixes:
[07_static_obstacle.md](07_static_obstacle.md).

---

## 3. Moving obstacle (preliminary, one seed)

**Task**: reaching (target 3.5–5 m away) while one cylinder (r 0.25–0.35 m) crosses the start→dock path at
0.10–0.25 m/s, timed to meet the robot. Expert: the one-cost DWA-style planner of `scripts/obstacle/dynamic_obstacle.py`.
Success = reached within 1 cm and still within 1 cm after a 10 s hold, no contact; 75 s cap. **Test**: 100 fresh scenes
(seed 4); development scenes seeds 1–2; training set seed 3 (1 600 expert episodes with DART noise, all successful, 0
contacts).

| policy | success | contacts | not reached | reached then left | navigation time (median) |
|---|---|---|---|---|---|
| expert (development scenes, 42) | 41 / 42 | 0 | 0 | 1 (hand at 10.5 mm) | 26 s |
| **expert (test, 100)** | **100** | 0 | 0 | 0 | — |
| blind template (no avoidance, test) | 46 | 54 | 0 | 0 | — |
| MLP-Ego-IC + lidar, seed 0 | 34 | 26 | 35 | 5 | — |
| GRU-Ego + lidar, seed 0 | 0 | 33 | 66 | 1 | — |

| MLP-Ego-IC + lidar, by scene type | succeeds | collides |
|---|---|---|
| the 54 scenes where the blind template collides (avoidance needed) | 16 | 22 |
| the 46 scenes where it does not | 18 | 4 |

Offline validation NRMSE: MLP 0.239, GRU 0.200 (static obstacle: MLP 0.11–0.16, GRU 0.08–0.10).

Reading: the expert solves the task (100 / 100, 0 contacts); a single-scan MLP learns part of the avoidance (it halves
the contacts of the blind template and succeeds in 16 scenes that need avoidance), but it does not learn the recovery
after a detour: in 35 scenes the base wanders away from the dock (final distance median 2.3 m) with the cylinder far
behind. The GRU fits the data poorly and never finishes. One seed: a preliminary result, reported as future work
(DAgger is directly applicable here, because this expert can label any state).
