# Best model so far: MLP 28 pure + 144 trigger copies on mixture v2 (report, 3 Nov 2026 06:00)

Status: interim, updated 3 Nov 13:40. The GRU sweep (§6ab of `BLACKBOX_PLAN_20261027.md`) and the parking remedies (§6ac)
are done and summarised where they bear on this model; nothing here uses the reserved test banks. Everything below is on the development validation geometries.

## 1. The model

| | |
|---|---|
| Run | `runs/training/v2_mlp28_trig144_seed0` (seeds 1 and 2: `…_seed1`, `…_seed2`) |
| Kind | stateless MLP regressor, 128 × 128 ReLU, 21 385 parameters, PyTorch, CUDA, deterministic |
| Inputs (28) | the lecturer's 21 observations (base x, y, z, bearing; 7 arm angles; 7 arm velocities; target x, y, z) plus 7 pure transforms of them: target in the base frame (x, y), target height above the base, planar distance, cos / sin of the heading error to the target, episode clock fraction (`raw21_targetpolar_clock_v1`) |
| Outputs (9 → 11) | 2 wheel velocities (left / right, each duplicated to the two wheels of its side) + 7 arm joint torques; no extra heads |
| Control period | 1 ms, the same as the expert; no memory, no previous command as input |
| Training data | mixture v2: 3 221 expert-success episodes (35.4 M rows) |
| Offline fit | validation episode-macro balanced NRMSE 0.139 (best epoch 27 of 35, 36 min on the GPU) |

Black-box compliance: every input is a function of the lecturer's observations and the task specification (target), the
two output groups are the arm and the wheel commands, and the one re-weighting device (trigger copies, §4) is derived from
the demonstrated wheel labels alone: no expert phase, no `base_navigation_done`, no simulator state.

## 2. The numbers (closed loop, 40 s episodes, 205 geometries each, stable arrival = inside the band for 0.5 s)

| | 1 / 2 / 5 cm | held ≥ 10 s | arm extended | base parked | drift after parking | inside the expert's 8 cm box at 40 s |
|---|---|---:|---:|---:|---:|---:|
| **Nominal start (unchanged since 27 Oct)** | | | | | | |
| seed 0 | **199 / 202 / 202** | 202 | 189 | 145 | 0.06 m | 30 |
| seed 1 | 193 / 198 / 204 | 182 | 199 | 175 | 0.25 m | 3 |
| seed 2 | 175 / 194 / 199 | 172 | 180 | 150 | 0.14 m | 52 |
| expert | 205 / 205 / 205 | 204 | 154 | 205 | | 129 |
| **Random start pose (v2 bank, start-referenced goal)** | | | | | | |
| seed 0 | **200 / 203 / 204** | 200 | 175 | 133 | 0.05 m | 38 |
| seed 1 | 197 / 203 / 203 | 185 | 195 | 169 | 0.20 m | 17 |
| seed 2 | 187 / 196 / 199 | 188 | 183 | 144 | 0.12 m | 53 |
| expert | 200 / 200 / 200 | 198 | 159 | 200 | | 128 |

Under the teammate's protocol v1.1 (touch 10 mm, no dwell, 100 ms traces): success 97.6 / 94.1 / 86.3% nominal and 97.6 /
96.6 / 91.7% random; reach 5 cm 97.6–99.5%; held 10 s 95.1 / 47.3 / 41.0% nominal. Their C3 reported 52–70% success.

Where it still differs from the expert: the base parks 10–25 cm from the expert's spot in most episodes (inside the 8 cm
box in 30 / 3 / 52 of 205 vs 129) and seed 1 keeps a slow residual wheel command (0.25 m drift, moving at 40 s in 129
episodes). The arm absorbs the base error; the task metric does not see it. §6ac.1 (3 Nov 11:06) fixes it: the 0.45 rad/s
wheel dead band at the output gives 198 / 195 / 190 at 1 cm with drift 0.07 m, inside the box 118–139 and held 10 s equal to the
arrivals, and categorical wheels + 144 copies do the same inside the model (176 / 176 / 195 at 1 cm, 201 / 201 / 204 at 5 cm).

## 3. How we got here: the chain of decisions

Each step was pre-registered in `BLACKBOX_PLAN_20261027.md` before its result, and each failure was explained with a
measurement before the next step. In order:

1. **Decomposition and pure inputs (§1, §6e).** The 21 raw inputs alone give a policy that drives but never reaches (0–2 at
   1 cm). Seven pure transforms of the target into the base frame (`raw21_targetpolar_clock_v1`) unlock the base and the arm
   on the training geometries without privileged information; the clock is the only non-geometric input.
2. **Noise injection (DART, every dataset).** The expert is recorded while executing its own command plus N(0, 0.5 rad/s)
   on the wheels and N(0, 1.0 Nm) on the arm torques, held for 50 ms; the labels are the clean commands. Without it the
   policies left the demonstrated states within seconds.
3. **Mixing random start poses (E1 / E3, §6d, §6f).** 1024 nominal-start geometries + 3072 random-start episodes (3 starts
   per target). Required for the random-start validation; alone it did not fix the two failure modes below.
4. **Diagnosing the two failures (§6c, §6h, §6i, §6t.1).** (a) The base never emits an exact zero: a regressor's "zero" is
   a small residual that creeps the base 0.2–0.5 m after parking. (b) The arm starts only when it is already moving: the
   demonstrated reach begins from a *resting* arm on about one row per 12 000, so the regressor learns "arm at rest ⇒ hold"
   and the arm stays home unless the creeping base jostles it. Measured with an offline probe (`arm_switch_probe.py`): the
   policy's arm output on the hold → reach axis at its own parked state was −0.02 (control) with the expert's base pose and
   clock making no difference; only the arm state did.
5. **Stop remedies that did not work (E7, E8; §6l–§6w).** Loss weight on zero rows, L1 wheels, categorical wheels, previous
   command as input, near-target starts (E2), evaluator dead band, learned gate, MLP→GRU encoder, transformer. All were
   measured with a sleeping arm; the ones that produced an exact zero (categorical, gate, GRU latch, dead band) then showed
   that the base reads the resting arm as "keep driving" and the arm reads the stopped base as nothing. The coupling, not
   the zero, was the problem.
6. **The ignition row (E9, §6x, §6y).** The teammate's C3 branch duplicated the first `nav_done` frame 16 times with arm-velocity
   jitter and went from 0.3% to 52–70% success. §6x explained it with step 4(b): the copies re-weight the one resting-arm
   reach row per episode. We implemented it in our trainer from the labels alone (first row of the terminal run of exactly
   zero wheel labels, equal to the expert's done flag in 78 of 80 episodes), swept K ∈ {16, 144, 512} at 1 kHz with a
   pre-registered selection rule, and chose **K = 144** with jitter σ_j = std of the arm velocities over the moving rows
   (0.004–0.085 rad/s per joint, fixed rng seed 0). On the v1 data: 96 / 142 / 132 at 1 cm; probe +0.96 at the parked state.
7. **Two data rules (§6z).** (a) **Train and select on expert successes only**: the v1 mixture carried 217 training episodes
   where the expert timed out at 25 s circling 8 cm outside its own tolerance and then held zero wheels 2.5 m from the goal
   with the arm reaching at 1.1 m (605k rows, 5% of all zero-wheel rows); failed scenarios were re-collected with a new
   start / noise seed and discarded if they failed again. (b) **Goal referenced to the start pose**: the lecturer's expert
   takes the goal bearing from the world origin, where his robot always starts; with a random start that sends the robot
   around the target to the far side of the standoff ring. `goal_reference: start` (a generalisation of the reference
   point, bit-identical for the nominal start) gives the near side. Mixture v2 = 819 nominal + 2 806 multistart v2 + 322
   replacements = 3 221 successes (887 validation geometries, 802 successes).
8. **K = 144 on mixture v2 (§6z.1).** 199 / 193 / 175 at 1 cm nominal, 200 / 197 / 187 random; held 10 s follows the arrivals
   for the first time. The ablation (§6z.2) separates the two data rules on the same 205 nominal geometries: dropping the
   failures alone gives 120 / 134 / 153 (and 121–157 instead of 103–108 on random starts); the start-referenced goal makes the
   jump to 175–199, and also removes a drive-away mode (a v1-successes seed ended 1.9 m from the target after reaching).

## 4. The one device, stated plainly

Trigger copies: for each training episode, the first row whose demonstrated wheel command is exactly zero and stays zero
to the end (the stop / reach onset) is appended 144 more times with N(0, σ_j) noise on its seven arm-velocity features,
labels unchanged; 463 824 rows added to 35.4 M (1.3%). The scaler is fitted on the recorded rows only. It is a re-weighting
of the demonstrations by a label-derived rule, equivalent to a per-row loss weight of 145 plus input noise (the weight
alone gives 150 / 181 / 183: §6aa.2). It uses no information the dataset does not already contain, but it is an engineering
decision about which row matters, and the report should say so. The GRU 28 trained on v2 with no such device reaches
73–78 at 1 cm with the arm igniting in all three seeds (§6aa.2) and 121–151 (5 cm 177–205) once trained to patience 40
(§6ab.2, the best model of the programme with no device: seed 2 gives 151 / 198 / 205 and holds 10 s in 193). That is the
reference for what the device buys: about 50 arrivals per seed at 1 cm, the ignition rate (arm extended 180–199 against
133–184) and an arrival inside 1 cm rather than between 1 and 2 cm.

## 5. What did not matter, and what to be careful about

- **Offline NRMSE ranks the models backwards** across the whole programme: the v2 control fits best (0.119) and arrives
  once; K 144 fits worse (0.139) and arrives 199 times. Model selection is closed loop on validation geometries only.
- **Seeds.** Three per arm, always reported separately. Seed spread at 1 cm: 175–199 on v2 (96–142 on v1).
- **Action chunking, trigger weight without jitter, dead band on v1** were tried after K 144 and did not improve on it
  (§6aa.1, §6aa.2).
- **Protocol differences with the teammate**: our 1 cm / 2 cm / 5 cm stable arrivals with 0.5 s dwell are stricter than their
  10 mm touch; our arm-extended flag counts an arm leaving home (ours travels 30 cm more than the expert's); their metrics
  from our traces are approximate at 100 ms (the end effector moves 0.18 mm median per 100 ms inside the 15 mm band).
- **Reserved test banks** (`datasets/test_reserved/*`) have never been loaded by a trainer, a selection step or an analysis.

## 6. Reproduction

```
pixi run --manifest-path ../RoboEnv/pixi.toml -e ml python scripts/train_egocentric.py \
  --dataset datasets/dart_stratified1024_low/20261027T060538Z_dart_stratified1024_wheel0_5_torque1_0_hold50_97ab75da \
  --split runs/splits/v2_mixture.json --config experiment_specs/training/mlp_pytorch_successes.json \
  --output-dir runs/training/v2_mlp28_trig144_seed0 --horizon-s 40 --device cuda \
  --inputs raw21_targetpolar_clock_v1 --option trigger_copies=144
pixi run --manifest-path ../RoboEnv/pixi.toml -e ml python scripts/evaluate_policy.py \
  --split runs/training/mlp_stratified1024_torch_baseline/split.json \
  --model mlp28_trig144_seed0=runs/training/v2_mlp28_trig144_seed0 --allow-cross-split --allow-external-holdout \
  --config experiment_specs/evaluation/validation.json --workers 8 \
  --output-dir runs/evaluation/v2_mlp28_trig144_on_clean_validation_20261102
```

Seeds 1 and 2 use `mlp_pytorch_successes_seed1.json` / `_seed2.json`. Datasets and splits are in the artifact tiers
(`docs/ARTIFACT_SYNC.md`); the training source snapshot and hashes are in each run directory. Code at commit `38d5905`
(trainer) and `3a2d55a` (this report's numbers).
