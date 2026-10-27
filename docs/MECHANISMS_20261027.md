# Why the results come out the way they do — mechanism diagnostics

Date: 27 October 2026, afternoon. Every claim below is backed by a
measurement on existing artefacts (expert source code, the DART-low and clean
datasets, the trained models, the closed-loop traces); nothing new was
trained. Numbers refer to the DART-low recipes unless stated. Companion to
`docs/NIGHT_RUN_20261027.md` (what happened) — this file is about *why*.

## 0. What the expert actually computes (`src/robot_learning/expert/controllers.py`)

Base (`base_controller`): a four-phase switch keyed on thresholds.

| Phase | Condition to leave it | Command |
|---|---|---|
| align | `|heading_err| < 0.05` or `dist < 0.08` | turn in place: `w = clip(3·heading_err, ±1)`, `v = 0` |
| drive | `dist < 0.08` or (`dist < 0.15` and `cos(heading_err) < 0`) | `v = clip(dist, 0.05, 0.25)·max(0, cos err)`, `w = clip(1.5·err, ±0.3)` |
| orient | `|bearing_err| < 0.05` → `done = True` | turn in place: `w = clip(3·bearing_err, ±1)`, `v = 0` |
| done (latched) | never (also forced by `nav_timeout`) | **exactly zero wheels, whatever the base error** |

with `heading_err = wrap(atan2(goal − base) − yaw)`, `bearing_err = wrap(goal_bearing − yaw)`,
`goal = target − standoff(target_z)·(cos, sin atan2(target_y, target_x))`.

Arm (`arm_controller`): holds `q_hold` until `done`; then differential IK on the
world-frame target with a reach limit (0.85 m from the shoulder), integrating
a joint reference `q_des` (hidden state) tracked by a PD + gravity law.

So the expert's wheel command is a function of **three relative angles and
distances plus one latched bit**; the arm's torque depends on an **integrated
reference** that is not in the observation. Everything below follows from this.

## 1. Why the 21 explicit inputs never work, and why the relative geometry does

Probe (`Diagnostic B1`): take 20 validation start states, sweep the base yaw
over the full circle (align phase, the expert turns toward the goal with
`w = clip(3·heading_err)`), and compare each model's turn-rate with the law.

| Model (DART low) | turn-rate RMSE on real early states | RMSE, yaw shifted ±0.3 rad | RMSE, full yaw sweep | sign agreement (sweep) |
|---|---:|---:|---:|---:|
| MLP 21 explicit | 0.056 | 0.400 | 1.008 | 0.55 |
| MLP 25 egocentric | 0.037 | 0.052 | 0.067 | 1.00 |
| MLP 32 (21 + geometry + clock) | 0.029 | 0.120 | 0.146 | 1.00 |

On the training manifold all three fit the law equally well (that is why
their offline errors are similar). Off it, the 21-input network does not
represent the law at all: with the yaw rotated, its turn direction is
right 55 % of the time — chance. A 128×128 ReLU network has to build
`wrap(atan2(goal − base) − yaw)` out of seven raw coordinates, through two
`atan2`, a table lookup and an angle wrap, and the 819 demonstrations only
visit a thin curve of that 7-D space. The moment the closed loop deviates by a
few tenths of a radian (which every rollout does), the network is asked for a
value it never learned. With the geometry features the law is almost linear
(`w ≈ 3·heading_err`), so a small network generalises it everywhere.

Consequence: the 21-input baseline's runaway is not an optimisation or data
quantity issue; it is a representation issue, and DART (which widens the
manifold by a few centimetres and tenths of a radian) cannot cover a full
circle of yaw. This is why "21 + DART" and "21 + clock" still fail (9 and 5–10
arrivals).

## 2. Why the clock alone does nothing

The phases are **state-triggered**, not time-triggered. In the clean
dataset `nav.done` fires at 5.7 / 7.8 / 9.8 s (p10/50/90); in the DART-low
dataset at 5.7 / 8.4 / 25 s (std 6.4 s; the 25 s tail is the navigation
timeout forcing `done` when noise prevents the orient phase from finishing).
Predicting `done` per step from the best single clock threshold gives
77 % (DART) / 86 % (clean) accuracy; the instantaneous geometry rule
`dist < 8 cm and |bearing_err| < 0.05` gives 87 % / 98 %. The clock is a weak
proxy for phase, and it gives no help at all with the control law itself
(section 1). Its permutation importance in the 32-input model is non-zero
for the wheels (0.60), consistent with a mild phase prior, but it cannot
carry the task.

## 3. Why DART removes saturation and runaway

The egocentric model trained on clean data clips the expert envelope on
27–33 % of steps and 25 of 205 episodes end beyond 10 m; the same model on
DART-low data clips 0 % and no episode ends beyond 10 m. The mechanism is the
standard one: clean data contains only states exactly on the expert's path, so
the regression is unconstrained one centimetre off it and extrapolates
linearly to huge commands; the perturbed trajectories (σ 0.5 rad/s on the
wheels, 1 N·m on the torques, held 50 ms) put labelled states in the tube
around the path, and the label there is the expert's *corrective* action, so
the learned map is contracting near the path. Medium noise
(σ 1.0 / 2.0) makes the *expert* fail in 46 % of demonstrations and removes the
end of the task from the data, which is why it trained worse models.

## 4. Why every policy arrives and then leaves (the hold problem)

Three facts, measured:

1. **The expert's post-arrival behaviour is a discontinuity.** After `done`
   the wheel command is exactly 0 for 30 s regardless of base error, while
   one step earlier (orient phase) it is `3·bearing_err`, mean 0.5 rad/s. In
   the DART-low training data, states near the goal with
   `0.05 < |bearing_err| < 0.5` carry label 0 in 60 % of steps (done, pushed
   by noise) and label 0.48 rad/s in 40 % (orient). A regressor must separate
   them from the state alone.
2. **Offline, every model does separate them** (`Diagnostic C1`): mean
   predicted |turn| is 0.003–0.034 rad/s in the done-and-pushed states versus
   0.45–0.49 in orient states, for the 21-, 25- and 32-input MLPs alike. The
   discriminating cue is the **arm**: after `done` the arm leaves its hold
   pose, so `arm_q_*` carry the latch (they are the top permutation
   importances for the torques, and significant for the wheels). So the
   policies do know when to hold.
3. **But "small" is not "zero", and it integrates.** In closed loop the mean
   |wheel command| after arrival is 0.06–0.19 rad/s (expert: 0.000). Over
   the 30 s after arrival this creep moves the base 6–36 cm and rotates it
   3–13°; the arm follows the moving target (EE-in-base moves 8–22 cm) but
   the torque regression is at its worst exactly here — validation NRMSE on
   torques is 0.04 in align, 0.07 in drive and **0.29–0.31 in orient/done**,
   because the expert's torque depends on the integrated `q_des` reference
   (the hidden state identified by the privileged ablation, −38 % error when
   given). The end-effector leaves the 5 cm ball after a median 4–13 s
   (`Diagnostic C2`); at the moment of leaving the base has moved 1–4 cm
   and the arm 0.3–4 cm, so both channels contribute, with the base creep
   as the trigger and the arm's poor tracking as the amplifier.

The expert never leaves (205 / 205) because its wheels are *exactly* zero and
its arm reference is integrated. No regressor on instantaneous state outputs
an exact zero over a region, which is why "arrive and stay" is the hardest
metric for every recipe and why a latched "arrived" bit as an **input**
(section 7) is the natural fix: it would let the network learn a separate,
near-constant branch for the hold regime instead of a smooth blend.

## 5. Where the remaining non-arrivals happen (`Diagnostics D, E`)

Taxonomy of the 205 closed-loop episodes:

| Policy | arrive & stay | arrive & leave | base within 15 cm, EE never arrives | base never near |
|---|---:|---:|---:|---:|
| MLP 25 egocentric, 5 seeds | 0–61 | 23–78 | **104–181** | 0–1 |
| MLP 32, 5 seeds | 17–96 | 41–145 | **42–90** | 0–2 |
| GRU 25, memory 20 ms, 5 seeds | 2–20 | 108–135 | 51–82 | 0–1 |
| MLP 21 explicit + DART | 0 | 9 | 182 | 14 |

The base reaches the neighbourhood of its goal in practically every episode
for every recipe. The dominant failure is the **drive-through**: in the
"base within 15 cm, EE never" episodes the base ends 0.5–2 m from the goal,
60–100 % of them are still moving at 40 s, and the arm extended in 0–78 %.
The policy enters the stop/orient region, does not complete the switch
(section 4 again: orient is a turn-in-place with `v = 0` triggered by a
threshold), drifts out of the 8 cm disc, re-enters a drive-like regime with
heading errors never seen at those states, and leaves. Off-manifold probe of
the orient law (turn-in-place toward the target bearing) gives turn-rate RMSE
0.33–0.62 rad/s for every model: the orient phase is the least covered regime
of the data (the arrival bearing error is determined by the approach
direction, so each demonstration shows a narrow slice of it).

## 6. Why memory helped with 25 inputs and stopped helping with 32 — and the 1 ms / 20 ms confound

With the 25 inputs the GRU halved the drive-through count (51–82 vs
104–181) and doubled the arrivals; with the 32 inputs the MLP reaches the
same drive-through count (42–90) without memory, and the GRU on 32 inputs is
no better. So the GRU's contribution was a *state-dependent latch* (its
hidden state keeps "I have arrived" while the instantaneous geometry
dithers around the threshold), and the 32-input MLP achieves the same from
state alone. Memory's cost shows in the hold metric: the GRUs leave more
(108–135 arrive-and-leave) because the hidden state drifts with the policy's
own creep (section 4 applies to the memory as well as the robot).

Honesty note on "memory every 20 ms works, every 1 ms fails": the two runs
differ in *two* things. Truncated BPTT uses 1000 decisions per chunk, which is
**1 s of gradient horizon at 1 ms** and **20 s at 20 ms**. A latch that must
persist 30 s cannot be learned through a 1 s gradient window; the 20 ms run
sees the whole navigation phase in one chunk. The faster hidden-state drift
of the 1 ms memory is the second difference. Separating them would require a
1 ms run with 20 000-decision chunks (memory ≈ 20× larger); not done.

## 7. What is explained and what is not

Explained with evidence: the failure of the 21 inputs (representation
off-manifold, §1); the uselessness of the clock (§2); the effect of DART (§3);
the universal hold failure (§4); where the residual failures occur (§5); the
role and limits of memory (§6).

**Not explained yet: why 32 inputs beat 25.** Offline, per regime and per
output group, the two are indistinguishable (wheels 0.095 vs 0.087, torques
0.141 vs 0.145); in the off-manifold probes of the align law the 25-input
model is actually better; in the near-goal stop-rule probe they tie
(stop accuracy 0.68–0.77 vs 0.63–0.79). Permutation importance of the seven
absolute inputs in the 32-model is near zero except `base_position_y`
(0.38 on the wheels) and `base_bearing` (0.15). The gain is real (5 seeds,
every column) and concentrated in the orient/stop regime (§5), but the
probes built today do not isolate the cause. Candidate hypotheses, each
testable with a 5-seed ablation on the same data:

- H1: redundancy — a second, independent encoding of "at the goal" (absolute
  base position ≈ the goal's absolute position, which is deterministic given
  the target) makes the stop decision robust where the relative features
  pass through zero and change sign. Test: 25 + base position only.
- H2: the arm — the IK target lives in the world frame; absolute target and
  yaw give the torque head a frame it was implicitly using. Test: 25 + target
  xyz + yaw only. (Against it: torque errors are identical offline.)
- H3: optimisation — a wider, correlated input simply conditions the problem
  better, lowering seed variance. Test: 25 + 7 random/irrelevant columns.

## 8. Implications for the next step

The one defect shared by every recipe is the hold regime, and it is a
discontinuity the state cannot express exactly. The cheapest deployable fix
is a latched `arrived` input bit (set when `dist < 8 cm and |bearing_err| <
0.05`, never reset), i.e. the expert's own `done` computed from the
observations the policy already has. It should (a) make the hold regime a
separate branch with an exact zero target for the wheels, and (b) give the
arm head the reach trigger explicitly. The second lever is the arm's hidden
reference `q_des`, for which the deployable surrogates are memory (a GRU on
the arm channel only) or an explicit integrator feature. DAgger remains the
general remedy for the orient-phase coverage problem (§5), since no amount of
perturbation of expert rollouts produces the policy's own approach
directions.
