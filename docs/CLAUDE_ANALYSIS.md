# External review of the `rodrigo` branch — Claude analysis

Date: 26 October 2026. Reviewer: Claude (Fable 5.1), at Rodrigo's request.
Scope: read-only review of the code, experiment specs and research log at
commit `315354d`. No experiment was run and no result below is claimed as
measured; every proposal is a hypothesis to test under the project's own
provenance rules. Written in English per the repository language rule.

## 1. What is working well

- One episode engine (`simulation/episode.py`) shared by the interactive
  simulator, the collector and the autonomous evaluator. No duplicated loop.
- Provenance is thorough: frozen scenarios, hashed NPZ episodes, frozen splits,
  source snapshots, exact Torch/NumPy inference agreement, 57 passing tests.
- The evaluator separates raw from applied commands, records the lecturer's
  first stop decision without stopping, and uses a fixed horizon. Failures are
  preserved rather than filtered.
- The research log is honest: 0/154 and 0/410 closed-loop arrivals are
  reported as such, and no causal story is overclaimed.

The engineering is not the bottleneck. The scientific framing of the failure is.

## 2. Main diagnosis: the expert is a hybrid system with hidden state

`expert/controllers.py` makes the expert action depend on three pieces of
controller memory that the 21 baseline inputs do not contain:

| Hidden state | Effect on the label | Where |
|---|---|---|
| `BaseNavState.phase` (align / drive / orient / done) | Same base pose yields turn-only, drive, or zero wheels depending on phase; phase depends on the whole history | `base_controller` |
| `nav.done` | Gates `reach_active`: before it the arm holds `q_hold`, after it the arm reaches. Same `(q, qd, target)`, two different torques | `arm_controller`, `Episode.transitions` |
| Integrated reference `q_des` | Enters the PD law with `kp = 100`. It is clipped to `q ± 0.15 rad`, so the label ambiguity from this term alone can reach ±15 N·m against a 30 N·m limit | `arm_controller` |

A regressor trained with MSE learns the conditional mean of the action given
the observation. When one observation maps to several expert actions, the mean
is a blend that the expert never emits. This is a partial-observability and
multimodality problem, not a capacity problem. It predicts three things that
the log already shows:

- Low supervised error with zero closed-loop success (R17, R40).
- The 200 ms window has lower validation NRMSE but diverges more (R40). A short
  window cannot recover phase, which depends on elapsed time and on whether the
  tolerance was ever met.
- Deeper or wider MLPs are unlikely to change the outcome (R41 campaign).

The research plan already lists the relevant remedies (option O privileged
diagnostics, F1 relative geometry, DART, gravity residual). The gap is
prioritisation, not ideas.

## 3. Proposals, in priority order

### P1. Privileged-input ablation (diagnostic, cheap, one or two days)

The episode NPZ already stores `expert_phase` and `q_des` (see
`Episode.transitions` yield and `data/storage.py`). Train the reference
128×128 MLP on the same 819/205 split with extra inputs: phase one-hot,
`nav.done`, and `q_des` (7 values). Evaluate on the same 410 paired goals.

- If success rises from zero, the hidden-state cause is demonstrated with a
  single controlled experiment. This is the kind of understanding the rubric
  rewards at 70+.
- If it stays at zero, the cause lies elsewhere (compounding at 1 kHz,
  coverage), and the temporal and architecture campaigns become more relevant.

Declare it as a diagnostic variant; never present it as the deliverable policy.

**How the oracle gets the phase at run time.** The learned policy cannot know
the expert's phase, so the closed-loop oracle needs a *shadow expert*: a second
`BaseNavState`, `q_des` and `StopCriteria` instance that is updated every step
from the robot's actual state along the learner's trajectory, whose action is
discarded. Its phase and `q_des` are fed to the oracle policy. This is the same
protocol question raised in limitation L2, used here only as a diagnostic.
Two stages:

1. Offline only, no shadow expert needed: compare validation error with and
   without privileged inputs. The drop measures how much label variance the
   hidden state explains.
2. Closed loop with the shadow expert: measures the success ceiling once the
   mode ambiguity is removed.

**Deployable surrogates for the hidden state** (not privileged, all computable
by the policy itself):

- Episode clock `t`. The expert uses it for the 25 s timeout; a controller
  always knows its own clock. Cheap and legitimate input.
- Base-goal geometry. `goal` and `goal_bearing` are deterministic functions of
  the target alone (standoff table in `Episode._initialize`). Distance to the
  base goal and bearing error to `goal_bearing` make the align/drive/orient
  decision observable. Fits inside P3.
- Memory for the hysteresis (`done` stays done): either the 200 ms window plus
  the clock, or a learned mode head that predicts `nav.done` from observable
  history and feeds it to the action head.

The oracle tells us which of these surrogates is worth building; the surrogates
are what the final policy can actually use.

### P2. Noise injection during collection (DART)

Resolves limitation L2 without DAgger. The expert instance stays alive across
the whole episode, so its clean label is well defined at every step even when
a perturbed action is applied. Record the clean action as label and the
perturbed one as applied; keep the expert's own state updates untouched.

Today every episode starts from the nominal pose, so coverage of
off-trajectory states is essentially zero. DART gives recovery data at the cost
of one collection campaign. Start with small Gaussian noise on wheels and
torques, two or three noise levels, same scenarios.

### P3. Egocentric feature variant (not privileged)

Add deterministic functions of the 21 inputs: target expressed in the base
frame, planar distance to the target, heading error as sine and cosine, and
bearing as sine and cosine. The expert decides on exactly these quantities, so
the mapping becomes close to piecewise linear.

This also fixes a concrete defect: raw yaw jumps at ±π, and the target box
includes negative x, where goal bearings sit near ±π and the robot crosses the
discontinuity. The plan mentions 22 inputs with sine/cosine; the code still
feeds raw yaw (`Episode.observe`).

### P4. Learn the torque residual over gravity compensation

`tau_g` depends only on `q_pin`, which is a function of the observed base pose
and joint angles. Predict `tau - tau_g(q)` and add `tau_g(q)` back at inference.
Same 11-output interface, same information, much lower label variance.
Pinocchio is already a runtime dependency. Section 12 of the plan requires the
cost of this analytic term to be counted; measure it.

### P5. Model selection with a closed-loop criterion

Early stopping currently uses row-uniform validation NRMSE. R40 shows this does
not track closed-loop success. Options, cheapest first:

- Phase-balanced or group-weighted NRMSE (wheels vs torques weighted by their
  closed-loop sensitivity).
- A short closed-loop proxy at each candidate checkpoint: about 8 rollouts of
  10 s. The evaluator already supports `--limit` and a custom horizon.

### P6. Reserve the final test set now

AGENTS.md states that no final test set is reserved, and all 410 validation
goals have already been used for checkpoint selection. Freeze a held-out set
of targets and initial conditions before any confirmatory run, including
interpolation, extrapolation and new-start cases, and never use it for
selection.

### P7. Replan against the 6 November deadline

Ten days remain and the in-person demo window opens on 30 October. Suggested
order: P1 and P6 first, then P2 and P3 together, then P4 only if the arm
remains the limiting subsystem. Let the running temporal campaign finish but
pause the depth/shape sweep. A negative result explained by P1 is worth more
in the report than six policies that fail the same way.

### P8. Recurrent policy (GRU or LSTM), after P1–P3

A recurrent network carries its own state from episode start, so it can learn
the `done` hysteresis and approximate the `q_des` integration. It addresses
identifiability, the cause described in section 2. It does **not** address
coverage: trained with teacher forcing on expert trajectories it still never
sees off-trajectory states and drifts in closed loop. Run it after DART (P2),
not instead of it.

Practical design, given 40 000 steps per episode at 1 kHz:

- Decide at 20 ms (the spacing already chosen for the window) and hold the
  action between decisions; sequences become about 2 000 steps.
- Truncated BPTT over 2–4 s chunks, carrying the hidden state across
  consecutive chunks of the same episode.
- Reset the hidden state at every episode, as `OnlineHistory` does today.
- Same 819/205 split, same scalers, parameter budget comparable to the
  2×128 window model, so memory is the only treatment.

Report it as a ladder that answers slide 5's "instant versus interval"
question directly: memoryless MLP → 200 ms window → recurrent. Run it only if
P1 shows hidden state matters and the cheap surrogates (clock, base-goal
geometry) leave a gap.

## 4. Secondary observations

- **Decision rate.** The policy acts at 1 kHz, 40 000 times per episode, on top
  of a stiff PD. Small torque errors compound fast. Holding each predicted
  action for k steps (for example 20 ms) is a declared variant worth one
  measurement; it also matches the 20 ms window spacing already chosen.
- **Wheel clipping above 70 % of steps** in all learned rollouts. A bounded
  output (scaled tanh to the expert command envelope) is cheap to try and is
  the natural companion of P3 and P4.
- **Documentation volume.** 42 documents and roughly 14 000 added lines for a
  six-page report. Worker benchmarks, CPU frequency and core affinity studies
  are useful as a log but consume time that P1–P3 need. Consider freezing the
  operational docs and keeping only RESEARCH_LOG and the protocol docs live.
- **Known expert quirks to cite, not fix:** joint 6 of the nominal posture sits
  below its declared lower limit; stall detection precedes success in
  `check_stop`; `max_steps` may run one extra step. All are already recorded.

## 5. Verification status

Verified on 26 October 2026 on a fresh machine with RoboEnv installed from
`pixi.toml` (environments `default` and `ml`, torch 2.14.1 with CUDA 13.0):

- RoboEnv smoke test passes (`pinocchio`, `pybullet`, `casadi`,
  `simulation_and_control`).
- Full project suite: **57 tests pass in 55 s** when run as
  `PYTHONPATH=src python -m unittest discover -s tests` from the `ml`
  environment, which includes the actual CPU/CUDA Torch training tests.

One portability defect found while doing so. Without `PYTHONPATH=src`, two
tests fail at import (`test_command_fidelity`, `test_control_quality`) and
the other 55 pass only because `test_data_collection` happens to insert `src`
into `sys.path` earlier in alphabetical discovery order. Only three test files
set the path themselves. The suite is therefore order-dependent and the README
command (`python -m unittest discover -s tests -v`) does not work on a clean
checkout. **Fixed on 27 October 2026:** every test module now inserts `src/`
into `sys.path` (the pattern three of them and all `scripts/` launchers already
used), and `scripts/run_tests.py` wraps discovery. Verified: the README
command without `PYTHONPATH`, the two formerly failing modules run alone, and
the launcher run from an unrelated directory all pass 57/57.

Not verified: datasets, runs and models are local and ignored by Git, so no
number in the research log was recomputed here.

## 6. Status after the 27 October experiments

The proposals above were executed on 27 October 2026 (overnight run plus a
morning replication batch). Full log, every number and its run directory:
`docs/NIGHT_RUN_20261027.md`; the pre-registered ablation protocol and its
result: `docs/PRIVILEGED_ABLATION.md`. Datasets, splits and run directories
stay local (ignored by Git); the recipes and code that produce them are
committed.

| Proposal | Outcome |
|---|---|
| P1 privileged ablation | Confirmed. Hidden phase + `nav.done` + integrated `q_des` reduce offline error by 38 %; error concentrates after `nav.done`. |
| P2 DART | Implemented (`action_noise` in collection config). Low noise (σ 0.5 rad/s, 1.0 N·m, held 50 ms) removes saturation and runaway in every architecture; medium noise is too strong. |
| P3 egocentric features | Implemented (`learning/egocentric.py`, 25 inputs = 14 joint + 10 relative geometry + episode clock). Necessary for every policy that arrives; the 21 raw inputs never arrive, with or without memory or DART. Rodrigo's lecturer asks that the 21 prescribed inputs be kept and extended, not replaced: an "all inputs" variant (21 + 10 + clock = 32) is the next experiment. |
| P5 closed-loop selection | Decisive. Offline NRMSE does not rank policies (best closed-loop policy has one of the worst offline errors). Every model was evaluated closed loop on 205 validation goals. |
| P6 reserved test set | Two frozen banks of 64 goals (nominal and random start), never used. |
| P8 recurrent policy | GRU with memory updated every 20 ms and a head reacting every 1 ms, on egocentric inputs and DART-low data, is the best recipe over 5 seeds: 123–154 of 205 stable arrivals at 5 cm versus 23–100 for the equivalent MLP. Memory updated every 1 ms fails (hidden-state drift). |
| Open problem | Holding: every learned policy arrives and then leaves; none ends within 1 cm at 40 s. The post-arrival regime is where the expert relies on its latched `nav.done`. Candidate fixes: an "arrived" latch as a deployable input, an auxiliary phase head, DAgger. |
