# Controlled experiments

This folder holds the controlled experiments behind the design of the final policy (MLP-Ego-TC, trained by the code
at the repository root). Each design question was isolated here and tested one variable at a time, in closed loop,
on fixed evaluation sets and under rules written down and hashed before the runs (pre-registration). The answers
were then carried into the final policy. The report cites every experiment below by its letter (Fig. 1).

| design choice in the final policy | question tested here | experiments |
|---|---|---|
| no measured wheel speed in the inputs | does the base copy its own previous command? | A |
| triggering copies of the onset row | why does the arm never start? is it that row, or just more rows? does a base/arm split fix it? | B, C3, R, S |
| DART noise and more data | where does more data help? | E (learning curve, stride) |
| model selection in closed loop only | does offline error rank the models? | offline vs closed loop (18 models) |
| a plain memoryless MLP | would EE-error inputs, DAgger, action chunking, a PD output layer or a bigger network do better? | D, F, G, P (+PC), W |
| the main result holds on new episodes | the final policy's weights in this harness, on fixed held-out sets | X (cross-evaluation) |
| extension: static obstacle | does an MSE base average the two detours and stall? does a side latch fix it? | step 9 (H-MM, H-FACT, H-COPY) |

## The copycat problem, in short

The expert's wheels are velocity controlled, so the measured wheel speed reproduces the previous wheel command
(R² = 0.99). A single-frame regressor given that input learns "keep doing what you did": it starts at rest, copies
zero, and the base never moves (experiment A; Wen et al. 2020, de Haan et al. 2019). Dropping the two wheel-speed
features (`ego` -> `ego-novw`) frees the base. The arm has the mirror problem: the reach starts on a single row per
episode (the first row after the base's done latch), every look-alike row is labelled "hold", so the regressor learns
"hold" (experiment B: with the expert driving the base, the learned arm never left rest in 1500/1500 runs). Appending
K jittered copies of that one row (triggering copies, C3) starts the arm; copies of a random row do not (R), and
splitting the policy into base and arm networks does not either, unless the arm net gets the copies (S).

## Report section -> code -> read-out

Paths are relative to this folder. `prereg_*_frozen.txt` files are the frozen pre-registrations (originals, in
Chinese, as hashed at freeze time); their English text is in `report.txt` under the same step.

| report section | experiment | code / run script | read-out |
|---|---|---|---|
| Problem formulation (latch, one-row onset, wheel-speed copy) | step 2 data analysis | `analyze_dataset.py`, `analyze_index.py` | `results/data_analysis/summary.txt` |
| Closed-loop evaluation and pre-registration | protocol v1.1; DEV / HARD / CONFIRM = seeds 200 / 201 / 202 | `evaluate_policy.py`, `analyze_rollouts.py`, `EVALUATION.md` | `results/expert_v11_{dev,hard,confirm}/` |
| Data collection | 2000 episodes, half with DART-style held noise | `collect_data.py`, `DATA_COLLECTION.md` | `report.txt` [Methodology] Formal dataset |
| Inputs, outputs and models | features `ego` (22), `ego-novw` (20); MLP 128x128 | `bc_features.py`, `train_policy.py`, `policies/bc_policy.py` | `trained/*.offline.json` |
| Triggering copies | K copies of the onset row with qd jitter | `train_policy.py build-trigger-cache`, `--trigger-copies` | `results/step3/C3_K_choice.json` |
| Offline fit vs closed loop | 18 models | `analyze_rollouts.py` | `results/step3/fig_step3_offline_vs_closed.pdf` |
| Copycat in the base | A: `ego` vs `ego-novw` | `results/step3/run_expA.sh`, `run_confirmA.sh` | `results/step3/A_ego_vs_ego-novw_{dev,hard,confirm}.txt` |
| Arm-onset failure | B: expert base + learned arm and vice versa | `results/step3/run_expB.sh`, `policies/hybrid.py` | `results/step3/fig_step3_split2x2.pdf`, `results/B-*` |
| | S: separate base and arm networks | `results/step11/step11.py`, `policies/split_policy.py` | `results/step11/analysis/SK*_vs_*.txt` |
| Triggering copies (results) | C3: `ego-novw` + 16 copies | `results/step3/run_expC3.sh`, `C3_probe.py`, `C3_mechanisms.py` | `results/step3/C3_mechanisms.txt`, `A_ego-novw_vs_ego-novw-trig_{dev,hard,confirm}.txt` |
| | R: copies of a random row | `results/step11/step11.py` | `results/step11/analysis/R_vs_*.txt` |
| | E: learning curve and stride | `results/step3/run_step4E.sh`, `train_step4e.py`, `step4E_analysis.py` | `results/step3/fig_step4_{learning_curve,stride}.pdf` |
| Cross-evaluation | X: the final policy's weights in this harness | `results/step10_cross/run_cross.sh`, `policies/rodrigo_policy.py`, `trained/rodrigo_v2/` | `results/step10_cross/cross_{dev,hard}.txt` |
| Negative results | D: end-effector error features | `results/step3/run_expD.sh`, `D_mechanisms.py` | `results/step3/D_mechanisms.txt` |
| | F: one round of DAgger; F0 tolerance curve | `results/step5/run_expF.sh`, `dagger_relabel.py` | `results/step5/F_mechanisms.txt`, `tolerance_curve.txt` |
| | G: action chunking + temporal ensembling | `results/step6/run_expG.sh` | `results/step6/G_pilot.txt` |
| | P: joint-reference output + fixed PD layer; PC: its confirmation | `results/step6/run_expP.sh`, `policies/pd_layer.py`, `results/step7/run_expPC.sh` | `results/step6/P_mechanisms.txt`, `results/step7/PC_mechanisms.txt` |
| | W: three 256-unit layers | `results/step11/run_step11_W.sh` | `results/step11/analysis/W*_vs_*.txt` |
| Static obstacle | L0, L1, L1-K0, L2; dev2 / confirm2 / test2 (test2 opened once) | `obstacle_task/`, `results/step9/run_step9_{sets,eval,test2}.sh` | `results/step9/step9_{dev2,confirm2,test2_final}.txt` |
| Fig. 1 | every comparison above | `figures/make_fig_effects.py` | `figures/fig_effects.pdf` |

Main models: C3 = `trained/ego-novw-trig_s{0..4}.npz` (DEV seeds) and `_s{100..104}` (CONFIRM seeds);
obstacle L2 = `trained/step9/L1arm_s*`, `L2cls_s*`, `L2base_s*`. Every run folder under `results/` has a
`summary.txt` and a `rollouts.csv` (one row per episode).

## Setup

The code runs in the course's RoboEnv pixi environment, from this folder:

```bash
export ROBOENV_DIR=/path/to/RoboEnv                 # your RoboEnv checkout (pixi.toml, models/)
cd controlled_experiments
ln -s "$ROBOENV_DIR/models" models                  # robot descriptions; configs/ is already here
PY="pixi run --manifest-path $ROBOENV_DIR/pixi.toml python"
```

`data/` (the recorded demonstrations and evaluation sets) is not in git; every set is regenerated deterministically
by `collect_data.py` from its seed (commands in `DATA_COLLECTION.md` and `EVALUATION.md`).

## Quick start: run the main model (C3) in closed loop

```bash
COMMON="--workers 20 --z-range 0.2 1.2 --theta-range -180 180 --init-radius 2.0 --init-yaw-range -180 180 --noise-frac 0.0"
$PY collect_data.py $COMMON --n-episodes 300 --seed 200 --r-range 0.1 3.0 --out-dir data/dev   # the DEV set
$PY evaluate_policy.py --policy expert --eval-set data/dev --episodes 0-19 --workers 20 \
    --out-dir results/quickstart_expert
$PY evaluate_policy.py --policy policies/bc_policy.py:make_policy --policy-arg trained/ego-novw-trig_s0.npz \
    --eval-set data/dev --episodes 0-19 --workers 20 --out-dir results/quickstart_c3
cat results/quickstart_c3/summary.txt
```

Add `--gui` and a single episode (`--episodes 7 --workers 1`) to watch it. To retrain C3 from the demonstrations:

```bash
$PY collect_data.py --n-episodes 2000 --workers 20 --seed 1 --r-range 0.1 3.0 --z-range 0.2 1.2 --init-radius 2.0 \
    --init-yaw-range -180 180 --noise-frac 0.5 --out-dir data/train
$PY train_policy.py build-cache                    # stride-9 rows of the split in trained/data_split.json
$PY train_policy.py build-trigger-cache            # the onset row of every training episode
$PY train_policy.py train --model ego-novw-trig --trigger-copies 16 --seed 0 --out trained/c3_retrained_s0.npz
```

The retrained arrays are bit-identical to `trained/ego-novw-trig_s0.npz` (checked in
`results/step5/backward_compatibility.json`). Tests: `$PY -m unittest discover -s tests` (322 tests, about 45 s);
the few that need regenerable caches or trajectories are skipped, with the reason, until those exist.

## Re-running the experiments

Each `results/step*/run_*.sh` is a resumable chain, one stage per call where the script lists stages: evaluation
stages skip every run folder that has a `summary.txt` (all committed runs do), analysis stages regenerate the
read-outs, and training stages refuse to overwrite a committed model. To re-run something from scratch, move its
models or run folders away and regenerate the data sets it uses (`DATA_COLLECTION.md`, `EVALUATION.md`,
`results/step9/run_step9_sets.sh`) and the feature caches (`train_policy.py build-cache`, `build-trigger-cache`).
Experiment B is run as `MODEL=ego-novw bash results/step3/run_expB.sh` (the arm adopted in experiment A); PC as
`bash results/step7/run_expPC.sh all --allow-confirm`. Re-running every chain in a fresh clone reproduces the
committed read-outs exactly, except figure metadata, recorded absolute paths and timestamps, and the few
descriptive numbers computed from per-episode logs or 1 kHz trajectories, which are not in git.

During the experiments, the chains of F, G/P and PC refused to start if the frozen plan, the notebook section or the
code had changed since the freeze (sha256 gates). This folder is a cleaned copy made after the experiments: comments
and messages translated into English, paths made relative, process files (execution logs, handover reports,
backups) removed; the program logic is unchanged (syntax trees equal with string literals masked). The frozen plans
are therefore absent here, and the gates print a "post-hoc re-run" notice and skip the hash comparison instead of
refusing. The frozen state, with the plans `STEP*_PLAN_PROMPT.md` that the code comments cite and with the strict
gates, is branch `feature/experimental-line-final` (commit 71ea436), folder `final/robotnik_arm_task/`.

## What is from the course template, what is modified, what is new

| status | files |
|---|---|
| course template, **unchanged** | `robotnik_arm_task.py`, `robotnik_arm_control.py` (the expert: base state machine + arm controller; byte-identical copies of `final/robotnik_arm_task/`), `configs/` (from RoboEnv) |
| course template, **modified** | none. The only change to the expert's behaviour, the base standoff pose on the line from the robot's start to the target instead of from the world origin, is implemented in our code (`evaluate_policy.expert_base_goal`, used by `collect_data.py`) |
| **written from scratch** | data: `collect_data.py`, `analyze_dataset.py`, `analyze_index.py`; evaluation: `evaluate_policy.py` (closed-loop harness, verified step by step against the recordings), `analyze_rollouts.py` (failure taxonomy, two-level bootstrap, Welch); learning: `bc_features.py`, `train_policy.py`, `train_step4e.py`, `dagger_relabel.py`, `policies/` (`rodrigo_policy.py` is an adapter that runs the final policy's weights in this harness); obstacle task: `obstacle_task/`; per-experiment scripts and read-outs: `results/`; `figures/`; `tests/`; documentation: this file, `DATA_COLLECTION.md`, `EVALUATION.md`, `report.txt` |

Not in git (regenerable): `data/`, `models/`, feature caches, 1 kHz trajectories, per-episode and console logs.
The data-analysis figures `report/*.png` come from `analyze_index.py`.
