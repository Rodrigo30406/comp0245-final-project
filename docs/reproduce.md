# Reproducing the report

Three levels, from seconds to days of compute. Commands from the project root with
`PY="pixi run --manifest-path ../RoboEnv/pixi.toml python"` (simulation) and `PYML="... -e ml python"` (training).

| level | what you get | needs | time |
|---|---|---|---|
| **1. tables from the evaluation summaries** | every reaching number of the report | this repository | seconds |
| **2. re-evaluate the trained models** | the closed-loop runs, traces and figures | `weights/` + the datasets of their splits (~64 GB, copy from the group's machine) | 15 min per evaluation of 205 episodes × 4 policies (16 workers) |
| **3. from scratch** | everything, including new datasets and models | RoboEnv, a GPU | ~1 h collection, ~30 GPU h training (all families × 5 seeds), ~20 h evaluation |

Each stage was checked against the report's artifacts: the target grid and the splits are regenerated identical,
collections reproduce the report's scenarios (and the obstacle recordings bit for bit; the expert itself is checked
against recorded lecturer episodes by the tests), retraining reproduces the weights bit for bit, and the evaluator
reproduces the closed-loop traces and metrics exactly.

## Level 1: tables

```bash
python3 scripts/evaluation/report_tables.py          # tables 1.1–1.4 of docs/results.md, from results/evaluations/
python3 scripts/evaluation/seed_stats.py results/evaluations/<dir> [...]     # any cell, mean ± sd (min–max)
```

## Level 2 and 3: table / figure → what to run

Evaluation directory names are those of the experiment records (`results/evaluations/`); "both banks" = run the same
command with `--split runs/splits/nominal_bank.json` and with `--split runs/splits/reaching_validation_random_start.json`.

| report item | models | evaluation | then |
|---|---|---|---|
| results 1.1, main table | the seven families of [03](03_models.md) 3.1, five seeds | both banks, protocol `validation.json`; +DB: `validation_deadband045.json` on the MLP-Ego-IC models | `seed_stats.py` |
| results 1.2, data controls | `v2_mlp28clean_*`, `v2_mlp28nom_trig144` | both banks | `seed_stats.py` |
| results 1.2, data design validation | `v2_mlp28_trig144_bal{128,256,512}`, `cap{400…2400}` | both banks | `summarize_eval.py` |
| results 1.3, generalisation | MLP-Ego-IC, Cat, GRU-Ego seeds 0–2 (+DB) | `--split runs/splits/bank_<far_targets|target_height|far_start>.json` (`make_bank_split.py` on the `ood_*` collections) | `summarize_eval.py` |
| results 1.4, robustness | MLP-Ego-IC, GRU-Ego seeds 0–2 | nominal bank, `--config experiment_specs/evaluation/validation_<adversity>.json` | `summarize_eval.py` |
| results 1.5, efficiency | all | — | `model_info.json` of each run |
| [02](02_inputs_and_outputs.md) 2.6, base parked / arm active | MLP-Base, MLP-Ego (+ copies) | nominal bank | `arm_base_decomposition.py` |
| [04](04_evaluation.md) offline vs closed loop | all families | nominal bank | `scripts/figures/offline_vs_closed_loop.py` |
| [05](05_arm_ignition.md) / [06](06_base_drift.md) figures | MLP-Ego, MLP-Ego-IC (+DB), Cat, GRU-Ego | nominal bank | `scripts/figures/problem_figures.py`, `phase_errors.py`, `arm_switch_probe.py` |
| report figures (trajectories, generalisation, geometry) | as above | as above | `scripts/figures/report_figures.py EPISODE OUT_DIR EVAL_DIR` |
| results 2, static obstacle | `scripts/obstacle/` (chapter 7) | group's harness on dev2 | `scripts/obstacle/dev2_table.py` |
| results 3, moving obstacle | `scripts/obstacle/dynamic_obstacle.py --policy` | seed 4, 100 scenes | `index.csv` of the run |

Example, the MLP-Ego-IC row of the main table from the shipped weights (level 2):

```bash
$PY scripts/data/make_train_val_split.py runs/splits/nominal_bank.json <datasets>/<…>_stratified1024_nominal_<…>
M=""; for s in 0 1 2 3 4; do M="$M --model ic_seed$s=weights/v2_mlp28_trig144_seed$s"; done
$PY scripts/evaluate_policy.py --split runs/splits/nominal_bank.json $M --allow-cross-split --allow-external-holdout \
    --workers 16 --output-dir runs/evaluation/ic_nominal
python3 scripts/evaluation/seed_stats.py runs/evaluation/ic_nominal        # 189.0 ± 9.3 (175–199) at 1 cm
```

From scratch (level 3): [01](01_data_collection.md) 1.9 (data, in order), [03](03_models.md) 3.5 (training commands),
[04](04_evaluation.md) 4.6 (evaluation commands).

## Known differences when reproducing

- **Figures**: rendering depends on the matplotlib version and fonts (same data, slightly different glyphs).
- **Training on another GPU or driver**: deterministic algorithms make a rerun identical on the same machine and software;
  another GPU model can change the last bits and, through the closed loop, a few arrivals per seed. The five-seed
  spread (sd 9 arrivals for MLP-Ego-IC) is the scale against which such differences should be read.
- **Re-collected datasets** contain the same scenarios and trajectories but get new directory names and file hashes
  (time stamps); the shipped weights must then be evaluated against the original datasets (level 2), or retrained.
