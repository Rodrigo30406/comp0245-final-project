# Trained weights (shared with the team)

Portable artifacts of the models that appear in the report, copied from `runs/training/` (git-ignored) on 4 Nov 2026.
Each directory is a complete trainer run minus its source snapshot and epoch history: `model.npz` (NumPy weights and
normalisers, no pickle), `model_info.json` (kind, inputs, parameter count, latency, stopping), `config.json` (the
training recipe as resolved), `metrics.json` (offline validation metrics), `run.json` (status and SHA-256 of the
artifacts). `split.json` is a symlink to the shared split in `v2/splits/` (identical bytes across the runs that share
it; the hash in `run.json` matches).

## Models

Mixture v2: 3 221 expert successes (nominal start + random start, goal referenced to the start pose), DART noise at
collection, 802 validation episodes. "clean" = the same mixture collected without DART (4 000 episodes). Stable
arrivals at 1 cm (0.5 s dwell) on the 205 held-out nominal geometries / the 205 random-start geometries, seeds 0 / 1 / 2;
the expert scores 205 / 200.

| directory | model | parameters | nominal | random start |
|---|---|---|---|---|
| `v2_mlp28_trig144_seed{0,1,2}` | MLP, 28 inputs, 144 ignition copies (the report's main policy) | 21 385 | 199 / 193 / 175 | 200 / 197 / 187 |
| `v2_mlp28_control_seed{0,1,2}` | same MLP without the ignition copies | 21 385 | 1 / 0 / 2 | 1 / 0 / 1 |
| `v2_gru28_patience40_seed{0,1,2}` | GRU, 28 inputs, 20 ms memory stride, no device | 81 929 | 138 / 121 / 151 | 118 / 103 / 140 |
| `v2_tf28_p20_seed{0,1,2}` | causal transformer, 28 inputs, 20 ms stride, 20 s window, no device | 81 993 | 166 / 85 / 104 | 153 / 73 / 113 |
| `v2_mlp28clean_trig144_seed{0,1,2}` | MLP + 144 copies trained on the clean (no-DART) mixture | 21 385 | 0 / 0 / 0 | 0 / 0 / 0 |
| `v2_mlp28clean_control_seed{0,1,2}` | MLP without copies, clean mixture | 21 385 | 0 / 0 / 0 | 0 / 0 / 0 |

Full tables and the mechanisms: `docs/BEST_MODEL_REPORT_20261103.md`, `docs/REPORT_PLAN_20261103.md`.

## Loading a policy

The policies run in NumPy (no PyTorch needed). The input is the lecturer's 21-vector in this order: base x, y, z,
bearing, the seven joint angles, the seven joint velocities, target x, y, z. The output is the 11-vector of commands:
wheels (front right, front left, rear left, rear right, rad/s) then the seven joint torques (N m); the evaluator clips
it to the expert's envelope (±5.27 rad/s, ±30 N m). Call `reset` at the start of every episode (the GRU and the
transformer keep state; the MLP keeps the 20 ms clock of its transformed inputs) and `predict_action` once per 1 ms
physics step.

```python
import json, numpy as np
from robot_learning.learning.egocentric import EgocentricPolicy      # kind egocentric_mlp_regressor
from robot_learning.learning.gru_policy import GRUPolicy             # kind gru_regressor
from robot_learning.learning.transformer_policy import TransformerPolicy  # kind transformer_regressor

run = "weights/v2/v2_mlp28_trig144_seed0"
kind = json.load(open(f"{run}/model_info.json"))["kind"]
cls = {"egocentric_mlp_regressor": EgocentricPolicy, "gru_regressor": GRUPolicy, "transformer_regressor": TransformerPolicy}[kind]
policy = cls.load(f"{run}/model.npz")
policy.reset(dt=0.001)
action = policy.predict_action(observation_21)   # -> np.ndarray of 11 commands
```

`src/` must be importable (`PYTHONPATH=src` or `pip install -e .`). With the project's evaluator the run directory is
passed as `--model LABEL=weights/v2/<directory>`; that path also needs the source dataset named in `split.json` for its
leakage checks, so on a machine without the datasets use the loader above.
