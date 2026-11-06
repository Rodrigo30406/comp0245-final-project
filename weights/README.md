# Trained models of the report

Every trained policy behind the tables of [docs/results.md](../docs/results.md), as portable files (NumPy, no PyTorch
needed to run them). Each directory is a complete training run minus its source copy and epoch history:

| file | content |
|---|---|
| `model.npz` | weights, input/output normalisation, input variant, clock (no pickle) |
| `model_info.json` | kind, inputs, parameters, data filter, epochs, offline validation error, inference time |
| `config.json` | the training recipe as resolved (seed, options) |
| `metrics.json` | offline errors (training and validation, per output) |
| `run.json` | status, command, SHA-256 of the model, split and datasets |
| `split.json` | link to the shared split in `splits/` (the hash in `run.json` matches) |

Directory names are the experiment-record names (`v2_<family>_seed<N>`; seed 0 = training seed 20261025, seeds 1–4 =
1–4). Five seeds of every family of the report:

| directory family | report name | inputs | parameters |
|---|---|---|---|
| `v2_mlp22raw_control_seed{0..4}` | MLP-Base | 21 + clock | 20 617 |
| `v2_mlp28_control_seed{0..4}` | MLP-Ego | 28 (egocentric) | 21 385 |
| `v2_mlp28_trig144_seed{0..4}` | **MLP-Ego-IC** (also the +DB variant: same model, dead band at evaluation) | 28 | 21 385 |
| `v2_mlp28_cat41_trig144_seed{0..4}` | MLP-Ego-Cat | 28 | 31 705 |
| `v2_gru28_patience40_seed{0..4}` | GRU-Ego | 28 | 81 929 |
| `v2_tf28_p40_seed{0..4}` | TF-Ego | 28 | 81 993 |
| `v2_mlp28clean_control_seed{0..4}`, `v2_mlp28clean_trig144_seed{0..4}` | controls without DART (MLP-Ego, MLP-Ego-IC) | 28 | 21 385 |
| `v2_mlp28nom_trig144_seed{0..4}` | control: nominal starts only (MLP-Ego-IC) | 28 | 21 385 |

Single seed (seed 0): `v2_mlp28_trig144_bal{128,256,512}` (data design validation, balanced coarser grids),
`v2_mlp28_trig144_cap{400,800,1600,2400}` (random subsets), `v2_mlp28_trig16`, `v2_mlp28_trig512` (number of ignition
copies), `v2_mlp22raw_trig144` (raw inputs with copies).

Splits: `splits/training_mixture_split.json` (the training mixture of every model except the controls),
`clean_mixture_split.json` (no DART), `nominal_only_split.json`.

## Using a model

```bash
pixi run --manifest-path ../RoboEnv/pixi.toml python scripts/run_policy.py --model weights/v2_mlp28_trig144_seed0 --target 1.5 1.0 0.6
```

```python
import json, numpy as np
from robot_learning.evaluation.worker import load_policy
run = "weights/v2_mlp28_trig144_seed0"
policy = load_policy(json.load(open(f"{run}/model_info.json"))["kind"], f"{run}/model.npz")
policy.reset(0.001)                         # once per episode (the clock and the GRU / transformer memory restart)
command = policy.predict_action(obs21)      # once per 1 ms step: 21 observations -> 11 commands
```

Observations: base x, y, z, bearing, seven arm joint angles, seven arm joint velocities, target x, y, z. Commands:
wheels front-right, front-left, back-left, back-right (rad/s), then the seven arm torques (N·m). The evaluator clips them
to the expert's envelope (±5.27 rad/s, ±30 N·m).

Evaluating these files again (`scripts/evaluate_policy.py --model LABEL=weights/<run>`) needs the datasets of their
split (the evaluator checks that no evaluated scenario was used in training): copy them under `datasets/` or set
`DATASETS_ROOT`; they are found by directory name. See [docs/reproduce.md](../docs/reproduce.md).
