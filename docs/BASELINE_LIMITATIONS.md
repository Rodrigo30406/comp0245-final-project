# Observed baseline limitations

## L1 — Unbounded commands and degradation in autonomous control

Recorded on 25 October 2026. This is the first limitation in this list, not a
claim that it is the sole or main cause of every failure. OLS and MLP have linear
outputs without amplitude constraints. Training on normalized, bounded labels
does not mathematically bound their predictions. The evaluator clips commands
externally and records raw requests and applied commands separately.

**Timing matters:** do not claim that both models produce enormous commands from
the start. OLS exceeds a wheel limit at t=0 in 10/13 scenarios; in the first case
it requests 5.760 rad/s against a 5.273 rad/s limit, a moderate excess. MLP starts
that case at (+2.182, −2.206) rad/s, within bounds; torque clipping starts at
0.347 s and wheel clipping at 0.552 s. Extreme commands appear later as the
rollout diverges. Imitation differences already exist in the first command.

### Verification on expert states

The frozen models were evaluated without clipping or retraining on **all**
490730 rows from 51 training episodes and 127147 rows from 13 validation episodes
in the original split. Model, split and NPZ hashes were checked using the shared
loader. Each maximum is over all channels in its group; wheel and torque units
are never combined into a single magnitude.

| Partition | Command source | Maximum absolute wheel command (rad/s) | Maximum absolute torque (Nm) |
|---|---|---:|---:|
| Training | Expert labels | 3.000 | 30.000 |
| Training | OLS prediction | 6.309 | 49.831 |
| Training | MLP prediction | 3.926 | 37.304 |
| Validation | Expert labels | 3.000 | 30.000 |
| Validation | OLS prediction | 6.326 | 46.318 |
| Validation | MLP prediction | 4.300 | 32.830 |

For evaluation limits of ±5.272727 rad/s for wheels and ±30 Nm for torques,
these are the percentages of training rows exceeding a limit in any group channel:

| Model | Any wheel | Any torque |
|---|---:|---:|
| OLS | 0.0882 % | 0.0840 % |
| MLP | 0 % | 0.2119 % |

These are pooled row proportions on expert states, not episode averages, applied
commands or evidence of autonomous stability. Exceeding the expert's observed
range (maximum 3 rad/s in this dataset) differs from exceeding the evaluation
limit (5.273 rad/s). Labels and predictions are distinct: labels respect the
expert's rules, whereas predictions can exceed them even on training inputs.
MLP has lower wheel maxima than OLS but more rows with some torque above 30 Nm;
maximum magnitude and frequency of excess describe different aspects.

**Measured conclusion:** expert-state predictions do not reach the enormous
amplitudes observed during autonomous divergence. Moderate excesses do occur,
even on training states. This is compatible with degradation through feedback:
the model changes the robot's state and subsequently receives inputs different
from the demonstrations. It does not establish which variables leave the data's
coverage, when that happens, or that this is the only cause. State distribution
and hidden controller memory require separate diagnosis.

### Evidence and next controls

The reproducible audit and complete per-channel minima, maxima, absolute P99,
and exceedance counts are stored locally in:
`runs/diagnostics/offline_output_ranges_20261025/{audit.py,ranges.json}`.
Artifacts are ignored by Git; no model or data was changed. To repeat the audit,
copy the script to another runs directory first: it refuses to overwrite
ranges.json. Run from the project root in RoboEnv with `PYTHONPATH=src`.

Clipping remains an explicit part of the protocol; it does not correct the wrong
direction or guarantee stopping. A bounded output must be assessed as a separate
variant. Do not increase limits to hide saturation. See
[saturation and orientation](SATURATION_AND_ORIENTATION.md) for timing and
sampling limitations. At the time of this audit P2 remained in diagnosis; it was
subsequently closed as an exploratory pilot (research log R17). No supervisor or
new training was introduced by this audit.

## L2 — Stateful expert labeling is unresolved for DAgger

Recorded on 26 October 2026 at Rodrigo's request. This is a methodological
limitation of a proposed extension, not a measured DAgger result. DAgger has not
been implemented or evaluated in this project.

DAgger collects observations visited by the learned policy and asks the expert
for action labels, then fits the policy on the aggregated labeled dataset. The
parameter updates remain supervised; collecting on-policy observations does not
make the method reinforcement learning. See [Ross et al. (2011)](https://proceedings.mlr.press/v15/ross11a.html).

Our expert is stateful: its action depends on the observation **and** controller
memory, including navigation phases, `nav.done`, and the integrated arm reference
`q_des`. The 21 baseline inputs do not explicitly contain this memory. Therefore,
an arbitrary learner observation alone does not specify a unique expert label
independently of history. Reinitializing the expert at every query would discard
that history and would not reproduce the original controller's semantics.

Before applying DAgger, define and verify a causal protocol for initializing and
updating an expert instance along each learner-driven episode, once per physical
step, including time, phase transitions, and reference integration. Distinguish
the commands queried from the expert from the commands actually applied by the
learner. Audit whether the resulting expert can recover from learner-visited
states; its availability does not guarantee useful recovery labels. Do not
silently alter its rules to obtain corrections or feed privileged expert memory
to the baseline policy. Any such change requires a declared variant.

Even a consistent labeling protocol does not establish that a memoryless MLP
with the original inputs can reproduce the stateful expert. Added history or
memory inputs remain separate ablations. Keep validation/test scenarios outside
aggregation and control the additional labeling budget in comparisons.

The assignment's slide 3 describes fully supervised learning from expert
trajectories. DAgger retains expert-supervised training but extends that data
collection protocol. Compatibility with the wording is an interpretation, not
confirmed lecturer approval; the slides do not explicitly include or exclude
DAgger. Source: [assignment slides](COMP0245-Assignment-2026-2.pdf).
