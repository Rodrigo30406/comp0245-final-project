# MLP stopping audit: complete sampling series

Stopping monitors validation episode-macro balanced NRMSE, not training loss.
Retain the lowest-validation checkpoint. Patience resets on an improvement larger
than min_delta=0.0001 relative to the last meaningful-improvement score; tiny
improvements can update the best weights without resetting patience. Stop after
8 stale epochs or at the original60-epoch cap.

| Design | Episodes | Epochs attempted | Selected epoch | Stop reason |
|---|---:|---:|---:|---|
| uniform | 64 | 22 | 21 | patience |
| uniform | 128 | 17 | 9 | patience |
| uniform | 256 | 24 | 16 | patience |
| uniform | 512 | 60 | 58 | max_epochs |
| stratified | 64 | 11 | 3 | patience |
| stratified | 128 | 19 | 11 | patience |
| stratified | 256 | 45 | 37 | patience |
| stratified | 512 | 35 | 27 | patience |

Any max_epochs stop is a budget limit, not convergence evidence. The original
uniform512 remains the60-epoch run with best58; it was not stopped by patience.
No extended-budget training has been mixed into this sampling experiment.

[All validation curves](../runs/sampling_comparison/full_series_20261026/training_stopping_full_series.png).
The curves use distinct native banks/scales, not one shared test. Exact continuation
would require Adam optimizer state, which existing inference artifacts do not
store; an extended seed-matched run would start over unless full optimizer
checkpointing is added. Full training metrics and limits: [series](SAMPLING_SERIES.md).

## Larger-budget intervention authorized and running

Rodrigo subsequently requests enough epochs to stop by patience, plus 1024 and
2048 episodes for both designs. A separate recipe, `mlp_patience.json`, removes
the epoch cap while retaining patience 8 and min_delta 1e-4. The capped uniform
512 is being rerun from the original seed and exact normalizers; seven existing
patience-stopped models are reused. Full Adam/RNG checkpoints are now stored for
new fits. The original study above remains frozen. See the current protocol and
execution status in [the patience-only extension](PATIENCE_SERIES.md).


## Patience-only extension completed

Rodrigo subsequently authorized removing the cap and extending both designs to
1024/2048. Historical stopping facts above stay frozen. All12 current MLPs now
stopped by patience, including a separate rerun of uniform512 with exact60-epoch
prefix reproduction. Full epoch table and measured offline/online outcomes:
[PATIENCE_SERIES.md](PATIENCE_SERIES.md).
