# Full native validation: complete corrected series

All **64/128/256/512** uniform/stratified cohorts are measured. Each model has
complete native-bank and fixed26-anchor evaluation; explicit external-bank guards
reject every condition found in any compared model training set. No final test
or multi-seed replication is implied. Corrected256 has52 native held-out goals.

| Episodes | Design | Own native goals | Valid | Task arrivals at1cm | Base-pose arrivals |
|---:|---|---:|---:|---:|---:|
| 64 | uniform MLP | 13 | 13 | 0 | 0 |
| 64 | stratified MLP | 13 | 13 | 0 | 0 |
| 128 | uniform MLP | 26 | 26 | 0 | 0 |
| 128 | stratified MLP | 26 | 26 | 0 | 0 |
| 256 | uniform MLP | 52 | 52 | 0 | 0 |
| 256 | stratified MLP | 52 | 52 | 0 | 0 |
| 512 | uniform MLP | 103 | 103 | 0 | 1 |
| 512 | stratified MLP | 103 | 103 | 0 | 1 |

See [full-series findings](SAMPLING_SERIES.md) for both policies on both banks,
OLS/expert references, common anchor and native union metrics, physical errors
and paired plots. A base-pose dwell is a strategy diagnostic, not task success
or proof of holding until40s. Incomplete runs remain in success denominators.
