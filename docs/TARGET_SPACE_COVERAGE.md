# Target-space coverage: corrected complete series

Measured **64/128/256/512** target clouds exist for uniform and stratified designs.
[Full target comparison](../runs/sampling_comparison/full_series_20261026/target_layout_full_series.png),
[interactive viewer](../runs/sampling_comparison/full_series_20261026/targets_3d_interactive.html)
and [coverage curves](../runs/sampling_comparison/full_series_20261026/target_coverage_learning_curve.png)
use the unchanged box X/Y [-2,2]m, Z [0.3,0.8]m. Height expansion is labelled
and changes display only. Blue=train, orange=validation.

The [full-series report](SAMPLING_SERIES.md) records spatial design and findings.
Larger stratified grids are independently jittered, not nested target prefixes;
uniform targets retain deterministic prefixes, but split membership must be
checked separately. Geometry uses100,000 shared uniform probes, training/validation
coverage separately. XYZ target coverage is not full state/history coverage or
control stability. Actual target coordinates and split IDs are in targets.csv.
