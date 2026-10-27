# Command errors and autonomous divergence

Primary fidelity is abs(pred-expert) <= 0.05*abs(expert), excluding numerical
zeros (abs(expert)<=1e-6), which receive separate absolute-error statistics.
Report conditional physical/relative mean, median and P95 for failures; a mean
relative error can be dominated by tiny nonzero references. Commands inside
limits can still be wrong, and offline fidelity does not guarantee stable control.

Dense 1 ms diagnostics for existing uniform models show differences from the
first command on identical initial input, with early trajectory departures before
clipping. This rules out clipping as the initiating explanation for those measured
early deviations, not as a possible contributor later. State-shift response and
missing expert memory remain hypotheses; no unique cause of instability is isolated.

The active [sampling comparison](SAMPLING_COMPARISON.md) retains these metrics.
Older off-schedule tables are withdrawn; exact audit artifacts remain unmodified
outside maintained documentation. The corrected plan includes 256, not a relabelled
historical result.
