# Step 10 (2026-11-05): cross-evaluation of the two lines on the same episodes

Descriptive only (no adoption rule; CONFIRM and data/test are not opened). Rodrigo's v2 policies
(`team/rodrigo` commit in `trained/rodrigo_v2/SOURCE_COMMIT.txt`, `weights/v2/*/model.npz`, copied to
`trained/rodrigo_v2/`) run under this task's protocol v1.1 harness through the adapter
`policies/rodrigo_policy.py` (written from scratch here; it re-implements his NumPy inference for the
28-feature variant `raw21_targetpolar_clock_v1`: MLP and GRU, clock counted from reset, outputs mapped
(right, left) -> [fr, fl, bl, br], wheels clipped to +-5.27 rad/s and torques to +-30 N m as his evaluator does).

Checks before the runs:
- offline (`offline_check.py`, clean pilot episodes, recorded observations -> adapter -> compared with the clean
  expert labels): K144 seed 0 wheels NRMSE 0.146 (his validation 0.13-0.14), torques 0.18-0.79 N m RMSE inside his
  height band and 2.9-3.4 N m on two low targets (z outside his 0.3-0.8 m band);
- closed-loop smoke (`smoke/rodrigo-k144_s0_pilot`, 12 episodes): 7/12 success, 7/7 held 10 s, 2 tipped, 3 timeouts,
  no policy / harness errors, 0.124 ms per step.

Chain: `run_cross.sh` (resumable) -> DEV and HARD runs `rodrigo-k144_s{0..4}`, `rodrigo-gru40_s{0..4}`,
`rodrigo-ctrl_s0` (DEV only), then `cross_analysis.py --set dev|hard` -> `cross_<set>.txt/.json`
(arms expert, C3, P, K144, GRU40, CTRL; subsets all / band z 0.3-0.8 / band & |x|,|y| <= 2 m; seed-mean rates
with the two-level bootstrap, paired differences to C3 with bootstrap and Welch; failure classes of this task).
