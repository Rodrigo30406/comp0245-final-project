"""Compile every closed-loop evaluation and training run into one markdown table.

Usage: python3 scripts/analysis/final_table.py   (from anywhere; paths resolve from the repository root)
"""
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVALS = {  # evaluation dir -> {policy label in summary: (display name, training run dir)}
    "stratified1024_baseline_validation_20261027": {"baseline": ("MLP 21 inputs (baseline), clean data", "mlp_stratified1024_torch_baseline")},
    "stratified1024_egocentric_validation_20261027": {"egocentric": ("MLP 25 egocentric+clock, clean data", "mlp_stratified1024_egocentric_clock")},
    "stratified1024_egocentric_seeds_validation_20261027": {"egocentric_seed1": ("MLP 25 egocentric+clock, clean, seed 1", "mlp_stratified1024_egocentric_clock_seed1"),
                                                             "egocentric_seed2": ("MLP 25 egocentric+clock, clean, seed 2", "mlp_stratified1024_egocentric_clock_seed2")},
    "stratified1024_window200ms_validation_20261027": {"window200ms": ("MLP 201 inputs, 200 ms window, clean data", "mlp_stratified1024_window200ms")},
    "stratified1024_gru_egocentric_validation_20261027": {"gru_ego_mem20ms": ("GRU 25 egocentric, memory 20 ms, clean", "gru_stratified1024_ego_stride20"),
                                                           "gru_ego_mem1ms": ("GRU 25 egocentric, memory 1 ms, clean", "gru_stratified1024_ego_stride1")},
    "stratified1024_gru_raw21_validation_20261027": {"gru_raw21_mem20ms": ("GRU 21 inputs, memory 20 ms, clean", "gru_stratified1024_raw21_stride20"),
                                                      "gru_raw21_mem1ms": ("GRU 21 inputs, memory 1 ms, clean", "gru_stratified1024_raw21_stride1")},
    "dart_w1t2_baseline_on_clean_validation_20261027": {"dart_baseline": ("MLP 21 inputs, DART medium noise", "mlp_dart_w1t2_torch_baseline")},
    "dart_w1t2_egocentric_on_clean_validation_20261027": {"dart_egocentric": ("MLP 25 egocentric+clock, DART medium noise", "mlp_dart_w1t2_egocentric_clock")},
    "dart_low_egocentric_on_clean_validation_20261027": {"dartlow_egocentric": ("MLP 25 egocentric+clock, DART low noise", "mlp_dart_low_egocentric_clock")},
    "dart_low_baseline_on_clean_validation_20261027": {"dartlow_baseline": ("MLP 21 inputs, DART low noise", "mlp_dart_low_torch_baseline")},
    "dart_low_egocentric_seeds_on_clean_validation_20261027": {f"dartlow_egocentric_seed{s}": (f"MLP 25 egocentric+clock, DART low noise, seed {s}", f"mlp_dart_low_egocentric_clock_seed{s}") for s in (1, 2, 3, 4)},
    "dart_low_gru_variants_on_clean_validation_20261027": {"gru_dartlow_ego_mem20ms": ("GRU 25 egocentric, memory 20 ms, DART low noise", "gru_dart_low_ego_stride20"),
                                                            "gru_dartlow_raw21_mem1ms": ("GRU 21 inputs, memory 1 ms, DART low noise", "gru_dart_low_raw21_stride1"),
                                                            "gru_dartlow_raw21_mem20ms": ("GRU 21 inputs, memory 20 ms, DART low noise", "gru_dart_low_raw21_stride20")},
    "dart_low_gru_ego_stride20_seeds_on_clean_validation_20261027": {f"gru_dartlow_ego_mem20ms_seed{s}": (f"GRU 25 egocentric, memory 20 ms, DART low noise, seed {s}", f"gru_dart_low_ego_stride20_seed{s}") for s in (1, 2, 3, 4)},
    "dart_low_mlp_raw21plus_seeds_on_clean_validation_20261027": {f"mlp_raw21plus_seed{s}": (f"MLP 32 inputs (21 explicit + 10 geometry + clock), DART low, seed {s}", f"mlp_dart_low_raw21plus_seed{s}") for s in range(5)},
    "dart_low_mlp_raw21clock_seeds_on_clean_validation_20261027": {f"mlp_raw21clock_seed{s}": (f"MLP 22 inputs (21 explicit + clock), DART low, seed {s}", f"mlp_dart_low_raw21clock_seed{s}") for s in range(5)},
    "dart_low_gru_raw21plus_seeds_on_clean_validation_20261027": {f"gru_raw21plus_mem20ms_seed{s}": (f"GRU 32 inputs, memory 20 ms, DART low, seed {s}", f"gru_dart_low_raw21plus_stride20_seed{s}") for s in range(5)},
    "dart_low_mlp_raw21target_seeds_on_clean_validation_20261027": {f"mlp_raw21target_seed{s}": (f"MLP 25 inputs (21 explicit + 3 target-in-base + clock), DART low, seed {s}", f"mlp_dart_low_raw21target_seed{s}") for s in range(5)},
    "dart_low_mlp_raw21polar_seeds_on_clean_validation_20261028": {f"mlp_raw21polar_seed{s}": (f"MLP 28 inputs (21 explicit + 3 target-in-base + 3 polar + clock), DART low, seed {s}", f"mlp_dart_low_raw21polar_seed{s}") for s in range(5)},
    "e1_mlp_raw21clock_seeds_on_clean_validation_20261027": {f"mlp_e1_raw21clock_seed{s}": (f"E1 random-pose data: MLP 22 (21 + clock), DART low, seed {s}", f"mlp_e1_raw21clock_seed{s}") for s in range(5)},
    "e1_mlp_raw21target_seeds_on_clean_validation_20261027": {f"mlp_e1_raw21target_seed{s}": (f"E1 random-pose data: MLP 25 (21 + 3 target-in-base + clock), DART low, seed {s}", f"mlp_e1_raw21target_seed{s}") for s in range(5)},
    "e1_mlp_raw21plus_seeds_on_clean_validation_20261027": {f"mlp_e1_raw21plus_seed{s}": (f"E1 random-pose data: MLP 32 (21 + 10 geometry + clock), DART low, seed {s}", f"mlp_e1_raw21plus_seed{s}") for s in range(5)},
    "e3_randomonly4096_on_clean_validation_20261027": {"randomonly4096_seed0": ("E3 random-only 4096: MLP 32, DART low, seed 0", "mlp_e3_randomonly4096_raw21plus_seed0")},
    "e3_randomonly4096_on_clean_validation_20261027_b": {f"randomonly4096_seed{s}": (f"E3 random-only 4096: MLP 32, DART low, seed {s}", f"mlp_e3_randomonly4096_raw21plus_seed{s}") for s in (1, 2)},
    "e3_mixture4096_on_clean_validation_20261027": {f"mixture4096_seed{s}": (f"E3 nominal + random 4096: MLP 32, DART low, seed {s}", f"mlp_e3_mixture4096_raw21plus_seed{s}") for s in (0, 1)},
    "e3_mixture4096_on_clean_validation_20261027_b": {"mixture4096_seed2": ("E3 nominal + random 4096: MLP 32, DART low, seed 2", "mlp_e3_mixture4096_raw21plus_seed2")},
    "e4_mlp_polarvel_on_clean_validation_20261029": {f"mlp_polarvel_seed{s}": (f"E4 nominal + random 4096: MLP 31 (28 pure + FD base velocity), seed {s}", f"mlp_e4_mixture4096_raw21polarvel_seed{s}") for s in range(3)},
    "e4_gru_polar_on_clean_validation_20261029": {f"gru_polar_mem20ms_seed{s}": (f"E4 nominal + random 4096: GRU 28 pure, memory 20 ms, seed {s}", f"gru_e4_mixture4096_raw21polar_stride20_seed{s}") for s in range(3)},
    "e4_mlp_polar_on_clean_validation_20261029": {f"mlp_polar_seed{s}": (f"E4 nominal + random 4096: MLP 28 pure (control), seed {s}", f"mlp_e4_mixture4096_raw21polar_seed{s}") for s in range(3)},
    "e5_tf_polar_on_clean_validation_20261030": {f"tf_polar_mem20s_seed{s}": (f"E5 nominal + random 4096: transformer 28 pure, attention 20 s, seed {s}", f"tf_e5_mixture4096_raw21polar_stride20_seed{s}") for s in range(3)},
    "e7a_mlp_stopw8_on_clean_validation_20261101": {f"mlp_stopw8_seed{s}": (f"E7a nominal + random 4096: MLP 28 pure, stop rows weighted x8, seed {s}", f"mlp_e7a_mixture4096_polar_stopw8_seed{s}") for s in range(3)},
    "e7b_mlp_l1_on_clean_validation_20261101": {f"mlp_l1_seed{s}": (f"E7b nominal + random 4096: MLP 28 pure, L1 wheel loss, seed {s}", f"mlp_e7b_mixture4096_polar_l1_seed{s}") for s in range(3)},
    "e7c_mlp_cat41_on_clean_validation_20261101": {f"mlp_cat41_seed{s}": (f"E7c nominal + random 4096: MLP 28 pure, categorical wheels (41 bins), seed {s}", f"mlp_e7c_mixture4096_polar_cat41_seed{s}") for s in range(3)},
    "e7d_mlp_prevwheel_on_clean_validation_20261101": {f"mlp_prevwheel_seed{s}": (f"E7d nominal + random 4096: MLP 30 (28 pure + previous wheel command), seed {s}", f"mlp_e7d_mixture4096_prevwheel_seed{s}") for s in range(3)},
    "e2_mlp_polar_on_clean_validation_20261101": {f"mlp_polar_e2_seed{s}": (f"E2 nominal + random + near-target 6144: MLP 28 pure, seed {s}", f"mlp_e2_mixture6144_polar_seed{s}") for s in range(3)},
    "e2_mlp_cat41_on_clean_validation_20261101": {f"mlp_cat41_e2_seed{s}": (f"E2 nominal + random + near-target 6144: MLP 28 pure, categorical wheels, seed {s}", f"mlp_e2_mixture6144_polar_cat41_seed{s}") for s in range(3)},
    "e8d_db045_mlp28_on_clean_validation_20261102": {f"mlp28_db045_seed{s}": (f"E8d: E4 MLP 28 control + wheel dead band 0.45 rad/s at evaluation, seed {s}", f"mlp_e4_mixture4096_raw21polar_seed{s}") for s in range(3)},
    "e8d_db045_gru28_on_clean_validation_20261102": {f"gru28_db045_seed{s}": (f"E8d: E4 GRU 28 pure, memory 20 ms + wheel dead band 0.45 rad/s at evaluation, seed {s}", f"gru_e4_mixture4096_raw21polar_stride20_seed{s}") for s in range(3)},
    "e8d_db045_tf28_on_clean_validation_20261102": {f"tf28_db045_seed{s}": (f"E8d: E5 transformer 28 pure + wheel dead band 0.45 rad/s at evaluation, seed {s}", f"tf_e5_mixture4096_raw21polar_stride20_seed{s}") for s in range(3)},
    "e8d_db045_mlp32_on_clean_validation_20261102": {f"mlp32_db045_seed{s}": (f"E8d: E3 MLP 32 mixture + wheel dead band 0.45 rad/s at evaluation, seed {s}", f"mlp_e3_mixture4096_raw21plus_seed{s}") for s in range(3)},
    "e8a_gated_on_clean_validation_20261102": {f"gated_seed_seed{s}": (f"E8a: gated stop (GRU 32 stop score, cummax gate, stateless head on 28 pure + gate), mixture 4096, seed {s}", f"gru_e8a_gated_mixture4096_polar_seed{s}") for s in range(3)},
    "e8b_mlpgru_on_clean_validation_20261102": {f"mlpgru_seed_seed{s}": (f"E8b: MLP 64 encoder -> GRU 128, memory 20 ms, 28 pure, mixture 4096, seed {s}", f"gru_e8b_mlpgru_mixture4096_polar_seed{s}") for s in range(3)},
    "e9_ksweep_mlp28_on_clean_validation_20261102": {f"trig{k}_seed0": (f"E9: MLP 28 pure + {k} trigger copies per episode (first zero-wheel row, arm-qd jitter), mixture 4096, seed 0", f"mlp_e9_mixture4096_polar_trig{k}_seed0") for k in (16, 144, 512)},
    "e9_trig144_mlp28_on_clean_validation_20261102": {f"trig144_seed{s}": (f"E9: MLP 28 pure + 144 trigger copies per episode, mixture 4096, seed {s}", f"mlp_e9_mixture4096_polar_trig144_seed{s}") for s in range(3)},
    "mlp28_trig144_v1succ_on_clean_validation_20261103": {f"mlp28_trig144_v1succ_seed{s}": (f"Ablation: MLP 28 pure + 144 trigger copies, mixture v1 (origin goal) with the expert failures dropped, seed {s}", f"mlp28_trig144_v1succ_seed{s}") for s in range(3)},
    "v2_mlp28_control_on_clean_validation_20261102": {f"mlp28_control_seed{s}": (f"Mixture v2 (start-referenced goal, expert successes only): MLP 28 pure control, seed {s}", f"v2_mlp28_control_seed{s}") for s in range(3)},
    "v2_mlp28_trig144_on_clean_validation_20261102": {f"mlp28_trig144_seed{s}": (f"Mixture v2: MLP 28 pure + 144 trigger copies per episode, seed {s}", f"v2_mlp28_trig144_seed{s}") for s in range(3)},
    "v2_mlp28_trigw144_on_clean_validation_20261102": {f"mlp28_trigw144_seed{s}": (f"Mixture v2: MLP 28 pure, trigger row loss weight 144, seed {s}", f"v2_mlp28_trigw144_seed{s}") for s in range(3)},
    "v2_mlp28_chunk10_on_clean_validation_20261102": {f"mlp28_chunk10_seed{s}": (f"Mixture v2: MLP 28 pure, action chunk 10 x 20 ms (first step executed), seed {s}", f"v2_mlp28_chunk10_seed{s}") for s in range(3)},
    "v2_gru28_trigw144_on_clean_validation_20261102": {f"gru28_trigw144_seed{s}": (f"Mixture v2: GRU 28 (stride 20 ms), trigger step loss weight 144, seed {s}", f"v2_gru28_trigw144_seed{s}") for s in range(3)},
    "v2_gated_trigw144_on_clean_validation_20261102": {f"gated_trigw144_seed{s}": (f"Mixture v2: E8a gated stop (hidden 32, stride 20 ms), trigger step loss weight 144, seed {s}", f"v2_gated_trigw144_seed{s}") for s in range(3)},
    "v2_gru28_h256_on_clean_validation_20261103": {f"gru28_h256_seed{s}": (f"Mixture v2: GRU 28 (stride 20 ms) hidden 256 / head 256, seed {s}", f"v2_gru28_h256_seed{s}") for s in range(3)},
    "v2_gru28_trigw144_jit_on_clean_validation_20261103": {f"gru28_trigw144_jit_seed{s}": (f"Mixture v2: GRU 28 (stride 20 ms) trigger step loss weight 144 + arm-velocity jitter, seed {s}", f"v2_gru28_trigw144_jit_seed{s}") for s in range(3)},
    "v2_gru28_patience20_on_clean_validation_20261103": {f"gru28_patience20_seed{s}": (f"Mixture v2: GRU 28 (stride 20 ms) patience 20, seed {s}", f"v2_gru28_patience20_seed{s}") for s in range(3)},
    "v2_gru28_stride10_on_clean_validation_20261103": {f"gru28_stride10_seed{s}": (f"Mixture v2: GRU 28 (stride 20 ms) memory every 10 ms, seed {s}", f"v2_gru28_stride10_seed{s}") for s in range(3)},
    "v2_gru28_lr3e-4_on_clean_validation_20261103": {f"gru28_lr3em4_seed{s}": (f"Mixture v2: GRU 28 (stride 20 ms) Adam 3e-4, seed {s}", f"v2_gru28_lr3e-4_seed{s}") for s in range(3)},
    "v2_gru28_epb4_on_clean_validation_20261103": {f"gru28_epb4_seed{s}": (f"Mixture v2: GRU 28 (stride 20 ms) 4 episodes per batch, seed {s}", f"v2_gru28_epb4_seed{s}") for s in range(3)},
    "v2_mlp28_trig144_db045_on_clean_validation_20261103": {f"mlp28_trig144_seed{s}": (f"Mixture v2: MLP 28 pure + 144 trigger copies, seed {s}, evaluated with the 0.45 rad/s wheel dead band", f"v2_mlp28_trig144_seed{s}") for s in range(3)},
    "v2_mlp28_cat41_trig144_on_clean_validation_20261103": {f"mlp28_cat41_trig144_seed{s}": (f"Mixture v2: MLP 28 pure, categorical wheels (41 bins) + 144 trigger copies, seed {s}", f"v2_mlp28_cat41_trig144_seed{s}") for s in range(3)},
    "v2_gru28_control_on_clean_validation_20261102": {f"gru28_control_seed{s}": (f"Mixture v2: GRU 28 (stride 20 ms) control, seed {s}", f"v2_gru28_control_seed{s}") for s in range(3)},
    "e9_trig144_db045_on_clean_validation_20261102": {f"trig144_seed{s}": (f"E9 K 144 seed {s} evaluated with the 0.45 rad/s wheel dead band (E8d option)", f"mlp_e9_mixture4096_polar_trig144_seed{s}") for s in range(3)},
    "e2_gru_polar_on_clean_validation_20261101": {f"gru_polar_e2_seed{s}": (f"E2 nominal + random + near-target 6144: GRU 28 pure, memory 20 ms, seed {s}", f"gru_e2_mixture6144_polar_stride20_seed{s}") for s in range(3)},
    "dart_low_gru_egocentric_on_clean_validation_20261027": {"gru_dartlow_ego_mem1ms": ("GRU 25 egocentric, memory 1 ms, DART low noise", "gru_dart_low_ego_stride1")},
}


def row(eval_dir, label, name, run):
    d = ROOT / "runs/evaluation" / eval_dir
    if not (d / "summary.json").exists() or label not in json.load(open(d / "summary.json")):
        return f"| {name} | pending | | | | | | | | |"
    s = json.load(open(d / "summary.json"))[label]
    fin, mn, fw, held = [], [], 0, 0
    for f in glob.glob(str(d / "episodes" / f"{label}_*.json")):
        m = json.load(open(f))["metrics"]
        if m and m.get("final_distance_m") is not None:
            fin.append(m["final_distance_m"]); mn.append(m["minimum_distance_m"])
            fw += m["by_tolerance"]["0.05"]["final_within_tolerance"]; held += m["by_tolerance"]["0.05"]["longest_sampled_dwell_s"] >= 10
    info = json.load(open(ROOT / "runs/training" / run / "model_info.json"))
    off = info["best_validation_episode_macro_balanced_nrmse"]
    clip = s["mean_final_clipping_fraction_per_command"]
    t = s["success_by_tolerance"]
    return (f"| {name} | {info['regression_parameters']:,} | {off:.4f} | {t['0.01']} / {t['0.02']} / {t['0.05']} | "
            f"{fw} | {held} | {np.median(fin):.2f} | {int((np.array(mn) < 0.05).sum())} | {clip[0]:.2f} / {np.mean(clip[4:]):.2f} | "
            f"{info['inference_single_observation_us']['median']:.0f} |")


lines = ["| Policy | Params | Offline val NRMSE | Stable arrival 1 / 2 / 5 cm (of 205) | Final within 5 cm | Held ≥ 10 s | Median final EE dist. (m) | Reached < 5 cm | Clip wheels / torque | Latency µs |",
         "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
         "| expert (lecturer controller) | — | — | 205 / 205 / 205 | 205 | 205 | 0.00 | 205 | 0.00 / 0.00 | — |"]
for eval_dir, policies in EVALS.items():
    for label, (name, run) in policies.items():
        lines.append(row(eval_dir, label, name, run))
print("\n".join(lines))
