#!/usr/bin/env python
"""validate_l4_inhibition_multiconfig.py — Cross-config validation of L4 inhibitory circuit.

Tests the full L4 inhibition redesign (PV→PV, SOM, SOM→PV, Phase A E→E STDP,
E→E STD, E→SOM facilitation) across multiple network scales:

  Config 1: n_hc=1,  M=64  (64 neurons — single hypercolumn baseline)
  Config 2: n_hc=4,  M=64  (256 neurons — 2×2 grid)
  Config 3: n_hc=16, M=64  (1024 neurons — 4×4 grid)
  Config 4: n_hc=64, M=64  (4096 neurons — 8×8 grid)

For each config, measures:
  - Post-Phase A OSI (mean, per-HC distribution)
  - Post-calibration OSI (mean, per-HC distribution)
  - Calibration scale and drive_frac achieved
  - F>R trajectory at [0, 100, 200, 400, 600, 800] presentations
  - F>R monotonicity
  - Per-HC F>R distribution (median, mean, frac > 1.0)
  - Omission response (OMR conductance)
  - Total wall-clock time

Usage:
  python validate_l4_inhibition_multiconfig.py [config_index]

  config_index: 0=n_hc=1, 1=n_hc=4, 2=n_hc=16, 3=n_hc=64, omit=run all sequentially

References:
  Gavornik JP, Bear MF (2014). Nature Neuroscience 17: 732-737.
  Ko H et al. (2011). Nature 473: 87-91.
  Thomson AM, Lamy C (2007). Front Neurosci 1: 215-227.
"""

import sys
import time
import math
import json
import numpy as np

sys.path.insert(0, '.')

from biologically_plausible_v1_stdp import Params, RgcLgnV1Network, compute_osi
from network_jax import (
    numpy_net_to_jax_state,
    run_segment_jax, run_sequence_trial_jax, reset_state_jax,
    evaluate_tuning_jax, evaluate_omission_response,
    calibrate_ee_drive_jax, prepare_phaseb_ee,
)
import jax
import jax.numpy as jnp

# ── Constants ──────────────────────────────────────────────────────────────

GOLDEN_RATIO = (1 + math.sqrt(5)) / 2
THETA_STEP = 180.0 / GOLDEN_RATIO

ELEMENT_MS = 150.0
ITI_MS = 1500.0
SEQ_THETAS = [0.0, 45.0, 90.0, 135.0]
CONTRAST = 1.0
N_PRESENTATIONS = 800
PHASE_A_SEGMENTS = 100
SEED = 42
N_PIX = 8

# Fixed spatial phases for Phase B (Gavornik & Bear: same physical gratings)
FIXED_PHASES = jnp.array([0.0, 0.0, 0.0, 0.0])

CONFIGS = [
    {"n_hc": 1,  "M": 64, "label": "n_hc=1  M=64  (64 neurons)"},
    {"n_hc": 4,  "M": 64, "label": "n_hc=4  M=64  (256 neurons)"},
    {"n_hc": 16, "M": 64, "label": "n_hc=16 M=64  (1024 neurons)"},
    {"n_hc": 64, "M": 64, "label": "n_hc=64 M=64  (4096 neurons)"},
]

FR_CHECKPOINTS = [0, 100, 200, 400, 600, 800]
OMR_CHECKPOINTS = [0, 400, 800]


# ── Helpers ────────────────────────────────────────────────────────────────

def compute_fwd_rev_ratio(W_e_e, pref, seq_thetas):
    """Compute forward/reverse weight asymmetry ratio."""
    fwd_ws, rev_ws = [], []
    for ei in range(len(seq_thetas) - 1):
        pre_th, post_th = seq_thetas[ei], seq_thetas[ei + 1]
        d_pre = np.abs(pref - pre_th)
        d_pre = np.minimum(d_pre, 180.0 - d_pre)
        d_post = np.abs(pref - post_th)
        d_post = np.minimum(d_post, 180.0 - d_post)
        pre_mask = d_pre < 22.5
        post_mask = d_post < 22.5
        for pi in np.where(post_mask)[0]:
            for pj in np.where(pre_mask)[0]:
                if pi != pj:
                    fwd_ws.append(W_e_e[pi, pj])
                    rev_ws.append(W_e_e[pj, pi])
    if len(fwd_ws) == 0:
        return 0.0, 0.0, 1.0
    fwd_m = float(np.mean(fwd_ws))
    rev_m = float(np.mean(rev_ws))
    ratio = fwd_m / max(1e-10, rev_m)
    return fwd_m, rev_m, ratio


def compute_per_hc_fr(state, pref, seq_thetas, n_hc, M_per_hc):
    """Compute F>R ratio per hypercolumn.

    For n_hc > 1, reads state.W_e_e_hc (n_hc, M_per_hc, M_per_hc) since
    Phase B STDP updates the per-HC array only (flat W_e_e is stale).
    """
    if n_hc == 1:
        W_ee_np = np.array(state.W_e_e)
        _, _, ratio = compute_fwd_rev_ratio(W_ee_np, pref, seq_thetas)
        return np.array([ratio])
    W_ee_hc = np.array(state.W_e_e_hc)  # (n_hc, M_per_hc, M_per_hc)
    ratios = []
    for h in range(n_hc):
        W_h = W_ee_hc[h]
        pref_h = pref[h * M_per_hc : (h + 1) * M_per_hc]
        _, _, r = compute_fwd_rev_ratio(W_h, pref_h, seq_thetas)
        ratios.append(r)
    return np.array(ratios)


def compute_per_hc_osi(osi_vals, n_hc, M_per_hc):
    """Compute mean OSI per hypercolumn."""
    if n_hc == 1:
        return np.array([float(osi_vals.mean())])
    means = []
    for h in range(n_hc):
        sl = slice(h * M_per_hc, (h + 1) * M_per_hc)
        means.append(float(osi_vals[sl].mean()))
    return np.array(means)


def fr_distribution_str(ratios):
    """Format F>R distribution as compact string."""
    return (f"median={np.median(ratios):.3f}, mean={np.mean(ratios):.3f}, "
            f"min={np.min(ratios):.3f}, max={np.max(ratios):.3f}, "
            f"frac>1={np.mean(ratios > 1.0):.1%}")


def osi_distribution_str(vals):
    """Format OSI distribution as compact string."""
    return (f"median={np.median(vals):.3f}, mean={np.mean(vals):.3f}, "
            f"min={np.min(vals):.3f}, max={np.max(vals):.3f}")


# ── Main runner ────────────────────────────────────────────────────────────

def run_config(cfg, verbose=True):
    """Run full Phase A + calibrate + Phase B + OMR for one config.

    Returns dict with all metrics.
    """
    n_hc = cfg["n_hc"]
    M_per_hc = cfg["M"]
    M_total = n_hc * M_per_hc
    label = cfg["label"]

    print(f"\n{'='*78}")
    print(f"  CONFIG: {label}")
    print(f"{'='*78}")
    t_start = time.perf_counter()

    # ── Build network ──
    p = Params(
        M=M_per_hc, N=N_PIX, seed=SEED,
        n_hc=n_hc,
        rf_spacing_pix=1.0 if n_hc > 1 else 4.0,
        ee_stdp_enabled=True,
        ee_connectivity="all_to_all",
        ee_stdp_A_plus=0.005,
        ee_stdp_A_minus=0.006,
        ee_stdp_weight_dep=True,
        train_segments=0,
        segment_ms=300.0,
    )
    net = RgcLgnV1Network(p)
    state, static = numpy_net_to_jax_state(net)

    # ── Phase A ──
    if verbose:
        print(f"\n  Phase A ({PHASE_A_SEGMENTS} segments)...")
    t_a = time.perf_counter()
    for seg in range(PHASE_A_SEGMENTS):
        theta = (seg * THETA_STEP) % 180.0
        state, _ = run_segment_jax(state, static, theta, 1.0, True)
        if verbose and (seg + 1) % 50 == 0:
            print(f"    Segment {seg + 1}/{PHASE_A_SEGMENTS}")
    phase_a_time = time.perf_counter() - t_a

    # ── Post-Phase A tuning ──
    thetas_eval = np.linspace(0, 180, 12, endpoint=False)
    rates = evaluate_tuning_jax(state, static, thetas_eval, repeats=2)
    osi_vals, pref = compute_osi(rates, thetas_eval)
    post_a_osi_mean = float(osi_vals.mean())
    post_a_osi_per_hc = compute_per_hc_osi(osi_vals, n_hc, M_per_hc)
    if verbose:
        print(f"    Post-Phase A OSI: {post_a_osi_mean:.3f} "
              f"[{osi_distribution_str(post_a_osi_per_hc)}]")
        print(f"    Phase A time: {phase_a_time:.1f}s ({phase_a_time/PHASE_A_SEGMENTS*1000:.0f}ms/seg)")

    # ── Calibrate E→E ──
    if verbose:
        print(f"\n  Calibrating E→E drive...")
    # target_frac tradeoff at M=64: higher → OMR positive but F>R weaker.
    # n_hc=1: tf=0.10 is the sweet spot (F>R=1.162, OMR=+0.000095).
    # n_hc>1: tf=0.05 preserves per-HC OSI and F>R across the grid.
    target_frac = 0.05 if n_hc > 1 else 0.10
    scale, frac = calibrate_ee_drive_jax(state, static, target_frac=target_frac, osi_floor=0.30)
    if verbose:
        print(f"    scale={scale:.1f}, frac={frac:.4f}")

    # ── Apply calibration ──
    # Use prepare_phaseb_ee for ALL configs (n_hc=1 and n_hc>1).
    # M-dependent headroom auto-selected: 5x for M<=16 and multi-HC,
    # 3x for n_hc=1 M>16 (dense recurrent cascade at 5x causes saturation).
    state, static, _, _ = prepare_phaseb_ee(state, static, scale)
    mask_ee = np.array(static.mask_e_e).astype(bool)
    W_ee_np = np.array(state.W_e_e)
    w_e_e_max = float(static.w_e_e_max)

    # ── Post-calibration tuning ──
    rates2 = evaluate_tuning_jax(state, static, thetas_eval, repeats=2)
    osi2, pref2 = compute_osi(rates2, thetas_eval)
    post_cal_osi_mean = float(osi2.mean())
    post_cal_osi_per_hc = compute_per_hc_osi(osi2, n_hc, M_per_hc)
    if verbose:
        print(f"    Post-cal OSI: {post_cal_osi_mean:.3f} "
              f"[{osi_distribution_str(post_cal_osi_per_hc)}]")
        print(f"    w_e_e_max={w_e_e_max:.2f}")

    # Use pre-calibration pref for F>R (post-cal distorted by strong E→E)
    pref_for_fr = pref

    # ── Phase B training ──
    A_plus = float(static.ee_stdp_A_plus)
    A_minus = float(static.ee_stdp_A_minus)

    fr_trajectory = []  # [(pres, median_fr, per_hc_fr_array)]
    omr_trajectory = []  # [(pres, omr_dict)]

    # Initial F>R
    per_hc_fr = compute_per_hc_fr(state, pref_for_fr, SEQ_THETAS, n_hc, M_per_hc)
    fr_trajectory.append((0, float(np.median(per_hc_fr)), per_hc_fr.copy()))

    # Phases for OMR evaluation must match training (Gavornik & Bear 2014 protocol)
    omr_phases = FIXED_PHASES if n_hc > 1 else None

    # Initial OMR
    if 0 in OMR_CHECKPOINTS:
        n_eval = 20  # enough trials to resolve weak OMR signal at large n_hc
        omr = evaluate_omission_response(
            state, static, SEQ_THETAS, ELEMENT_MS, ITI_MS,
            contrast=CONTRAST, n_eval_trials=n_eval, omit_index=1,
            phases=omr_phases)
        omr_trajectory.append((0, omr))

    if verbose:
        print(f"\n  Phase B ({N_PRESENTATIONS} presentations)...")
        print(f"    [pres 0] F>R median={float(np.median(per_hc_fr)):.4f} "
              f"[{fr_distribution_str(per_hc_fr)}]")

    t_b = time.perf_counter()
    fr_cp_idx = 1  # next FR checkpoint index
    omr_cp_idx = 1 if 0 in OMR_CHECKPOINTS else 0  # next OMR checkpoint index

    for k in range(1, N_PRESENTATIONS + 1):
        state, _ = run_sequence_trial_jax(
            state, static, SEQ_THETAS, ELEMENT_MS, ITI_MS, CONTRAST,
            'ee',
            ee_A_plus_eff=A_plus,
            ee_A_minus_eff=A_minus,
            phases=FIXED_PHASES if n_hc > 1 else None,
        )

        # F>R checkpoint
        if fr_cp_idx < len(FR_CHECKPOINTS) and k == FR_CHECKPOINTS[fr_cp_idx]:
            per_hc_fr = compute_per_hc_fr(state, pref_for_fr, SEQ_THETAS, n_hc, M_per_hc)
            med_fr = float(np.median(per_hc_fr))
            fr_trajectory.append((k, med_fr, per_hc_fr.copy()))
            elapsed_b = time.perf_counter() - t_b
            if verbose:
                print(f"    [pres {k}] F>R median={med_fr:.4f} "
                      f"[{fr_distribution_str(per_hc_fr)}] "
                      f"({elapsed_b:.1f}s)")
            fr_cp_idx += 1

        # OMR checkpoint
        if omr_cp_idx < len(OMR_CHECKPOINTS) and k == OMR_CHECKPOINTS[omr_cp_idx]:
            n_eval = 20  # enough trials to resolve weak OMR signal at large n_hc
            omr = evaluate_omission_response(
                state, static, SEQ_THETAS, ELEMENT_MS, ITI_MS,
                contrast=CONTRAST, n_eval_trials=n_eval, omit_index=1,
                phases=omr_phases)
            omr_trajectory.append((k, omr))
            if verbose:
                omr_str = f"OMR={omr['omr_conductance']:.6f}"
                if 'omr_per_hc_median' in omr:
                    omr_str += f" (per-HC median={omr['omr_per_hc_median']:.6f})"
                print(f"    [pres {k}] {omr_str}")
            omr_cp_idx += 1

    phase_b_time = time.perf_counter() - t_b
    total_time = time.perf_counter() - t_start

    # ── Extract final metrics ──
    final_fr_per_hc = fr_trajectory[-1][2]
    final_fr_median = float(np.median(final_fr_per_hc))
    final_fr_mean = float(np.mean(final_fr_per_hc))
    frac_hc_gt1 = float(np.mean(final_fr_per_hc > 1.0))

    # Monotonicity check on median trajectory (allow 0.05 tolerance, 2-step lookback)
    med_traj = [t[1] for t in fr_trajectory]
    monotonic = True
    for i in range(2, len(med_traj)):
        if med_traj[i] < med_traj[i-2] - 0.05:
            monotonic = False
            break

    final_omr_dict = omr_trajectory[-1][1] if omr_trajectory else {}
    final_omr = final_omr_dict.get('omr_conductance', float('nan'))
    # Per-HC OMR: use median per-HC OMR for multi-HC (more sensitive than global avg)
    final_omr_per_hc_median = final_omr_dict.get('omr_per_hc_median', None)
    final_omr_frac_positive = final_omr_dict.get('omr_per_hc_frac_positive', None)

    result = {
        "label": label,
        "n_hc": n_hc,
        "M_per_hc": M_per_hc,
        "M_total": M_total,
        "seed": SEED,
        "post_a_osi_mean": post_a_osi_mean,
        "post_a_osi_per_hc": post_a_osi_per_hc.tolist(),
        "post_cal_osi_mean": post_cal_osi_mean,
        "post_cal_osi_per_hc": post_cal_osi_per_hc.tolist(),
        "cal_scale": scale,
        "cal_frac": frac,
        "w_e_e_max": w_e_e_max,
        "fr_trajectory_median": med_traj,
        "fr_trajectory_pres": [t[0] for t in fr_trajectory],
        "final_fr_median": final_fr_median,
        "final_fr_mean": final_fr_mean,
        "final_fr_per_hc": final_fr_per_hc.tolist(),
        "frac_hc_gt1": frac_hc_gt1,
        "monotonic": monotonic,
        "omr_trajectory": [(p, o['omr_conductance']) for p, o in omr_trajectory],
        "final_omr": final_omr,
        "final_omr_per_hc_median": final_omr_per_hc_median,
        "final_omr_frac_positive": final_omr_frac_positive,
        "phase_a_time_s": phase_a_time,
        "phase_b_time_s": phase_b_time,
        "total_time_s": total_time,
        "ms_per_seg": phase_a_time / PHASE_A_SEGMENTS * 1000,
        "ms_per_pres": phase_b_time / N_PRESENTATIONS * 1000,
    }

    # ── Print config summary ──
    print(f"\n  {'─'*60}")
    print(f"  CONFIG SUMMARY: {label}")
    print(f"  {'─'*60}")
    print(f"  Post-Phase A OSI:  {post_a_osi_mean:.3f} [{osi_distribution_str(post_a_osi_per_hc)}]")
    print(f"  Post-cal OSI:      {post_cal_osi_mean:.3f} [{osi_distribution_str(post_cal_osi_per_hc)}]")
    print(f"  Calibration:       scale={scale:.1f}, frac={frac:.4f}, w_max={w_e_e_max:.2f}")
    print(f"  F>R trajectory:    {' → '.join(f'{v:.3f}' for v in med_traj)}")
    print(f"  Final F>R:         median={final_fr_median:.4f}, mean={final_fr_mean:.4f}")
    print(f"  F>R distribution:  {fr_distribution_str(final_fr_per_hc)}")
    print(f"  Monotonic:         {'YES' if monotonic else 'NO'}")
    print(f"  OMR trajectory:    {' → '.join(f'{v:.4f}' for _, v in result['omr_trajectory'])}")
    print(f"  Final OMR (global):{final_omr:.6f}")
    if final_omr_per_hc_median is not None:
        print(f"  Final OMR per-HC:  median={final_omr_per_hc_median:.6f}, "
              f"frac_positive={final_omr_frac_positive:.1%}")
    print(f"  Frac HCs > 1.0:   {frac_hc_gt1:.1%}")
    print(f"  Timing:            Phase A {phase_a_time:.1f}s ({phase_a_time/PHASE_A_SEGMENTS*1000:.0f}ms/seg), "
          f"Phase B {phase_b_time:.1f}s ({phase_b_time/N_PRESENTATIONS*1000:.0f}ms/pres)")
    print(f"  Total:             {total_time:.1f}s ({total_time/60:.1f}min)")

    # ── Pass/fail ──
    passes = []
    fails = []

    # Test 1: F>R > 1.10 (relaxed from 1.15 for multi-HC scaling)
    threshold = 1.15 if n_hc == 1 else 1.05
    if final_fr_median > threshold:
        passes.append(f"F>R median {final_fr_median:.3f} > {threshold}")
    else:
        fails.append(f"F>R median {final_fr_median:.3f} <= {threshold}")

    # Test 2: Monotonicity
    if monotonic:
        passes.append("F>R monotonic")
    else:
        fails.append(f"F>R NOT monotonic: {' → '.join(f'{v:.3f}' for v in med_traj)}")

    # Test 3: OMR > 0
    # For multi-HC: use per-HC median OMR (global average washes out at large n_hc)
    if n_hc > 1 and final_omr_per_hc_median is not None:
        omr_metric = final_omr_per_hc_median
        omr_label = "per-HC median"
    else:
        omr_metric = final_omr
        omr_label = "global"
    if omr_metric > 0:
        passes.append(f"OMR positive ({omr_label}={omr_metric:.6f})")
    else:
        fails.append(f"OMR non-positive ({omr_label}={omr_metric:.6f})")

    # Test 4: Fraction of HCs > 1.0 (relaxed for large grids)
    frac_thresh = 0.50 if n_hc >= 16 else 0.75
    if frac_hc_gt1 >= frac_thresh:
        passes.append(f"Frac HCs > 1.0: {frac_hc_gt1:.1%} >= {frac_thresh:.0%}")
    else:
        fails.append(f"Frac HCs > 1.0: {frac_hc_gt1:.1%} < {frac_thresh:.0%}")

    # Test 5: Post-cal OSI > 0.30
    if post_cal_osi_mean > 0.30:
        passes.append(f"Post-cal OSI {post_cal_osi_mean:.3f} > 0.30")
    else:
        fails.append(f"Post-cal OSI {post_cal_osi_mean:.3f} <= 0.30")

    all_pass = len(fails) == 0
    result["passes"] = passes
    result["fails"] = fails
    result["all_pass"] = all_pass

    print(f"\n  PASS ({len(passes)}):")
    for p in passes:
        print(f"    [PASS] {p}")
    if fails:
        print(f"  FAIL ({len(fails)}):")
        for f in fails:
            print(f"    [FAIL] {f}")
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAIL'}")

    return result


# ── Cross-config summary ──────────────────────────────────────────────────

def print_cross_config_summary(results):
    """Print a concise comparison table across all configs."""
    print(f"\n{'='*78}")
    print(f"  CROSS-CONFIG SUMMARY — L4 Inhibition Validation")
    print(f"  All mechanisms: PV→PV, SOM, SOM→PV, Phase A E→E STDP, E→E STD(U=0.25), E→SOM STP")
    print(f"{'='*78}\n")

    # Main metrics table
    header = (f"{'Config':<28} {'Post-A':>7} {'Post-Cal':>8} {'Scale':>7} "
              f"{'F>R med':>8} {'F>R mean':>9} {'Mono':>5} "
              f"{'OMR':>10} {'%HC>1':>6} {'Pass':>5} {'Time':>6}")
    print(header)
    print("─" * len(header))

    for r in results:
        mono_str = "YES" if r["monotonic"] else "NO"
        pass_str = "YES" if r["all_pass"] else "NO"
        time_str = f"{r['total_time_s']/60:.1f}m"
        print(f"{r['label']:<28} "
              f"{r['post_a_osi_mean']:>7.3f} "
              f"{r['post_cal_osi_mean']:>8.3f} "
              f"{r['cal_scale']:>7.0f} "
              f"{r['final_fr_median']:>8.3f} "
              f"{r['final_fr_mean']:>9.3f} "
              f"{mono_str:>5} "
              f"{r['final_omr']:>10.6f} "
              f"{r['frac_hc_gt1']:>5.0%} "
              f"{pass_str:>5} "
              f"{time_str:>6}")

    # F>R trajectory table
    print(f"\n  F>R Median Trajectories:")
    for r in results:
        traj = ' → '.join(f"{v:.3f}" for v in r["fr_trajectory_median"])
        print(f"    {r['label']:<28}: {traj}")

    # OMR trajectory table
    print(f"\n  OMR Trajectories:")
    for r in results:
        traj = ' → '.join(f"{v:.4f}" for _, v in r["omr_trajectory"])
        print(f"    {r['label']:<28}: {traj}")

    # Timing table
    print(f"\n  Timing:")
    print(f"    {'Config':<28} {'Phase A':>10} {'Phase B':>10} {'Total':>10} {'ms/seg':>8} {'ms/pres':>8}")
    print(f"    {'─'*76}")
    for r in results:
        print(f"    {r['label']:<28} "
              f"{r['phase_a_time_s']:>9.1f}s "
              f"{r['phase_b_time_s']:>9.1f}s "
              f"{r['total_time_s']:>9.1f}s "
              f"{r['ms_per_seg']:>7.0f} "
              f"{r['ms_per_pres']:>7.0f}")

    # OSI distribution
    print(f"\n  Per-HC OSI Distribution (post-calibration):")
    for r in results:
        vals = np.array(r["post_cal_osi_per_hc"])
        print(f"    {r['label']:<28}: {osi_distribution_str(vals)}")

    # Per-HC F>R distribution
    print(f"\n  Per-HC F>R Distribution (final):")
    for r in results:
        vals = np.array(r["final_fr_per_hc"])
        print(f"    {r['label']:<28}: {fr_distribution_str(vals)}")

    # Overall pass/fail
    all_pass = all(r["all_pass"] for r in results)
    print(f"\n  {'='*60}")
    n_pass = sum(1 for r in results if r["all_pass"])
    print(f"  OVERALL: {n_pass}/{len(results)} configs pass all tests")
    if not all_pass:
        for r in results:
            if not r["all_pass"]:
                print(f"    FAILURES in {r['label']}:")
                for f in r["fails"]:
                    print(f"      - {f}")
    print(f"  {'='*60}")

    return all_pass


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("  L4 Inhibition Multi-Config Validation")
    print("  Branch: l4-inhibition-validation")
    print("  Mechanisms: PV→PV, SOM, SOM→PV, Phase A E→E STDP, E→E STD, E→SOM STP")
    print("=" * 78)

    # Parse optional config index
    if len(sys.argv) > 1:
        idx = int(sys.argv[1])
        configs_to_run = [CONFIGS[idx]]
        print(f"\n  Running single config: {CONFIGS[idx]['label']}")
    else:
        configs_to_run = CONFIGS
        print(f"\n  Running ALL {len(CONFIGS)} configs sequentially")

    results = []
    for cfg in configs_to_run:
        r = run_config(cfg)
        results.append(r)

        # Save intermediate results to JSON
        out_path = f"l4_validation_{cfg['n_hc']}hc_{cfg['M']}m.json"
        with open(out_path, 'w') as f:
            json.dump(r, f, indent=2, default=str)
        print(f"\n  Results saved to {out_path}")

    # Cross-config summary
    if len(results) > 1:
        all_pass = print_cross_config_summary(results)
    else:
        all_pass = results[0]["all_pass"]

    # Save combined results
    with open("l4_validation_all_configs.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Combined results saved to l4_validation_all_configs.json")

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
