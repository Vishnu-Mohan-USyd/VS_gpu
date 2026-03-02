#!/usr/bin/env python
"""diag_std_sweep.py — Sweep E→E STD parameters to find optimal setting.

Tests how E→E short-term depression (Tsodyks-Markram) interacts with
the omission response and F>R learning. Hypothesis: U=0.5 is too aggressive,
causing OMR negativity and F>R reversal.

Conditions:
  A. ee_std_enabled=False (all other mechanisms ON)
  B. ee_std_U=0.15
  C. ee_std_U=0.25
  D. ee_std_U=0.35
  E. ee_std_U=0.50 (current default — baseline)

Reports: F>R trajectory, OMR conductance, post-cal OSI, calibration scale.
"""

import sys
import time
import math
import numpy as np

sys.path.insert(0, '.')

from biologically_plausible_v1_stdp import Params, RgcLgnV1Network, compute_osi
from network_jax import (
    numpy_net_to_jax_state,
    run_segment_jax, run_sequence_trial_jax, reset_state_jax,
    evaluate_tuning_jax, evaluate_omission_response,
    calibrate_ee_drive_jax,
)
import jax
import jax.numpy as jnp

GOLDEN_RATIO = (1 + math.sqrt(5)) / 2
THETA_STEP = 180.0 / GOLDEN_RATIO
ELEMENT_MS = 150.0
ITI_MS = 1500.0
SEQ_THETAS = [0.0, 45.0, 90.0, 135.0]
CONTRAST = 1.0
N_PRESENTATIONS = 800
SEED = 42
M = 16
N = 8


def compute_fwd_rev_ratio(W_e_e, pref, seq_thetas):
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


def run_condition(label, param_overrides, verbose=True):
    """Run one condition: Phase A + calibrate + Phase B + evaluate OMR."""
    print(f"\n{'='*70}")
    print(f"Condition: {label}")
    print(f"  Overrides: {param_overrides}")
    print(f"{'='*70}")
    t_start = time.perf_counter()

    # Build network with overrides
    base_params = dict(
        M=M, N=N, seed=SEED,
        ee_stdp_enabled=True,
        ee_connectivity="all_to_all",
        ee_stdp_A_plus=0.005,
        ee_stdp_A_minus=0.006,
        ee_stdp_weight_dep=True,
        train_segments=0,
        segment_ms=300.0,
    )
    base_params.update(param_overrides)
    p = Params(**base_params)
    net = RgcLgnV1Network(p)
    state, static = numpy_net_to_jax_state(net)

    # Phase A (100 segments)
    if verbose:
        print(f"  Phase A (100 segments)...")
    for seg in range(100):
        theta = (seg * THETA_STEP) % 180.0
        state, _ = run_segment_jax(state, static, theta, 1.0, True)

    # Evaluate tuning
    thetas_eval = np.linspace(0, 180, 12, endpoint=False)
    rates = evaluate_tuning_jax(state, static, thetas_eval, repeats=2)
    osi_vals, pref = compute_osi(rates, thetas_eval)
    post_a_osi = float(osi_vals.mean())
    if verbose:
        print(f"    Post-Phase A OSI: {post_a_osi:.3f}")

    # Calibrate E→E drive
    if verbose:
        print(f"  Calibrating E→E drive...")
    scale, frac = calibrate_ee_drive_jax(state, static)
    if verbose:
        print(f"    scale={scale:.1f}, frac={frac:.4f}")

    # Apply calibration
    eye_M = jnp.eye(M, dtype=jnp.float32)
    W_e_e_cal = state.W_e_e * scale * (1.0 - eye_M)
    state = state._replace(W_e_e=W_e_e_cal)

    mask_ee = np.array(static.mask_e_e).astype(bool)
    W_ee_np = np.array(W_e_e_cal)
    cal_mean = float(W_ee_np[mask_ee].mean()) if mask_ee.any() else float(W_ee_np.mean())
    new_w_max = max(cal_mean * 3.0, float(static.w_e_e_max))
    static = static._replace(w_e_e_max=new_w_max)

    # Post-calibration tuning
    rates2 = evaluate_tuning_jax(state, static, thetas_eval, repeats=2)
    osi2, pref2 = compute_osi(rates2, thetas_eval)
    post_cal_osi = float(osi2.mean())
    if verbose:
        print(f"    Post-cal OSI: {post_cal_osi:.3f}, w_e_e_max={new_w_max:.4f}")

    # Phase B training with checkpoints
    A_plus = float(static.ee_stdp_A_plus)
    A_minus = float(static.ee_stdp_A_minus)
    checkpoints = [0, 100, 200, 400, 600, 800]
    fr_results = []

    W_ee_np = np.array(state.W_e_e)
    _, _, ratio = compute_fwd_rev_ratio(W_ee_np, pref, SEQ_THETAS)
    fr_results.append((0, ratio))

    if verbose:
        print(f"  Phase B ({N_PRESENTATIONS} presentations)...")
    next_cp = 1
    for k in range(1, N_PRESENTATIONS + 1):
        state, _ = run_sequence_trial_jax(
            state, static, SEQ_THETAS, ELEMENT_MS, ITI_MS, CONTRAST,
            'ee', ee_A_plus_eff=A_plus, ee_A_minus_eff=A_minus)
        if next_cp < len(checkpoints) and k == checkpoints[next_cp]:
            W_ee_np = np.array(state.W_e_e)
            _, _, ratio = compute_fwd_rev_ratio(W_ee_np, pref, SEQ_THETAS)
            fr_results.append((k, ratio))
            if verbose:
                print(f"    [pres {k}] F>R={ratio:.4f}")
            next_cp += 1

    # Evaluate OMR
    if verbose:
        print(f"  Evaluating omission response...")
    omr = evaluate_omission_response(
        state, static, SEQ_THETAS, ELEMENT_MS, ITI_MS,
        contrast=CONTRAST, n_eval_trials=10, omit_index=1)

    elapsed = time.perf_counter() - t_start
    final_fr = fr_results[-1][1]
    omr_cond = omr['omr_conductance']
    fr_traj = [r for _, r in fr_results]

    # Check monotonicity (allow 0.05 tolerance for 2-step lookback)
    monotonic = True
    for i in range(2, len(fr_traj)):
        if fr_traj[i] < fr_traj[i-2] - 0.05:
            monotonic = False
            break

    result = {
        'label': label,
        'post_a_osi': post_a_osi,
        'post_cal_osi': post_cal_osi,
        'scale': scale,
        'w_e_e_max': new_w_max,
        'fr_trajectory': fr_traj,
        'final_fr': final_fr,
        'monotonic': monotonic,
        'omr_conductance': omr_cond,
        'omr_spikes': omr['omr_spikes'],
        'elapsed': elapsed,
    }

    print(f"\n  RESULT: F>R={final_fr:.4f}, OMR={omr_cond:.6f}, "
          f"post-cal OSI={post_cal_osi:.3f}, monotonic={monotonic}")
    print(f"  F>R trajectory: {' → '.join(f'{r:.3f}' for r in fr_traj)}")
    print(f"  Time: {elapsed:.1f}s")

    return result


def main():
    print("=" * 70)
    print("E→E STD Parameter Sweep — Diagnosing OMR/F>R Regression")
    print("=" * 70)

    conditions = [
        ("A: STD OFF", {"ee_std_enabled": False}),
        ("B: STD U=0.15", {"ee_std_U": 0.15}),
        ("C: STD U=0.25", {"ee_std_U": 0.25}),
        ("D: STD U=0.35", {"ee_std_U": 0.35}),
        ("E: STD U=0.50 (current)", {"ee_std_U": 0.50}),
    ]

    results = []
    for label, overrides in conditions:
        r = run_condition(label, overrides)
        results.append(r)

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Condition':<25} {'Post-A OSI':>10} {'Post-Cal OSI':>12} {'Scale':>7} "
          f"{'Final F>R':>10} {'Monotonic':>10} {'OMR':>12} {'Pass?':>6}")
    print("-" * 95)
    for r in results:
        fr_pass = r['final_fr'] > 1.15 and r['monotonic']
        omr_pass = r['omr_conductance'] > 0
        all_pass = fr_pass and omr_pass
        marker = "YES" if all_pass else "NO"
        print(f"{r['label']:<25} {r['post_a_osi']:>10.3f} {r['post_cal_osi']:>12.3f} "
              f"{r['scale']:>7.1f} {r['final_fr']:>10.4f} "
              f"{'YES' if r['monotonic'] else 'NO':>10} "
              f"{r['omr_conductance']:>12.6f} {marker:>6}")

    print("\n  F>R trajectories:")
    for r in results:
        traj = ' → '.join(f'{v:.3f}' for v in r['fr_trajectory'])
        print(f"    {r['label']:<25}: {traj}")

    print(f"\n  Total elapsed: {sum(r['elapsed'] for r in results):.0f}s")

    # Recommendation
    print("\n  RECOMMENDATION:")
    passing = [r for r in results if r['final_fr'] > 1.15 and r['monotonic'] and r['omr_conductance'] > 0]
    if passing:
        best = max(passing, key=lambda r: r['omr_conductance'])
        print(f"    Best passing condition: {best['label']}")
        print(f"    F>R={best['final_fr']:.4f}, OMR={best['omr_conductance']:.6f}, OSI={best['post_cal_osi']:.3f}")
    else:
        # Find best F>R among conditions with positive OMR
        pos_omr = [r for r in results if r['omr_conductance'] > 0]
        if pos_omr:
            best = max(pos_omr, key=lambda r: r['final_fr'])
            print(f"    No fully passing condition. Best with positive OMR: {best['label']}")
            print(f"    F>R={best['final_fr']:.4f}, OMR={best['omr_conductance']:.6f}")
        else:
            print(f"    ALL conditions have negative OMR — issue is not STD-specific")

    return 0


if __name__ == '__main__':
    sys.exit(main())
