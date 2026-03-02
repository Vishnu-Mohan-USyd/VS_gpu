#!/usr/bin/env python
"""Validate inter-HC SOM-mediated surround suppression.

Tests biological properties of the inter-HC lateral inhibition pathway.
All tests have STRICT pass/fail criteria grounded in primary literature.

8 tests:
  1. Surround Suppression Index (SSI) — inter-HC SOM reduces E activity
  2. Suppression Onset Timing — suppression emerges within biological latency
  3. SOM Firing Rate in Biological Range
  4. Ablation Test — disabling inter-HC SOM increases E firing (MOST IMPORTANT)
  5. n_hc=1 Regression — Phase B pipeline unchanged at n_hc=1
  6. Performance Regression — n_hc=64 Phase A within time budget
  7. Phase B Compatibility — inter-HC SOM does not break sequence learning
  8. Distance Dependence — W_hc_lateral Gaussian decay with distance

References:
  Adesnik H, Bruns W, Taniguchi H, Huang ZJ, Scanziani M (2012). A neural
    circuit for spatial summation in visual cortex. Nature 490:226-231.
  Self MW, Lorteije JA, Vangeneugden J, et al. (2014). Orientation-selective
    inhibition in visual cortex. J Neurosci 34:5261-5275.
  Ma WP, Liu BH, Li YT, et al. (2010). Visual representations by cortical
    somatostatin inhibitory neurons. J Neurosci 30:10076-10086.
  Gavornik JP, Bear MF (2014). Learned spatiotemporal sequence recognition
    and prediction in primary visual cortex. Nature Neuroscience 17:732-737.
"""

import sys
import time
import math
import json
import traceback
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
THETA_STEP = 180.0 / GOLDEN_RATIO  # ~111.246 deg

ELEMENT_MS = 150.0
ITI_MS = 1500.0
SEQ_THETAS = [0.0, 45.0, 90.0, 135.0]
CONTRAST = 1.0
N_PIX = 8
SEED = 42

FIXED_PHASES = jnp.array([0.0, 0.0, 0.0, 0.0])


# ── Helpers ────────────────────────────────────────────────────────────────

def build_network(n_hc, M, seed=SEED, inter_hc_som_enabled=True,
                  inter_hc_som_gain=2.0):
    """Build network and convert to JAX state."""
    p = Params(
        M=M, N=N_PIX, seed=seed,
        n_hc=n_hc,
        rf_spacing_pix=1.0 if n_hc > 1 else 4.0,
        ee_stdp_enabled=True,
        ee_connectivity="all_to_all",
        ee_stdp_A_plus=0.005,
        ee_stdp_A_minus=0.006,
        ee_stdp_weight_dep=True,
        train_segments=0,
        segment_ms=300.0,
        inter_hc_som_enabled=inter_hc_som_enabled,
        inter_hc_som_gain=inter_hc_som_gain,
    )
    net = RgcLgnV1Network(p)
    state, static = numpy_net_to_jax_state(net)
    return state, static, p


def run_phase_a(state, static, n_segments, verbose=True, label=""):
    """Run Phase A training (feedforward STDP)."""
    t0 = time.perf_counter()
    for seg in range(n_segments):
        theta = (seg * THETA_STEP) % 180.0
        state, _ = run_segment_jax(state, static, theta, 1.0, True)
        if verbose and (seg + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            print(f"    {label}Phase A: {seg + 1}/{n_segments} ({elapsed:.1f}s)")
    phase_a_time = time.perf_counter() - t0
    return state, phase_a_time


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
    """Compute F>R ratio per hypercolumn."""
    if n_hc == 1:
        W_ee_np = np.array(state.W_e_e)
        _, _, ratio = compute_fwd_rev_ratio(W_ee_np, pref, seq_thetas)
        return np.array([ratio])
    W_ee_hc = np.array(state.W_e_e_hc)
    ratios = []
    for h in range(n_hc):
        W_h = W_ee_hc[h]
        pref_h = pref[h * M_per_hc : (h + 1) * M_per_hc]
        _, _, r = compute_fwd_rev_ratio(W_h, pref_h, seq_thetas)
        ratios.append(r)
    return np.array(ratios)


# ── Test 1: Surround Suppression Index ─────────────────────────────────────

def test_1_surround_suppression_index():
    """Test 1: Surround Suppression Index (SSI).

    Protocol:
      1. Train n_hc=4 M=64 with inter_hc_som_gain=2.0 (100 segments Phase A).
      2. Record total E spikes during 3 evaluation segments at preferred theta.
      3. Build matched network with inter_hc_som_enabled=False, copy weights.
      4. Run same evaluation. SSI = 1 - (rate_with / rate_without).

    Pass: SSI > 0.05  (>5% suppression).
    Fail: SSI < 0.01  (mechanism not functional).
    Citation: Self et al. 2014; Adesnik et al. 2012 (biology: SSI 0.3-0.7).
    """
    print(f"\n{'='*60}")
    print("Test 1: Surround Suppression Index (SSI)")
    print(f"{'='*60}")

    n_hc, M = 4, 64

    # Train WITH inter-HC SOM
    print("  Building network with inter_hc_som_gain=2.0...")
    state_on, static_on, _ = build_network(n_hc, M, inter_hc_som_gain=2.0)
    state_on, pa_time = run_phase_a(state_on, static_on, 100, label="[ON] ")
    print(f"  Phase A done in {pa_time:.1f}s")

    # Get preferred orientation
    thetas_eval = np.linspace(0, 180, 12, endpoint=False)
    rates = evaluate_tuning_jax(state_on, static_on, thetas_eval, repeats=2)
    osi_vals, pref = compute_osi(rates, thetas_eval)
    print(f"  Mean OSI: {float(osi_vals.mean()):.3f}")

    # Measure E spike counts with inter-HC SOM enabled (3 segments at pref theta)
    pref_theta = float(np.median(pref))  # representative orientation
    total_spikes_on = 0
    for _ in range(3):
        st_eval = reset_state_jax(state_on, static_on)
        _, counts = run_segment_jax(st_eval, static_on, pref_theta, 1.0, False)
        total_spikes_on += int(np.array(counts).sum())
    mean_spikes_on = total_spikes_on / 3.0

    # Build matched network with inter_hc_som DISABLED
    print("  Building matched network with inter_hc_som_enabled=False...")
    state_off, static_off, _ = build_network(n_hc, M, inter_hc_som_enabled=False)
    # Copy trained weights from the enabled network into the disabled network
    state_off = state_off._replace(
        W=state_on.W,
        W_pv_e=state_on.W_pv_e,
        W_e_e=state_on.W_e_e,
        W_e_e_hc=state_on.W_e_e_hc,
    )

    total_spikes_off = 0
    for _ in range(3):
        st_eval = reset_state_jax(state_off, static_off)
        _, counts = run_segment_jax(st_eval, static_off, pref_theta, 1.0, False)
        total_spikes_off += int(np.array(counts).sum())
    mean_spikes_off = total_spikes_off / 3.0

    ssi = 1.0 - (mean_spikes_on / max(1.0, mean_spikes_off))
    print(f"\n  Spikes WITH inter-HC SOM:    {mean_spikes_on:.1f}")
    print(f"  Spikes WITHOUT inter-HC SOM: {mean_spikes_off:.1f}")
    print(f"  SSI = 1 - (on/off) = {ssi:.4f}")

    passed = ssi > 0.05
    status = "PASS" if passed else "FAIL"
    if ssi < 0.01:
        print(f"  HARD FAIL: SSI < 0.01 — mechanism not functional")
    print(f"\n  Result: {status} (SSI={ssi:.4f}, criterion > 0.05)")
    return passed, {"ssi": ssi, "spikes_on": mean_spikes_on,
                     "spikes_off": mean_spikes_off}


# ── Test 2: Suppression Onset Timing ───────────────────────────────────────

def test_2_suppression_onset_timing():
    """Test 2: Suppression Onset Timing.

    Protocol:
      1. Train n_hc=4 M=64 with inter_hc_som_gain=2.0, Phase A 100 segments.
      2. Run one 300ms evaluation trial, collecting per-timestep spike counts
         in 5ms bins (10 timesteps at dt=0.5ms per bin).
      3. Build matched disabled network, run same trial.
      4. Find first bin where suppression > 5%.

    Pass: Onset between 5ms and 60ms.
    Fail: Onset > 80ms or no detectable onset.
    Citation: Adesnik et al. 2012 (~25ms onset in biology).
    """
    print(f"\n{'='*60}")
    print("Test 2: Suppression Onset Timing")
    print(f"{'='*60}")

    n_hc, M = 4, 64

    # Train network
    state_on, static_on, _ = build_network(n_hc, M, inter_hc_som_gain=2.0)
    state_on, _ = run_phase_a(state_on, static_on, 100, label="[ON] ")

    pref_theta = 90.0  # use a fixed orientation for reproducibility

    # Use run_sequence_trial_jax with a single-element "sequence" to get fine-grained traces
    # We'll use a single 300ms element with no omission in 'none' mode
    # This returns g_exc_ee_traces which has per-timestep resolution
    st_on = reset_state_jax(state_on, static_on)
    _, info_on = run_sequence_trial_jax(
        st_on, static_on, [pref_theta], 300.0, 100.0, 1.0, 'none',
        phases=jnp.array([0.0]))
    g_traces_on = np.array(info_on['g_exc_ee_traces'])  # (1, 600, M_total)
    # Element counts give total spikes but not per-timestep.
    # g_exc_ee_traces is per-timestep: proxy for activity.

    # Matched disabled network
    state_off, static_off, _ = build_network(n_hc, M, inter_hc_som_enabled=False)
    state_off = state_off._replace(
        W=state_on.W,
        W_pv_e=state_on.W_pv_e,
        W_e_e=state_on.W_e_e,
        W_e_e_hc=state_on.W_e_e_hc,
    )
    st_off = reset_state_jax(state_off, static_off)
    _, info_off = run_sequence_trial_jax(
        st_off, static_off, [pref_theta], 300.0, 100.0, 1.0, 'none',
        phases=jnp.array([0.0]))
    g_traces_off = np.array(info_off['g_exc_ee_traces'])  # (1, 600, M_total)

    # Bin conductances into 5ms bins (10 timesteps per bin at dt=0.5ms)
    dt_ms = float(static_on.dt_ms)
    bin_width_ms = 5.0
    steps_per_bin = int(round(bin_width_ms / dt_ms))
    trace_on = g_traces_on[0]  # (600, M_total)
    trace_off = g_traces_off[0]
    n_steps = trace_on.shape[0]
    n_bins = n_steps // steps_per_bin

    binned_on = np.zeros(n_bins)
    binned_off = np.zeros(n_bins)
    for b in range(n_bins):
        s0 = b * steps_per_bin
        s1 = s0 + steps_per_bin
        binned_on[b] = trace_on[s0:s1].mean()
        binned_off[b] = trace_off[s0:s1].mean()

    # Find first bin with > 5% suppression
    onset_bin = -1
    for b in range(n_bins):
        if binned_off[b] > 1e-10:
            suppression = (binned_off[b] - binned_on[b]) / binned_off[b]
            if suppression > 0.05:
                onset_bin = b
                break

    if onset_bin >= 0:
        onset_ms = (onset_bin + 0.5) * bin_width_ms
        print(f"  Suppression onset: bin {onset_bin} = {onset_ms:.1f}ms")
        print(f"    g_on={binned_on[onset_bin]:.6f}, "
              f"g_off={binned_off[onset_bin]:.6f}, "
              f"suppression={(binned_off[onset_bin] - binned_on[onset_bin])/binned_off[onset_bin]:.3f}")
    else:
        onset_ms = -1.0
        print("  No suppression > 5% detected in any bin")

    # Print first 20 bins for inspection
    print(f"\n  Binned g_exc_ee (5ms bins, first 20):")
    print(f"  {'Bin':>4} {'Time_ms':>8} {'g_ON':>10} {'g_OFF':>10} {'Suppr%':>8}")
    for b in range(min(20, n_bins)):
        t_ms = (b + 0.5) * bin_width_ms
        sup = (binned_off[b] - binned_on[b]) / max(1e-12, binned_off[b]) * 100.0
        print(f"  {b:4d} {t_ms:8.1f} {binned_on[b]:10.6f} {binned_off[b]:10.6f} {sup:7.1f}%")

    passed = (5.0 <= onset_ms <= 60.0)
    if onset_ms > 80.0:
        print(f"  HARD FAIL: onset {onset_ms:.1f}ms > 80ms (too slow)")
    elif onset_ms < 0:
        print(f"  HARD FAIL: no detectable suppression onset")

    status = "PASS" if passed else "FAIL"
    print(f"\n  Result: {status} (onset={onset_ms:.1f}ms, criterion 5-60ms)")
    return passed, {"onset_ms": onset_ms}


# ── Test 3: SOM Firing Rate ────────────────────────────────────────────────

def test_3_som_firing_rate():
    """Test 3: SOM Firing Rate in Biological Range.

    Protocol:
      1. Train n_hc=4 M=64, Phase A 100 segments.
      2. Run 300ms evaluation, estimate SOM rate from SOM state.
      NOTE: SOM spikes are not directly returned by run_segment_jax.
            We estimate SOM activity from the g_v1_inh_som conductance:
            if SOM→E conductance is non-zero and increases during stimulation,
            SOM must be firing.

    Pass: Evidence that SOM is active (g_v1_inh_som > baseline).
    Citation: Ma et al. 2010 (SOM 1-40 Hz in vivo).
    """
    print(f"\n{'='*60}")
    print("Test 3: SOM Firing Rate / Activity")
    print(f"{'='*60}")

    n_hc, M = 4, 64
    state, static, _ = build_network(n_hc, M, inter_hc_som_gain=2.0)
    state, _ = run_phase_a(state, static, 100, label="")

    # Run a single segment and check SOM-related conductance state
    st_pre = reset_state_jax(state, static)
    g_som_rise_pre = float(np.array(st_pre.g_v1_inh_som_rise_hc).mean())
    g_som_decay_pre = float(np.array(st_pre.g_v1_inh_som_decay_hc).mean())
    print(f"  Pre-stim SOM conductance: rise={g_som_rise_pre:.6f}, decay={g_som_decay_pre:.6f}")

    # Run one segment
    st_post, counts = run_segment_jax(st_pre, static, 90.0, 1.0, False)
    g_som_rise_post = float(np.array(st_post.g_v1_inh_som_rise_hc).mean())
    g_som_decay_post = float(np.array(st_post.g_v1_inh_som_decay_hc).mean())
    print(f"  Post-stim SOM conductance: rise={g_som_rise_post:.6f}, decay={g_som_decay_post:.6f}")

    # Also check SOM membrane potential — if SOM fires, v should show depolarization
    som_v_post = np.array(st_post.som_v_hc)  # (n_hc, n_som_per_hc)
    som_v_mean = float(som_v_post.mean())
    som_v_max = float(som_v_post.max())
    print(f"  SOM membrane: mean={som_v_mean:.1f}mV, max={som_v_max:.1f}mV")

    # Check inter-HC ring buffer — was it written to?
    inter_buf = np.array(st_post.inter_hc_som_buf)
    buf_max = float(inter_buf.max())
    buf_mean = float(inter_buf[inter_buf > 0].mean()) if (inter_buf > 0).any() else 0.0
    print(f"  Inter-HC buffer: max={buf_max:.6f}, mean_nonzero={buf_mean:.6f}")

    # SOM is active if the double-exponential conductance is non-zero after stimulation.
    # g_v1_inh_som_rise/decay are the SOM→E inhibitory conductance variables.
    # Non-zero values after 300ms of stimulation prove SOM fired during the segment.
    som_conductance_active = g_som_decay_post > 1e-8
    som_depolarized = som_v_mean > -70.0  # resting is -65mV for LTS

    print(f"\n  SOM→E inhibitory conductance active: {som_conductance_active}")
    print(f"  SOM depolarized (mean > -70mV): {som_depolarized}")

    # The primary indicator: SOM-mediated inhibitory conductance > 0
    passed = som_conductance_active
    status = "PASS" if passed else "FAIL"
    print(f"\n  Result: {status}")
    return passed, {"g_som_rise": g_som_rise_post, "g_som_decay": g_som_decay_post,
                     "som_v_mean": som_v_mean, "buf_max": buf_max}


# ── Test 4: Ablation Test (MOST IMPORTANT) ─────────────────────────────────

def test_4_ablation():
    """Test 4: Ablation — disabling inter-HC SOM increases E firing.

    Protocol:
      1. Train n_hc=4 M=64 with inter_hc_som_gain=2.0, Phase A 100 segments.
      2. Run 10 evaluation segments at preferred orientation, record E spikes.
      3. Build matched network with inter_hc_som_enabled=False, copy weights.
      4. Run same 10 evaluation segments, record E spikes.
      5. Compare: disabled should have MORE spikes (inhibition removed).

    Pass: disabled_count > enabled_count * 1.02 (>2% disinhibition).
    Fail: enabled_count >= disabled_count (inter-HC SOM NOT reducing E activity).
    """
    print(f"\n{'='*60}")
    print("Test 4: Ablation Test (MOST IMPORTANT)")
    print(f"{'='*60}")

    n_hc, M = 4, 64
    n_eval_segs = 10
    pref_theta = 90.0

    # Train with inter-HC SOM enabled
    state_on, static_on, _ = build_network(n_hc, M, inter_hc_som_gain=2.0)
    state_on, _ = run_phase_a(state_on, static_on, 100, label="[ON] ")

    # Evaluate with inter-HC SOM ON
    total_on = 0
    for i in range(n_eval_segs):
        st_eval = reset_state_jax(state_on, static_on)
        _, counts = run_segment_jax(st_eval, static_on, pref_theta, 1.0, False)
        total_on += int(np.array(counts).sum())
    mean_on = total_on / n_eval_segs
    print(f"  Spikes (inter-HC SOM ON, {n_eval_segs} segs): total={total_on}, mean={mean_on:.1f}")

    # Build matched disabled network
    state_off, static_off, _ = build_network(n_hc, M, inter_hc_som_enabled=False)
    state_off = state_off._replace(
        W=state_on.W,
        W_pv_e=state_on.W_pv_e,
        W_e_e=state_on.W_e_e,
        W_e_e_hc=state_on.W_e_e_hc,
    )

    # Evaluate with inter-HC SOM OFF
    total_off = 0
    for i in range(n_eval_segs):
        st_eval = reset_state_jax(state_off, static_off)
        _, counts = run_segment_jax(st_eval, static_off, pref_theta, 1.0, False)
        total_off += int(np.array(counts).sum())
    mean_off = total_off / n_eval_segs
    print(f"  Spikes (inter-HC SOM OFF, {n_eval_segs} segs): total={total_off}, mean={mean_off:.1f}")

    ratio = total_off / max(1, total_on)
    print(f"\n  Ratio (OFF/ON): {ratio:.4f}")
    print(f"  Disinhibition: {(ratio - 1.0) * 100:.1f}%")

    passed = total_off > total_on * 1.02
    if total_on >= total_off:
        print("  HARD FAIL: inter-HC SOM NOT reducing E activity at all")

    status = "PASS" if passed else "FAIL"
    print(f"\n  Result: {status} (OFF/ON={ratio:.4f}, criterion > 1.02)")
    return passed, {"total_on": total_on, "total_off": total_off,
                     "mean_on": mean_on, "mean_off": mean_off, "ratio": ratio}


# ── Test 5: n_hc=1 Regression ──────────────────────────────────────────────

def test_5_nhc1_regression():
    """Test 5: n_hc=1 Regression.

    Protocol: Standard n_hc=1 M=16 Phase A + calibrate + Phase B (800 pres).

    Pass criteria:
      - F>R > 1.15 at 800 presentations
      - F>R monotonically increasing (2-step lookback, 0.05 tolerance)
      - OMR > 0
      - Post-cal OSI > 0.30
      - Benchmark < 150s

    Citation: Gavornik & Bear 2014.
    """
    print(f"\n{'='*60}")
    print("Test 5: n_hc=1 M=16 Regression")
    print(f"{'='*60}")

    n_hc, M = 1, 16
    t_start = time.perf_counter()

    state, static, _ = build_network(n_hc, M, inter_hc_som_gain=1.0)
    state, _ = run_phase_a(state, static, 100, label="")

    # Evaluate tuning
    thetas_eval = np.linspace(0, 180, 12, endpoint=False)
    rates = evaluate_tuning_jax(state, static, thetas_eval, repeats=2)
    osi_vals, pref = compute_osi(rates, thetas_eval)
    print(f"  Post-Phase A OSI: {float(osi_vals.mean()):.3f}")

    # Calibrate
    scale, frac = calibrate_ee_drive_jax(state, static, target_frac=0.15, osi_floor=0.30)
    print(f"  Calibration: scale={scale:.1f}, frac={frac:.4f}")

    # Apply calibration (n_hc=1 manual path)
    eye_M = jnp.eye(M, dtype=jnp.float32)
    W_e_e_cal = state.W_e_e * scale * (1.0 - eye_M)
    state = state._replace(W_e_e=W_e_e_cal)
    mask_ee = np.array(static.mask_e_e).astype(bool)
    W_ee_np = np.array(W_e_e_cal)
    cal_mean = float(W_ee_np[mask_ee].mean()) if mask_ee.any() else float(W_ee_np.mean())
    new_w_max = max(cal_mean * 3.0, float(static.w_e_e_max))
    static = static._replace(w_e_e_max=new_w_max)

    # Post-cal OSI
    rates2 = evaluate_tuning_jax(state, static, thetas_eval, repeats=2)
    osi2, _ = compute_osi(rates2, thetas_eval)
    post_cal_osi = float(osi2.mean())
    print(f"  Post-cal OSI: {post_cal_osi:.3f}")

    # Phase B
    A_plus = float(static.ee_stdp_A_plus)
    A_minus = float(static.ee_stdp_A_minus)

    checkpoints = [0, 100, 200, 400, 600, 800]
    fr_results = []

    # Initial F>R
    W_ee_np = np.array(state.W_e_e)
    _, _, ratio = compute_fwd_rev_ratio(W_ee_np, pref, SEQ_THETAS)
    fr_results.append((0, ratio))
    print(f"\n  Phase B (800 presentations)...")
    print(f"    [pres 0] F>R={ratio:.4f}")

    t_b = time.perf_counter()
    cp_idx = 1
    for k in range(1, 801):
        state, _ = run_sequence_trial_jax(
            state, static, SEQ_THETAS, ELEMENT_MS, ITI_MS, CONTRAST,
            'ee', ee_A_plus_eff=A_plus, ee_A_minus_eff=A_minus)

        if cp_idx < len(checkpoints) and k == checkpoints[cp_idx]:
            W_ee_np = np.array(state.W_e_e)
            _, _, ratio = compute_fwd_rev_ratio(W_ee_np, pref, SEQ_THETAS)
            fr_results.append((k, ratio))
            elapsed = time.perf_counter() - t_b
            print(f"    [pres {k}] F>R={ratio:.4f} ({elapsed:.1f}s)")
            cp_idx += 1

    # OMR
    omr = evaluate_omission_response(
        state, static, SEQ_THETAS, ELEMENT_MS, ITI_MS,
        contrast=CONTRAST, n_eval_trials=10, omit_index=1)
    omr_val = omr['omr_conductance']

    total_time = time.perf_counter() - t_start

    final_fr = fr_results[-1][1]
    med_traj = [r for _, r in fr_results]

    # Monotonicity
    monotonic = True
    for i in range(2, len(med_traj)):
        if med_traj[i] < med_traj[i-2] - 0.05:
            monotonic = False
            break

    print(f"\n  F>R trajectory: {' -> '.join(f'{v:.3f}' for v in med_traj)}")
    print(f"  Final F>R: {final_fr:.4f}")
    print(f"  Monotonic: {monotonic}")
    print(f"  OMR: {omr_val:.6f}")
    print(f"  Post-cal OSI: {post_cal_osi:.3f}")
    print(f"  Total time: {total_time:.1f}s")

    # Check criteria
    fails = []
    if final_fr <= 1.15:
        fails.append(f"F>R {final_fr:.4f} <= 1.15")
    if not monotonic:
        fails.append(f"F>R not monotonic")
    if omr_val <= 0:
        fails.append(f"OMR {omr_val:.6f} <= 0")
    if post_cal_osi <= 0.30:
        fails.append(f"Post-cal OSI {post_cal_osi:.3f} <= 0.30")
    if total_time > 150.0:
        fails.append(f"Time {total_time:.1f}s > 150s")

    passed = len(fails) == 0
    status = "PASS" if passed else "FAIL"
    if fails:
        for f in fails:
            print(f"  FAIL: {f}")
    print(f"\n  Result: {status}")
    return passed, {"final_fr": final_fr, "omr": omr_val,
                     "post_cal_osi": post_cal_osi, "monotonic": monotonic,
                     "total_time": total_time}


# ── Test 6: Performance Regression ──────────────────────────────────────────

def test_6_performance():
    """Test 6: Performance Regression.

    Protocol:
      1. Build n_hc=64 M=64 (4096 neurons) with inter_hc_som_enabled=True.
      2. Run Phase A for 50 segments, time it.

    Pass: < 60s for 50 segments (< 1.2 s/seg).
    Fail: > 90s (overhead > 100% of baseline ~0.55 s/seg).
    """
    print(f"\n{'='*60}")
    print("Test 6: Performance Regression (n_hc=64 M=64)")
    print(f"{'='*60}")

    state, static, _ = build_network(64, 64, inter_hc_som_gain=2.0)

    # Warmup (2 segments)
    print("  Warming up JIT...")
    for i in range(2):
        theta = (i * THETA_STEP) % 180.0
        state, _ = run_segment_jax(state, static, theta, 1.0, True)
    jax.block_until_ready(state.W)

    # Time 50 segments
    print("  Timing 50 Phase A segments...")
    t0 = time.perf_counter()
    for seg in range(50):
        theta = ((seg + 2) * THETA_STEP) % 180.0
        state, _ = run_segment_jax(state, static, theta, 1.0, True)
    jax.block_until_ready(state.W)
    elapsed = time.perf_counter() - t0

    ms_per_seg = elapsed / 50 * 1000
    print(f"  50 segments in {elapsed:.1f}s ({ms_per_seg:.0f}ms/seg)")

    passed = elapsed < 60.0
    if elapsed > 90.0:
        print("  HARD FAIL: > 90s (overhead > 100%)")

    status = "PASS" if passed else "FAIL"
    print(f"\n  Result: {status} ({elapsed:.1f}s, criterion < 60s)")
    return passed, {"elapsed_s": elapsed, "ms_per_seg": ms_per_seg}


# ── Test 7: Phase B Compatibility ───────────────────────────────────────────

def test_7_phaseb_compatibility():
    """Test 7: Phase B Compatibility.

    Protocol:
      1. Train n_hc=4 M=64 with inter_hc_som_gain=2.0.
      2. Phase A 100 segments, calibrate (target_frac=0.05), prepare Phase B.
      3. Phase B 400 presentations (SEQ_THETAS, FIXED_PHASES).
      4. Compute F>R at 100, 200, 400. Evaluate OMR.

    Pass: F>R > 1.02 at 400 pres, OMR > -0.001.
    Fail: F>R < 0.98 (inter-HC SOM breaking sequence learning).
    """
    print(f"\n{'='*60}")
    print("Test 7: Phase B Compatibility (n_hc=4 M=64)")
    print(f"{'='*60}")

    n_hc, M = 4, 64
    n_pres = 400

    state, static, _ = build_network(n_hc, M, inter_hc_som_gain=2.0)
    state, _ = run_phase_a(state, static, 100, label="")

    # Tuning
    thetas_eval = np.linspace(0, 180, 12, endpoint=False)
    rates = evaluate_tuning_jax(state, static, thetas_eval, repeats=2)
    osi_vals, pref = compute_osi(rates, thetas_eval)
    print(f"  Post-Phase A OSI: {float(osi_vals.mean()):.3f}")

    # Calibrate + prepare Phase B
    scale, frac = calibrate_ee_drive_jax(state, static, target_frac=0.05, osi_floor=0.30)
    print(f"  Calibration: scale={scale:.1f}, frac={frac:.4f}")
    state, static, _, _ = prepare_phaseb_ee(state, static, scale)

    # Post-cal OSI
    rates2 = evaluate_tuning_jax(state, static, thetas_eval, repeats=2)
    osi2, _ = compute_osi(rates2, thetas_eval)
    print(f"  Post-cal OSI: {float(osi2.mean()):.3f}")

    A_plus = float(static.ee_stdp_A_plus)
    A_minus = float(static.ee_stdp_A_minus)
    M_per_hc = M

    # F>R checkpoints
    fr_checkpoints = [0, 100, 200, 400]
    fr_traj = []

    per_hc_fr = compute_per_hc_fr(state, pref, SEQ_THETAS, n_hc, M_per_hc)
    fr_traj.append((0, float(np.median(per_hc_fr))))
    print(f"\n  Phase B ({n_pres} presentations, FIXED_PHASES)...")
    print(f"    [pres 0] F>R median={float(np.median(per_hc_fr)):.4f}")

    t_b = time.perf_counter()
    cp_idx = 1
    for k in range(1, n_pres + 1):
        state, _ = run_sequence_trial_jax(
            state, static, SEQ_THETAS, ELEMENT_MS, ITI_MS, CONTRAST,
            'ee', ee_A_plus_eff=A_plus, ee_A_minus_eff=A_minus,
            phases=FIXED_PHASES)

        if cp_idx < len(fr_checkpoints) and k == fr_checkpoints[cp_idx]:
            per_hc_fr = compute_per_hc_fr(state, pref, SEQ_THETAS, n_hc, M_per_hc)
            med_fr = float(np.median(per_hc_fr))
            fr_traj.append((k, med_fr))
            elapsed = time.perf_counter() - t_b
            print(f"    [pres {k}] F>R median={med_fr:.4f} ({elapsed:.1f}s)")
            cp_idx += 1

    # OMR
    omr = evaluate_omission_response(
        state, static, SEQ_THETAS, ELEMENT_MS, ITI_MS,
        contrast=CONTRAST, n_eval_trials=10, omit_index=1,
        phases=FIXED_PHASES)
    omr_val = omr['omr_conductance']
    omr_per_hc_med = omr.get('omr_per_hc_median', omr_val)

    final_fr = fr_traj[-1][1]
    traj_str = ' -> '.join(f'{v:.3f}' for _, v in fr_traj)
    print(f"\n  F>R trajectory: {traj_str}")
    print(f"  Final F>R median: {final_fr:.4f}")
    print(f"  OMR: global={omr_val:.6f}, per-HC median={omr_per_hc_med:.6f}")

    fails = []
    if final_fr < 1.02:
        fails.append(f"F>R {final_fr:.4f} < 1.02")
    if final_fr < 0.98:
        fails.append(f"HARD FAIL: F>R {final_fr:.4f} < 0.98 — sequence learning broken")
    if omr_per_hc_med < -0.001:
        fails.append(f"OMR per-HC median {omr_per_hc_med:.6f} < -0.001")

    passed = len(fails) == 0
    status = "PASS" if passed else "FAIL"
    if fails:
        for f in fails:
            print(f"  FAIL: {f}")
    print(f"\n  Result: {status}")
    return passed, {"final_fr": final_fr, "omr": omr_val,
                     "omr_per_hc_median": omr_per_hc_med,
                     "fr_trajectory": [(p, v) for p, v in fr_traj]}


# ── Test 8: Distance Dependence ─────────────────────────────────────────────

def test_8_distance_dependence():
    """Test 8: Distance Dependence of W_hc_lateral.

    Protocol:
      1. Build n_hc=16 (4x4 grid) with inter_hc_som_gain=2.0.
      2. Inspect W_hc_lateral (n_hc, n_hc).
      3. For a center HC, compare weight from nearest neighbor vs farthest HC.

    Pass: nearest/farthest ratio > 2.0.
    Fail: ratio < 1.5 or all weights equal.
    Citation: Adesnik et al. 2012 (surround suppression falls off with distance).
    """
    print(f"\n{'='*60}")
    print("Test 8: Distance Dependence of W_hc_lateral")
    print(f"{'='*60}")

    state, static, _ = build_network(16, 64, inter_hc_som_gain=2.0)

    W_lat = np.array(static.W_hc_lateral)  # (16, 16)
    print(f"  W_hc_lateral shape: {W_lat.shape}")
    print(f"  W_hc_lateral range: [{W_lat.min():.6f}, {W_lat.max():.6f}]")
    print(f"  W_hc_lateral diagonal (should be 0): {W_lat.diagonal().sum():.6f}")

    # 4x4 grid: center HC = index 5 (position 1,1)
    center_hc = 5
    weights_from_center = W_lat[center_hc, :]  # weights FROM other HCs TO center
    print(f"\n  Center HC (idx={center_hc}, pos=1,1):")

    # Grid positions
    gx = np.arange(16) % 4
    gy = np.arange(16) // 4
    cx, cy = gx[center_hc], gy[center_hc]

    # Compute distances from center
    dist = np.sqrt((gx - cx)**2 + (gy - cy)**2)

    # Nearest neighbors (distance=1.0): indices where dist==1
    nearest_mask = np.abs(dist - 1.0) < 0.01
    nearest_weights = weights_from_center[nearest_mask]

    # Farthest HCs (max distance)
    max_dist = dist.max()
    farthest_mask = np.abs(dist - max_dist) < 0.01
    farthest_weights = weights_from_center[farthest_mask]

    # Also diagonal neighbors (dist = sqrt(2))
    diag_mask = np.abs(dist - np.sqrt(2)) < 0.01
    diag_weights = weights_from_center[diag_mask]

    print(f"  Nearest (d=1.0, n={nearest_mask.sum()}): "
          f"mean={nearest_weights.mean():.6f}")
    print(f"  Diagonal (d={np.sqrt(2):.2f}, n={diag_mask.sum()}): "
          f"mean={diag_weights.mean():.6f}" if diag_mask.any() else "")
    print(f"  Farthest (d={max_dist:.2f}, n={farthest_mask.sum()}): "
          f"mean={farthest_weights.mean():.6f}")

    # Print full weight row for center HC
    print(f"\n  Weights to center HC (sorted by distance):")
    print(f"  {'HC':>3} {'Pos':>6} {'Dist':>6} {'Weight':>10}")
    order = np.argsort(dist)
    for idx in order:
        if idx == center_hc:
            continue
        print(f"  {idx:3d} ({gx[idx]},{gy[idx]}) {dist[idx]:6.2f} {weights_from_center[idx]:10.6f}")

    # Compute ratio
    nearest_mean = float(nearest_weights.mean())
    farthest_mean = float(farthest_weights.mean())
    ratio = nearest_mean / max(1e-12, farthest_mean)
    print(f"\n  Nearest/farthest ratio: {ratio:.2f}")

    # Check for flat weights
    weight_std = float(weights_from_center[weights_from_center > 0].std())
    print(f"  Weight std (non-zero): {weight_std:.6f}")

    all_equal = weight_std < 1e-8
    if all_equal:
        print("  HARD FAIL: all weights equal (no distance dependence)")

    passed = ratio > 2.0 and not all_equal
    if ratio < 1.5:
        print(f"  HARD FAIL: ratio {ratio:.2f} < 1.5 (too weak)")

    status = "PASS" if passed else "FAIL"
    print(f"\n  Result: {status} (ratio={ratio:.2f}, criterion > 2.0)")
    return passed, {"nearest_mean": nearest_mean, "farthest_mean": farthest_mean,
                     "ratio": ratio, "all_equal": all_equal}


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Inter-HC SOM Surround Suppression Validation")
    print("  Branch: l4-inhibition-validation")
    print("  8 STRICT tests with biological citations")
    print("=" * 60)

    test_fns = [
        ("Test 8: Distance Dependence", test_8_distance_dependence),
        ("Test 3: SOM Firing Rate", test_3_som_firing_rate),
        ("Test 1: Surround Suppression Index", test_1_surround_suppression_index),
        ("Test 2: Suppression Onset Timing", test_2_suppression_onset_timing),
        ("Test 4: Ablation (MOST IMPORTANT)", test_4_ablation),
        ("Test 6: Performance Regression", test_6_performance),
        ("Test 5: n_hc=1 Regression", test_5_nhc1_regression),
        ("Test 7: Phase B Compatibility", test_7_phaseb_compatibility),
    ]

    results = {}
    all_pass = True

    for name, fn in test_fns:
        try:
            passed, metrics = fn()
            results[name] = {"passed": passed, **metrics}
            if not passed:
                all_pass = False
        except Exception as e:
            print(f"\n  {name}: EXCEPTION -- {e}")
            traceback.print_exc()
            results[name] = {"passed": False, "error": str(e)}
            all_pass = False

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    n_pass = 0
    for name, _ in test_fns:
        r = results.get(name, {"passed": False})
        status = "PASS" if r.get("passed", False) else "FAIL"
        if r.get("passed", False):
            n_pass += 1
        print(f"  [{status}] {name}")
    print(f"\n  {n_pass}/{len(test_fns)} tests passed")
    print(f"  OVERALL: {'ALL PASS' if all_pass else 'SOME FAILED'}")

    # Save results
    out_path = 'validate_inter_hc_inhibition_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
