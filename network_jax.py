"""network_jax.py — JAX port of the hot-path (grating stimulus) from biologically_plausible_v1_stdp.py.

All functions are pure (no mutation), suitable for ``jax.jit`` and ``jax.lax.scan``.
Spikes are float32 (0.0/1.0) throughout.

Provides:
    * ``SimState`` / ``StaticConfig`` NamedTuples (JAX pytree-compatible)
    * ``numpy_net_to_jax_state`` / ``jax_state_to_numpy_net`` — conversion helpers
    * ``run_segment_jax`` — JIT-compiled segment runner (drop-in replacement)
"""

from __future__ import annotations

import functools
import math
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# State containers (NamedTuples are JAX pytree-compatible)
# ---------------------------------------------------------------------------

class IzhState(NamedTuple):
    v: jnp.ndarray  # membrane potential
    u: jnp.ndarray  # recovery variable


class STDPTraces(NamedTuple):
    x_pre: jnp.ndarray       # (M, n_lgn)
    x_pre_slow: jnp.ndarray  # (M, n_lgn)
    x_post: jnp.ndarray      # (M,)
    x_post_slow: jnp.ndarray # (M,)


class SimState(NamedTuple):
    lgn_v: jnp.ndarray           # (n_lgn,)
    lgn_u: jnp.ndarray           # (n_lgn,)
    v1_v: jnp.ndarray            # (M,)
    v1_u: jnp.ndarray            # (M,)
    pv_v: jnp.ndarray            # (n_pv,)
    pv_u: jnp.ndarray            # (n_pv,)
    som_v: jnp.ndarray           # (n_som,)
    som_u: jnp.ndarray           # (n_som,)
    I_lgn: jnp.ndarray           # (n_lgn,)
    g_exc_ff: jnp.ndarray        # (M,)
    g_exc_ee: jnp.ndarray        # (M,)
    g_v1_inh_pv_rise: jnp.ndarray  # (M,)
    g_v1_inh_pv_decay: jnp.ndarray # (M,)
    g_v1_inh_som: jnp.ndarray    # (M,)
    g_v1_apical: jnp.ndarray     # (M,)
    I_pv: jnp.ndarray            # (n_pv,)
    I_pv_inh: jnp.ndarray        # (n_pv,)
    I_som: jnp.ndarray           # (n_som,)
    I_som_inh: jnp.ndarray       # (n_som,)
    I_v1_bias: jnp.ndarray       # (M,)
    delay_buf: jnp.ndarray       # (L, n_lgn)
    ptr: jnp.ndarray             # int32 scalar
    delay_buf_ee: jnp.ndarray    # (L_ee, M)
    ptr_ee: jnp.ndarray          # int32 scalar
    stdp_x_pre: jnp.ndarray      # (M, n_lgn)
    stdp_x_pre_slow: jnp.ndarray # (M, n_lgn)
    stdp_x_post: jnp.ndarray     # (M,)
    stdp_x_post_slow: jnp.ndarray # (M,)
    pv_istdp_x_post: jnp.ndarray # (M,)
    W: jnp.ndarray               # (M, n_lgn)
    W_pv_e: jnp.ndarray          # (M, n_pv)
    W_e_e: jnp.ndarray           # (M, M)
    ee_pre_trace: jnp.ndarray     # (M, M) per-synapse pre-trace for delay-aware E→E STDP
    ee_post_trace: jnp.ndarray   # (M,) per-neuron post-trace for E→E STDP
    prev_v1_spk: jnp.ndarray     # (M,)
    rng_key: jnp.ndarray         # PRNGKey
    rate_avg: jnp.ndarray        # (M,)
    lgn_rgc_drive: jnp.ndarray   # (n_lgn,)
    drive_acc_ff: jnp.ndarray    # (M,) float32 (promoted to float64 outside JIT if needed)
    drive_acc_ee: jnp.ndarray    # (M,) float32
    drive_acc_steps: jnp.ndarray # int32 scalar


class StaticConfig(NamedTuple):
    # Connection matrices (immutable during a segment)
    W_rgc_lgn: jnp.ndarray       # (n_lgn, n_lgn)
    W_e_pv: jnp.ndarray          # (n_pv, M)
    W_lgn_pv: jnp.ndarray        # (n_pv, n_lgn)
    W_e_som: jnp.ndarray         # (n_som, M)
    W_som_e: jnp.ndarray         # (M, n_som)
    tc_mask_e_f32: jnp.ndarray   # (M, n_lgn)
    tc_mask_pv_f32: jnp.ndarray  # (n_pv, n_lgn)
    lgn_mask_e: jnp.ndarray      # (M, n_lgn)
    mask_pv_e: jnp.ndarray       # (M, n_pv) float32
    mask_e_e: jnp.ndarray        # (M, M) float32
    D: jnp.ndarray               # (M, n_lgn) int32
    D_pv: jnp.ndarray            # (n_pv, n_lgn) int32
    D_ee: jnp.ndarray            # (M, M) int32
    lgn_ids: jnp.ndarray         # (1, n_lgn) int32
    X_on: jnp.ndarray            # (N, N)
    Y_on: jnp.ndarray            # (N, N)
    X_off: jnp.ndarray           # (N, N)
    Y_off: jnp.ndarray           # (N, N)
    on_to_off: jnp.ndarray       # (N*N,) int32
    off_to_on: jnp.ndarray       # (N*N,) int32
    split_target_on: jnp.ndarray # (M,)
    split_target_off: jnp.ndarray # (M,)
    eye_M: jnp.ndarray           # (M, M)
    arange_M: jnp.ndarray        # (M,) int32
    arange_lgn: jnp.ndarray      # (n_lgn,) int32
    # Scalar params
    N: int
    M: int
    n_lgn: int
    n_pv: int
    n_som: int
    n_pix: int
    L: int
    L_ee: int
    dt_ms: float
    segment_ms: float
    steps: int
    spatial_freq: float
    temporal_freq: float
    base_rate: float
    gain_rate: float
    dog_grating_gain: float
    w_rgc_lgn_scalar: float
    decay_ampa: float
    decay_gaba: float
    decay_gaba_rise_pv: float
    decay_apical: float
    w_exc_gain: float
    E_exc: float
    E_inh: float
    # Izhikevich params — LGN (TC)
    lgn_a: float
    lgn_b: float
    lgn_c: float
    lgn_d: float
    lgn_v_peak: float
    # Izhikevich params — V1 E (RS)
    v1_a: float
    v1_b: float
    v1_c: float
    v1_d: float
    v1_v_peak: float
    # Izhikevich params — PV (FS)
    pv_a: float
    pv_b: float
    pv_c: float
    pv_d: float
    pv_v_peak: float
    # Izhikevich params — SOM (LTS)
    som_a: float
    som_b: float
    som_c: float
    som_d: float
    som_v_peak: float
    # STDP params
    decay_pre: float
    decay_pre_slow: float
    decay_post: float
    decay_post_slow: float
    A2_plus: float
    A3_plus: float
    A2_minus: float
    w_max: float
    A_het: float
    A_split: float
    w_decay: float
    # PV iSTDP params
    pv_istdp_decay: float
    pv_istdp_eta: float
    pv_istdp_rho: float
    w_pv_e_max: float
    # Homeostasis
    target_rate_hz: float
    homeostasis_rate: float
    homeostasis_clip: float
    homeostasis_decay: float
    # Split constraint
    split_constraint_rate: float
    split_constraint_clip: float
    # Bias homeostasis
    v1_bias_eta: float
    v1_bias_clip: float
    # Plasticity flags
    pv_inhib_plastic: bool
    ff_plastic_enabled: bool
    lgn_rgc_alpha: float
    w_lgn_pv_gain: float
    # E→E delay-aware STDP
    ee_stdp_enabled: bool
    ee_stdp_decay_pre: float
    ee_stdp_decay_post: float
    ee_stdp_A_plus: float
    ee_stdp_A_minus: float
    ee_stdp_weight_dep: bool
    w_e_e_min: float
    w_e_e_max: float


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def numpy_net_to_jax_state(net) -> Tuple[SimState, StaticConfig]:
    """Extract JAX (SimState, StaticConfig) from a numpy RgcLgnV1Network."""
    p = net.p

    state = SimState(
        lgn_v=jnp.array(net.lgn.v, dtype=jnp.float32),
        lgn_u=jnp.array(net.lgn.u, dtype=jnp.float32),
        v1_v=jnp.array(net.v1_exc.v, dtype=jnp.float32),
        v1_u=jnp.array(net.v1_exc.u, dtype=jnp.float32),
        pv_v=jnp.array(net.pv.v, dtype=jnp.float32),
        pv_u=jnp.array(net.pv.u, dtype=jnp.float32),
        som_v=jnp.array(net.som.v, dtype=jnp.float32),
        som_u=jnp.array(net.som.u, dtype=jnp.float32),
        I_lgn=jnp.array(net.I_lgn, dtype=jnp.float32),
        g_exc_ff=jnp.array(net.g_exc_ff, dtype=jnp.float32),
        g_exc_ee=jnp.array(net.g_exc_ee, dtype=jnp.float32),
        g_v1_inh_pv_rise=jnp.array(net.g_v1_inh_pv_rise, dtype=jnp.float32),
        g_v1_inh_pv_decay=jnp.array(net.g_v1_inh_pv_decay, dtype=jnp.float32),
        g_v1_inh_som=jnp.array(net.g_v1_inh_som, dtype=jnp.float32),
        g_v1_apical=jnp.array(net.g_v1_apical, dtype=jnp.float32),
        I_pv=jnp.array(net.I_pv, dtype=jnp.float32),
        I_pv_inh=jnp.array(net.I_pv_inh, dtype=jnp.float32),
        I_som=jnp.array(net.I_som, dtype=jnp.float32),
        I_som_inh=jnp.array(net.I_som_inh, dtype=jnp.float32),
        I_v1_bias=jnp.array(net.I_v1_bias, dtype=jnp.float32),
        delay_buf=jnp.array(net.delay_buf, dtype=jnp.float32),
        ptr=jnp.int32(net.ptr),
        delay_buf_ee=jnp.array(net.delay_buf_ee, dtype=jnp.float32),
        ptr_ee=jnp.int32(net.ptr_ee),
        stdp_x_pre=jnp.array(net.stdp.x_pre, dtype=jnp.float32),
        stdp_x_pre_slow=jnp.array(net.stdp.x_pre_slow, dtype=jnp.float32),
        stdp_x_post=jnp.array(net.stdp.x_post, dtype=jnp.float32),
        stdp_x_post_slow=jnp.array(net.stdp.x_post_slow, dtype=jnp.float32),
        pv_istdp_x_post=jnp.array(net.pv_istdp.x_post, dtype=jnp.float32),
        W=jnp.array(net.W, dtype=jnp.float32),
        W_pv_e=jnp.array(net.W_pv_e, dtype=jnp.float32),
        W_e_e=jnp.array(net.W_e_e, dtype=jnp.float32),
        ee_pre_trace=jnp.array(net.delay_ee_stdp.pre_trace, dtype=jnp.float32),
        ee_post_trace=jnp.array(net.delay_ee_stdp.post_trace, dtype=jnp.float32),
        prev_v1_spk=jnp.array(net.prev_v1_spk, dtype=jnp.float32),
        rng_key=jax.random.PRNGKey(p.seed),
        rate_avg=jnp.array(net.homeostasis.rate_avg, dtype=jnp.float32),
        lgn_rgc_drive=jnp.array(net._lgn_rgc_drive, dtype=jnp.float32),
        drive_acc_ff=jnp.zeros(p.M, dtype=jnp.float32),
        drive_acc_ee=jnp.zeros(p.M, dtype=jnp.float32),
        drive_acc_steps=jnp.int32(0),
    )

    # Compute derived scalar parameters
    dt = float(p.dt_ms)
    decay_ampa = math.exp(-dt / float(p.tau_ampa))
    decay_gaba = math.exp(-dt / float(p.tau_gaba))
    decay_gaba_rise_pv = math.exp(-dt / max(1e-3, float(p.tau_gaba_rise_pv)))
    decay_apical = math.exp(-dt / max(1e-3, float(p.tau_apical)))

    # STDP decay constants
    decay_pre = math.exp(-dt / float(p.tau_plus))
    decay_pre_slow = math.exp(-dt / float(p.tau_x))
    decay_post = math.exp(-dt / float(p.tau_minus))
    decay_post_slow = math.exp(-dt / float(p.tau_y))

    # PV iSTDP
    pv_istdp_decay = math.exp(-dt / float(p.tau_pv_istdp))
    pv_istdp_rho = float(p.target_rate_hz * (p.tau_pv_istdp / 1000.0))

    # Homeostasis decay
    homeostasis_decay = math.exp(-dt / float(p.tau_homeostasis))

    # LGN-RGC temporal smoothing alpha
    lgn_rgc_alpha = 0.0
    if p.lgn_rgc_tau_ms > 0:
        lgn_rgc_alpha = float(1.0 - math.exp(-dt / max(1e-6, float(p.lgn_rgc_tau_ms))))

    # DoG grating gain
    dog_grating_gain = getattr(net, '_rgc_dog_grating_gain', 1.0)

    n_pix = int(p.N * p.N)

    static = StaticConfig(
        W_rgc_lgn=jnp.array(net.W_rgc_lgn, dtype=jnp.float32),
        W_e_pv=jnp.array(net.W_e_pv, dtype=jnp.float32),
        W_lgn_pv=jnp.array(net.W_lgn_pv, dtype=jnp.float32),
        W_e_som=jnp.array(net.W_e_som, dtype=jnp.float32),
        W_som_e=jnp.array(net.W_som_e, dtype=jnp.float32),
        tc_mask_e_f32=jnp.array(net.tc_mask_e_f32, dtype=jnp.float32),
        tc_mask_pv_f32=jnp.array(net.tc_mask_pv_f32, dtype=jnp.float32),
        lgn_mask_e=jnp.array(net.lgn_mask_e, dtype=jnp.float32),
        mask_pv_e=jnp.array(net.mask_pv_e, dtype=jnp.float32),
        mask_e_e=jnp.array(net.mask_e_e, dtype=jnp.float32),
        D=jnp.array(net.D, dtype=jnp.int32),
        D_pv=jnp.array(net.D_pv, dtype=jnp.int32),
        D_ee=jnp.array(net.D_ee, dtype=jnp.int32),
        lgn_ids=jnp.arange(net.n_lgn, dtype=jnp.int32).reshape(1, -1),
        X_on=jnp.array(net.X_on, dtype=jnp.float32),
        Y_on=jnp.array(net.Y_on, dtype=jnp.float32),
        X_off=jnp.array(net.X_off, dtype=jnp.float32),
        Y_off=jnp.array(net.Y_off, dtype=jnp.float32),
        on_to_off=jnp.array(net.on_to_off, dtype=jnp.int32),
        off_to_on=jnp.array(net.off_to_on, dtype=jnp.int32),
        split_target_on=jnp.array(net.split_target_on, dtype=jnp.float32),
        split_target_off=jnp.array(net.split_target_off, dtype=jnp.float32),
        eye_M=jnp.eye(p.M, dtype=jnp.float32),
        arange_M=jnp.arange(p.M, dtype=jnp.int32),
        arange_lgn=jnp.arange(net.n_lgn, dtype=jnp.int32),
        N=int(p.N),
        M=int(p.M),
        n_lgn=int(net.n_lgn),
        n_pv=int(net.n_pv),
        n_som=int(net.n_som),
        n_pix=n_pix,
        L=int(net.L),
        L_ee=int(net.L_ee),
        dt_ms=float(p.dt_ms),
        segment_ms=float(p.segment_ms),
        steps=int(p.segment_ms / p.dt_ms),
        spatial_freq=float(p.spatial_freq),
        temporal_freq=float(p.temporal_freq),
        base_rate=float(p.base_rate),
        gain_rate=float(p.gain_rate),
        dog_grating_gain=float(dog_grating_gain),
        w_rgc_lgn_scalar=float(p.w_rgc_lgn),
        decay_ampa=decay_ampa,
        decay_gaba=decay_gaba,
        decay_gaba_rise_pv=decay_gaba_rise_pv,
        decay_apical=decay_apical,
        w_exc_gain=float(p.w_exc_gain),
        E_exc=float(p.E_exc),
        E_inh=float(p.E_inh),
        # LGN (TC)
        lgn_a=0.02, lgn_b=0.25, lgn_c=-65.0, lgn_d=0.05, lgn_v_peak=30.0,
        # V1 E (RS)
        v1_a=0.02, v1_b=0.2, v1_c=-65.0, v1_d=8.0, v1_v_peak=30.0,
        # PV (FS)
        pv_a=0.1, pv_b=0.2, pv_c=-65.0, pv_d=2.0, pv_v_peak=30.0,
        # SOM (LTS)
        som_a=0.02, som_b=0.25, som_c=-65.0, som_d=2.0, som_v_peak=30.0,
        # STDP
        decay_pre=decay_pre,
        decay_pre_slow=decay_pre_slow,
        decay_post=decay_post,
        decay_post_slow=decay_post_slow,
        A2_plus=float(p.A2_plus),
        A3_plus=float(p.A3_plus),
        A2_minus=float(p.A2_minus),
        w_max=float(p.w_max),
        A_het=float(p.A_het),
        A_split=float(p.A_split),
        w_decay=float(p.w_decay),
        # PV iSTDP
        pv_istdp_decay=pv_istdp_decay,
        pv_istdp_eta=float(p.eta_pv_istdp),
        pv_istdp_rho=pv_istdp_rho,
        w_pv_e_max=float(p.w_pv_e_max),
        # Homeostasis
        target_rate_hz=float(p.target_rate_hz),
        homeostasis_rate=float(p.homeostasis_rate),
        homeostasis_clip=float(p.homeostasis_clip),
        homeostasis_decay=homeostasis_decay,
        # Split constraint
        split_constraint_rate=float(p.split_constraint_rate),
        split_constraint_clip=float(p.split_constraint_clip),
        # Bias homeostasis
        v1_bias_eta=float(p.v1_bias_eta),
        v1_bias_clip=float(p.v1_bias_clip),
        # Flags
        pv_inhib_plastic=bool(p.pv_inhib_plastic),
        ff_plastic_enabled=bool(net.ff_plastic_enabled),
        lgn_rgc_alpha=lgn_rgc_alpha,
        w_lgn_pv_gain=float(p.w_lgn_pv_gain),
        # E→E delay-aware STDP
        ee_stdp_enabled=bool(p.ee_stdp_enabled),
        ee_stdp_decay_pre=math.exp(-dt / max(1e-6, float(p.ee_stdp_tau_pre_ms))),
        ee_stdp_decay_post=math.exp(-dt / max(1e-6, float(p.ee_stdp_tau_post_ms))),
        ee_stdp_A_plus=float(p.ee_stdp_A_plus),
        ee_stdp_A_minus=float(p.ee_stdp_A_minus),
        ee_stdp_weight_dep=bool(p.ee_stdp_weight_dep),
        w_e_e_min=float(p.w_e_e_min),
        w_e_e_max=float(p.w_e_e_max),
    )

    return state, static


def jax_state_to_numpy_net(state: SimState, net) -> None:
    """Write JAX SimState back into the corresponding numpy RgcLgnV1Network (in-place)."""
    # Population states
    net.lgn.v = np.array(state.lgn_v, dtype=np.float32)
    net.lgn.u = np.array(state.lgn_u, dtype=np.float32)
    net.v1_exc.v = np.array(state.v1_v, dtype=np.float32)
    net.v1_exc.u = np.array(state.v1_u, dtype=np.float32)
    net.pv.v = np.array(state.pv_v, dtype=np.float32)
    net.pv.u = np.array(state.pv_u, dtype=np.float32)
    net.som.v = np.array(state.som_v, dtype=np.float32)
    net.som.u = np.array(state.som_u, dtype=np.float32)

    # Synaptic / conductance state
    net.I_lgn = np.array(state.I_lgn, dtype=np.float32)
    net.g_exc_ff = np.array(state.g_exc_ff, dtype=np.float32)
    net.g_exc_ee = np.array(state.g_exc_ee, dtype=np.float32)
    net.g_v1_inh_pv_rise = np.array(state.g_v1_inh_pv_rise, dtype=np.float32)
    net.g_v1_inh_pv_decay = np.array(state.g_v1_inh_pv_decay, dtype=np.float32)
    net.g_v1_inh_som = np.array(state.g_v1_inh_som, dtype=np.float32)
    net.g_v1_apical = np.array(state.g_v1_apical, dtype=np.float32)
    net.I_pv = np.array(state.I_pv, dtype=np.float32)
    net.I_pv_inh = np.array(state.I_pv_inh, dtype=np.float32)
    net.I_som = np.array(state.I_som, dtype=np.float32)
    net.I_som_inh = np.array(state.I_som_inh, dtype=np.float32)
    net.I_v1_bias = np.array(state.I_v1_bias, dtype=np.float32)

    # Delay buffers
    net.delay_buf = np.array(state.delay_buf, dtype=np.uint8)
    net.ptr = int(state.ptr)
    net.delay_buf_ee = np.array(state.delay_buf_ee, dtype=np.uint8)
    net.ptr_ee = int(state.ptr_ee)

    # STDP traces
    net.stdp.x_pre = np.array(state.stdp_x_pre, dtype=np.float32)
    net.stdp.x_pre_slow = np.array(state.stdp_x_pre_slow, dtype=np.float32)
    net.stdp.x_post = np.array(state.stdp_x_post, dtype=np.float32)
    net.stdp.x_post_slow = np.array(state.stdp_x_post_slow, dtype=np.float32)

    # PV iSTDP trace
    net.pv_istdp.x_post = np.array(state.pv_istdp_x_post, dtype=np.float32)

    # E→E delay-aware STDP traces
    net.delay_ee_stdp.pre_trace = np.array(state.ee_pre_trace, dtype=np.float32)
    net.delay_ee_stdp.post_trace = np.array(state.ee_post_trace, dtype=np.float32)

    # Weights
    net.W = np.array(state.W, dtype=np.float32)
    net.W_pv_e = np.array(state.W_pv_e, dtype=np.float32)
    net.W_e_e = np.array(state.W_e_e, dtype=np.float32)

    # Previous spikes
    net.prev_v1_spk = np.array(state.prev_v1_spk, dtype=np.uint8)

    # Homeostasis
    net.homeostasis.rate_avg = np.array(state.rate_avg, dtype=np.float32)

    # LGN-RGC drive
    net._lgn_rgc_drive = np.array(state.lgn_rgc_drive, dtype=np.float32)

    # Drive accumulators
    net._drive_acc_ff = np.array(state.drive_acc_ff, dtype=np.float64)
    net._drive_acc_ee = np.array(state.drive_acc_ee, dtype=np.float64)
    net._drive_acc_steps = int(state.drive_acc_steps)


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def izh_step(v, u, I_ext, a, b, c, d, v_peak, dt):
    """Izhikevich neuron step (2 sub-steps for stability).

    All inputs/outputs are float32 arrays. Spikes are float32 0.0/1.0.

    Parameters
    ----------
    v, u : (N,) membrane potential and recovery variable
    I_ext : (N,) external current
    a, b, c, d : Izhikevich parameters (scalars)
    v_peak : spike threshold (scalar)
    dt : timestep in ms (scalar)

    Returns
    -------
    v_new, u_new, spikes : updated state and float32 spike indicator
    """
    dt_sub = dt / 2.0

    # Sub-step 1
    v_c = jnp.clip(v, -100.0, v_peak)
    dv = (0.04 * v_c * v_c + 5.0 * v_c + 140.0 - u + I_ext) * dt_sub
    du = a * (b * v_c - u) * dt_sub
    v = v + dv
    u = u + du

    # Sub-step 2
    v_c = jnp.clip(v, -100.0, v_peak)
    dv = (0.04 * v_c * v_c + 5.0 * v_c + 140.0 - u + I_ext) * dt_sub
    du = a * (b * v_c - u) * dt_sub
    v = v + dv
    u = u + du

    # Spike detection and reset
    spikes = (v >= v_peak).astype(jnp.float32)
    v = jnp.where(spikes > 0.5, c, v)
    u = jnp.where(spikes > 0.5, u + d, u)

    return v, u, spikes


def grating_on_coords(theta_deg, t_ms, phase, X, Y, spatial_freq, temporal_freq):
    """Drifting grating sampled at coordinate arrays X, Y.

    Returns float32 array same shape as X, Y.
    """
    th = theta_deg * (jnp.pi / 180.0)
    coord = X * jnp.cos(th) + Y * jnp.sin(th)
    return jnp.sin(
        2.0 * jnp.pi * (spatial_freq * coord - temporal_freq * (t_ms / 1000.0)) + phase
    )


def rgc_drives_grating_jax(theta_deg, t_ms, phase, contrast,
                            X_on, Y_on, X_off, Y_off,
                            spatial_freq, temporal_freq, dog_grating_gain):
    """Compute (ON, OFF) RGC drive for a drifting grating.

    For the default padded_fft path, the DoG is a scalar gain on grating input.
    No FFT needed — gratings are eigenfunctions of LTI filters.

    Returns drive_on (N,N), drive_off (N,N) as float32.
    """
    g = dog_grating_gain * contrast
    drive_on = g * grating_on_coords(theta_deg, t_ms, phase, X_on, Y_on,
                                     spatial_freq, temporal_freq)
    drive_off = g * grating_on_coords(theta_deg, t_ms, phase, X_off, Y_off,
                                      spatial_freq, temporal_freq)
    return drive_on, drive_off


def rgc_spikes_jax(drive_on, drive_off, base_rate, gain_rate, dt_ms, key):
    """Generate ON and OFF RGC Poisson spikes from drive fields.

    Simplified path: no temporal filter, no refractory period.

    Parameters
    ----------
    drive_on, drive_off : (N, N) drive fields
    base_rate, gain_rate : RGC rate parameters
    dt_ms : timestep in ms
    key : JAX PRNGKey

    Returns
    -------
    on_spk, off_spk : (N, N) float32 spike arrays (0.0/1.0)
    """
    dt_s = dt_ms / 1000.0
    on_rate = base_rate + gain_rate * jnp.clip(drive_on, 0.0, None)
    off_rate = base_rate + gain_rate * jnp.clip(-drive_off, 0.0, None)

    key_on, key_off = jax.random.split(key)
    on_spk = (jax.random.uniform(key_on, drive_on.shape) < (on_rate * dt_s)).astype(jnp.float32)
    off_spk = (jax.random.uniform(key_off, drive_off.shape) < (off_rate * dt_s)).astype(jnp.float32)

    return on_spk, off_spk


def triplet_stdp_update(stdp_x_pre, stdp_x_pre_slow, stdp_x_post, stdp_x_post_slow,
                        arrivals, v1_spk, W,
                        decay_pre, decay_pre_slow, decay_post, decay_post_slow,
                        A2_plus, A3_plus, A2_minus, w_max, A_het, A_split,
                        on_to_off, off_to_on):
    """Triplet STDP update (pure functional).

    Ports TripletSTDP.update() from the numpy code.
    n_pix is derived from array shapes (W has shape (M, 2*n_pix)).

    Returns (new_x_pre, new_x_pre_slow, new_x_post, new_x_post_slow, dW).
    """
    # Derive n_pix from array shape (concrete during tracing)
    n_pix = W.shape[1] // 2

    # 1. Decay all traces
    x_pre = stdp_x_pre * decay_pre
    x_pre_slow = stdp_x_pre_slow * decay_pre_slow
    x_post = stdp_x_post * decay_post
    x_post_slow = stdp_x_post_slow * decay_post_slow

    dW = jnp.zeros_like(W)

    # 2. LTD: when pre spike arrives, depress based on OLD post trace
    # Multiplicative: dW- proportional to W (stronger synapses lose more)
    dW = dW - A2_minus * arrivals * x_post[:, None] * W

    # 3. Update pre traces BEFORE computing LTP
    x_pre = x_pre + arrivals
    x_pre_slow = x_pre_slow + arrivals

    # 4. LTP: when post fires, potentiate based on NEW pre trace (includes current arrivals)
    post_mask = v1_spk  # already float32
    triplet_boost = 1.0 + A3_plus * x_post_slow[:, None] / (A2_plus + 1e-30)
    dW = dW + A2_plus * post_mask[:, None] * x_pre * (w_max - W) * triplet_boost

    # 5. Heterosynaptic depression: postsynaptic spiking depresses inactive synapses
    inactive = 1.0 - arrivals
    dW = dW - A_het * post_mask[:, None] * inactive * W

    # 6. ON/OFF split competition
    # When a postsynaptic spike occurs, recently active ON inputs weaken their OFF
    # counterparts (and vice versa).
    on_trace = x_pre[:, :n_pix]
    off_trace = x_pre[:, n_pix:]

    # Map traces across channels using the nearest-neighbor ON<->OFF mapping
    on_at_off = on_trace[:, off_to_on]  # ON trace at OFF positions
    off_at_on = off_trace[:, on_to_off]  # OFF trace at ON positions

    # Depression of OFF weights by ON activity
    dW_off = -A_split * post_mask[:, None] * on_at_off * W[:, n_pix:]
    # Depression of ON weights by OFF activity
    dW_on = -A_split * post_mask[:, None] * off_at_on * W[:, :n_pix]

    dW = dW.at[:, :n_pix].add(dW_on)
    dW = dW.at[:, n_pix:].add(dW_off)

    # 7. Update post traces AFTER computing plasticity
    x_post = x_post + v1_spk
    x_post_slow = x_post_slow + v1_spk

    return x_pre, x_pre_slow, x_post, x_post_slow, dW


def pv_istdp_update(x_post, pv_spk, v1_spk, W_pv_e, mask_pv_e,
                    decay, eta, rho, w_max):
    """PV inhibitory plasticity update (pure functional).

    Ports PVInhibitoryPlasticity.update() from the numpy code.

    Returns (new_x_post, new_W_pv_e).
    """
    # Decay and add post spikes
    x_post_new = x_post * decay + v1_spk

    # iSTDP: on pre (PV) spikes, update inhibitory weights
    delta = eta * (x_post_new - rho)  # (M,)
    dW = delta[:, None] * pv_spk[None, :] * mask_pv_e
    W_pv_e_new = jnp.clip(W_pv_e + dW, 0.0, w_max)

    return x_post_new, W_pv_e_new


def timestep(state, static, t_ms, theta_deg, phase, contrast, step_key):
    """Advance the network by one timestep (pure function).

    Ports step() from the numpy code for the grating stimulus path.

    Parameters
    ----------
    state : SimState
    static : StaticConfig
    t_ms : float, current time within segment
    theta_deg : float, grating orientation
    phase : float, grating phase
    contrast : float, stimulus contrast
    step_key : JAX PRNGKey for this timestep

    Returns
    -------
    (new_state, v1_spk, arrivals_tc, pv_spk, ee_arrivals)
    """
    s = static  # shorthand

    # --- RGC spikes ---
    drive_on, drive_off = rgc_drives_grating_jax(
        theta_deg, t_ms, phase, contrast,
        s.X_on, s.Y_on, s.X_off, s.Y_off,
        s.spatial_freq, s.temporal_freq, s.dog_grating_gain)

    key_rgc, key_rest = jax.random.split(step_key)
    on_spk, off_spk = rgc_spikes_jax(drive_on, drive_off,
                                      s.base_rate, s.gain_rate,
                                      s.dt_ms, key_rgc)

    # Combine ON/OFF RGC spikes: (n_lgn,)
    rgc = jnp.concatenate([on_spk.ravel(), off_spk.ravel()])

    # RGC->LGN pooling
    rgc_lgn = s.W_rgc_lgn @ rgc

    # Optional temporal smoothing of pooled RGC drive
    lgn_rgc_drive = state.lgn_rgc_drive
    lgn_rgc_drive = jnp.where(
        s.lgn_rgc_alpha > 0.0,
        lgn_rgc_drive + s.lgn_rgc_alpha * (rgc_lgn - lgn_rgc_drive),
        lgn_rgc_drive
    )
    rgc_lgn_eff = jnp.where(s.lgn_rgc_alpha > 0.0, lgn_rgc_drive, rgc_lgn)

    # --- LGN layer ---
    I_lgn = state.I_lgn * s.decay_ampa + s.w_rgc_lgn_scalar * rgc_lgn_eff
    lgn_v, lgn_u, lgn_spk = izh_step(
        state.lgn_v, state.lgn_u, I_lgn,
        s.lgn_a, s.lgn_b, s.lgn_c, s.lgn_d, s.lgn_v_peak, s.dt_ms)

    # Store LGN spikes in delay buffer
    delay_buf = state.delay_buf.at[state.ptr, :].set(lgn_spk)

    # Get delayed LGN spikes arriving at V1: (M, n_lgn)
    idx = (state.ptr - s.D) % s.L
    arrivals = delay_buf[idx, s.arange_lgn[None, :]]  # (M, n_lgn)
    arrivals_tc = arrivals * s.tc_mask_e_f32

    # --- V1 feedforward input (no STP) ---
    I_ff = (state.W * arrivals_tc).sum(axis=1)  # (M,)

    # --- V1 excitatory conductances ---
    g_exc_ff = state.g_exc_ff * s.decay_ampa + s.w_exc_gain * I_ff

    # Recurrent E->E conductance (delayed lateral excitation)
    g_exc_ee = state.g_exc_ee * s.decay_ampa
    ee_idx = (state.ptr_ee - s.D_ee) % s.L_ee  # (M, M)
    ee_arrivals = state.delay_buf_ee[ee_idx, s.arange_M[None, :]]  # (M, M)
    # Zero diagonal (no self-connections)
    ee_arrivals = ee_arrivals * (1.0 - s.eye_M)
    I_ee = (state.W_e_e * ee_arrivals).sum(axis=1)
    g_exc_ee = g_exc_ee + s.w_exc_gain * I_ee

    # Drive accumulators
    drive_acc_ff = state.drive_acc_ff + g_exc_ff
    drive_acc_ee = state.drive_acc_ee + g_exc_ee
    drive_acc_steps = state.drive_acc_steps + 1

    # Apical conductance decay (no external apical drive in grating path)
    g_v1_apical = state.g_v1_apical * s.decay_apical

    # --- Inhibitory conductances (GABA decay) ---
    g_v1_inh_pv_rise = state.g_v1_inh_pv_rise * s.decay_gaba_rise_pv
    g_v1_inh_pv_decay = state.g_v1_inh_pv_decay * s.decay_gaba
    g_v1_inh_som = state.g_v1_inh_som * s.decay_gaba

    # --- PV interneurons (feedforward inhibition; runs BEFORE E) ---
    I_pv = state.I_pv * s.decay_ampa
    I_pv_inh = state.I_pv_inh * s.decay_gaba

    # Thalamocortical drive to PV
    idx_pv = (state.ptr - s.D_pv) % s.L
    arrivals_pv = delay_buf[idx_pv, s.arange_lgn[None, :]]  # (n_pv, n_lgn)
    arrivals_pv_tc = arrivals_pv * s.tc_mask_pv_f32
    I_pv = I_pv + s.w_lgn_pv_gain * (s.W_lgn_pv * arrivals_pv_tc).sum(axis=1)

    # Local recurrent E->PV (delayed by one step)
    I_pv = I_pv + s.W_e_pv @ state.prev_v1_spk

    # PV step
    pv_v, pv_u, pv_spk = izh_step(
        state.pv_v, state.pv_u, I_pv - I_pv_inh,
        s.pv_a, s.pv_b, s.pv_c, s.pv_d, s.pv_v_peak, s.dt_ms)

    # PV->E inhibition (GABA conductance increment with rise time)
    g_pv_inc = state.W_pv_e @ pv_spk  # (M,)
    g_v1_inh_pv_rise = g_v1_inh_pv_rise + g_pv_inc
    g_v1_inh_pv_decay = g_v1_inh_pv_decay + g_pv_inc

    # --- V1 excitatory integration (conductance-based) ---
    g_pv = jnp.clip(g_v1_inh_pv_decay - g_v1_inh_pv_rise, 0.0, None)
    g_inh = g_pv + g_v1_inh_som
    g_v1_exc = g_exc_ff + g_exc_ee
    I_exc = g_v1_exc * (s.E_exc - state.v1_v)
    I_v1_total = I_exc + g_inh * (s.E_inh - state.v1_v) + state.I_v1_bias

    v1_v, v1_u, v1_spk = izh_step(
        state.v1_v, state.v1_u, I_v1_total,
        s.v1_a, s.v1_b, s.v1_c, s.v1_d, s.v1_v_peak, s.dt_ms)

    # --- SOM interneurons (lateral inhibition; updated AFTER E, affects next step) ---
    I_som = state.I_som * s.decay_ampa
    I_som_inh = state.I_som_inh * s.decay_gaba
    I_som = I_som + s.W_e_som @ v1_spk
    som_v, som_u, som_spk = izh_step(
        state.som_v, state.som_u, I_som - I_som_inh,
        s.som_a, s.som_b, s.som_c, s.som_d, s.som_v_peak, s.dt_ms)

    # SOM->E lateral inhibition (GABA conductance increment; affects next step)
    g_v1_inh_som = g_v1_inh_som + s.W_som_e @ som_spk

    # --- Write V1 E spikes into E->E delay buffer ---
    delay_buf_ee = state.delay_buf_ee.at[state.ptr_ee, :].set(v1_spk)

    # --- Update delay buffer pointers ---
    ptr = (state.ptr + 1) % s.L
    ptr_ee = (state.ptr_ee + 1) % s.L_ee

    # Assemble new state (plasticity fields will be updated conditionally)
    new_state = SimState(
        lgn_v=lgn_v,
        lgn_u=lgn_u,
        v1_v=v1_v,
        v1_u=v1_u,
        pv_v=pv_v,
        pv_u=pv_u,
        som_v=som_v,
        som_u=som_u,
        I_lgn=I_lgn,
        g_exc_ff=g_exc_ff,
        g_exc_ee=g_exc_ee,
        g_v1_inh_pv_rise=g_v1_inh_pv_rise,
        g_v1_inh_pv_decay=g_v1_inh_pv_decay,
        g_v1_inh_som=g_v1_inh_som,
        g_v1_apical=g_v1_apical,
        I_pv=I_pv,
        I_pv_inh=I_pv_inh,
        I_som=I_som,
        I_som_inh=I_som_inh,
        I_v1_bias=state.I_v1_bias,
        delay_buf=delay_buf,
        ptr=ptr,
        delay_buf_ee=delay_buf_ee,
        ptr_ee=ptr_ee,
        stdp_x_pre=state.stdp_x_pre,
        stdp_x_pre_slow=state.stdp_x_pre_slow,
        stdp_x_post=state.stdp_x_post,
        stdp_x_post_slow=state.stdp_x_post_slow,
        pv_istdp_x_post=state.pv_istdp_x_post,
        W=state.W,
        W_pv_e=state.W_pv_e,
        W_e_e=state.W_e_e,
        ee_pre_trace=state.ee_pre_trace,
        ee_post_trace=state.ee_post_trace,
        prev_v1_spk=v1_spk,
        rng_key=state.rng_key,
        rate_avg=state.rate_avg,
        lgn_rgc_drive=lgn_rgc_drive,
        drive_acc_ff=drive_acc_ff,
        drive_acc_ee=drive_acc_ee,
        drive_acc_steps=drive_acc_steps,
    )

    return new_state, v1_spk, arrivals_tc, pv_spk, ee_arrivals


def timestep_plastic(state, static, t_ms, theta_deg, phase, contrast, step_key):
    """Timestep with plasticity updates.

    Calls timestep() then applies STDP and PV iSTDP.
    """
    s = static
    new_state, v1_spk, arrivals_tc, pv_spk, _ee_arrivals = timestep(
        state, static, t_ms, theta_deg, phase, contrast, step_key)

    # --- Feedforward STDP ---
    x_pre, x_pre_slow, x_post, x_post_slow, dW = triplet_stdp_update(
        new_state.stdp_x_pre, new_state.stdp_x_pre_slow,
        new_state.stdp_x_post, new_state.stdp_x_post_slow,
        arrivals_tc, v1_spk, new_state.W,
        s.decay_pre, s.decay_pre_slow, s.decay_post, s.decay_post_slow,
        s.A2_plus, s.A3_plus, s.A2_minus, s.w_max,
        s.A_het, s.A_split,
        s.on_to_off, s.off_to_on)

    # Apply weight changes
    W_new = new_state.W + dW
    # Weight decay
    W_new = W_new * (1.0 - s.w_decay)
    # Clip to valid range
    W_new = jnp.clip(W_new, 0.0, s.w_max)
    # Retinotopic cap
    W_new = jnp.minimum(W_new, s.w_max * s.lgn_mask_e)
    # Structural sparsity mask
    W_new = W_new * s.tc_mask_e_f32

    # --- PV iSTDP ---
    pv_istdp_x_post_new, W_pv_e_new = pv_istdp_update(
        new_state.pv_istdp_x_post, pv_spk, v1_spk,
        new_state.W_pv_e, s.mask_pv_e,
        s.pv_istdp_decay, s.pv_istdp_eta, s.pv_istdp_rho, s.w_pv_e_max)

    # --- Homeostatic rate estimate (EMA) ---
    instant_rate = v1_spk * (1000.0 / s.dt_ms)
    rate_avg_new = s.homeostasis_decay * new_state.rate_avg + (1.0 - s.homeostasis_decay) * instant_rate

    new_state = new_state._replace(
        stdp_x_pre=x_pre,
        stdp_x_pre_slow=x_pre_slow,
        stdp_x_post=x_post,
        stdp_x_post_slow=x_post_slow,
        pv_istdp_x_post=pv_istdp_x_post_new,
        W=W_new,
        W_pv_e=W_pv_e_new,
        rate_avg=rate_avg_new,
    )

    return new_state, v1_spk


def timestep_nonplastic(state, static, t_ms, theta_deg, phase, contrast, step_key):
    """Timestep without plasticity (evaluation mode)."""
    new_state, v1_spk, _arrivals_tc, _pv_spk, _ee_arrivals = timestep(
        state, static, t_ms, theta_deg, phase, contrast, step_key)
    return new_state, v1_spk


def delay_aware_ee_stdp_update(
    ee_pre_trace, ee_post_trace,
    ee_arrivals, post_spikes,
    W_e_e, mask_e_e,
    decay_pre, decay_post,
    A_plus, A_minus,
    w_min, w_max,
    weight_dep,
):
    """Delay-aware pair-based STDP for E→E connections (pure functional).

    Ports DelayAwareEESTDP.update() from biologically_plausible_v1_stdp.py.

    Uses per-synapse pre-traces (M×M) because heterogeneous conduction delays
    make pre-spike arrival times synapse-specific. Post traces are per-neuron (M).

    Critical order:
    1. Decay traces
    2. LTD: use OLD post trace with ee_arrivals
    3. Update pre trace += arrivals
    4. LTP: use NEW pre trace with post_spikes
    5. Update post trace += post_spikes
    6. Apply mask

    Parameters
    ----------
    ee_pre_trace : (M, M) float32 — per-synapse pre-trace
    ee_post_trace : (M,) float32 — per-neuron post-trace
    ee_arrivals : (M, M) float32 — delayed pre-synaptic spike arrivals (post, pre)
    post_spikes : (M,) float32 — binary post-synaptic spike vector
    W_e_e : (M, M) float32 — current E→E weight matrix
    mask_e_e : (M, M) float32 — structural connectivity mask
    decay_pre, decay_post : float — trace decay constants
    A_plus, A_minus : float — learning rates (already ramped)
    w_min, w_max : float — weight bounds
    weight_dep : bool — if True, use weight-dependent STDP (concrete in closure)

    Returns
    -------
    (pre_trace, post_trace, dW) : updated traces and weight change matrix
    """
    # 1. Decay traces
    pre_trace = ee_pre_trace * decay_pre
    post_trace = ee_post_trace * decay_post

    dW = jnp.zeros_like(W_e_e)

    # 2. LTD: on pre-arrival, depress using OLD post trace
    if weight_dep:
        dW = dW - A_minus * ee_arrivals * post_trace[:, None] * (W_e_e - w_min)
    else:
        dW = dW - A_minus * ee_arrivals * post_trace[:, None]

    # 3. Update pre trace with current arrivals
    pre_trace = pre_trace + ee_arrivals

    # 4. LTP: on post spike, potentiate using NEW pre trace
    if weight_dep:
        dW = dW + A_plus * post_spikes[:, None] * pre_trace * (w_max - W_e_e)
    else:
        dW = dW + A_plus * post_spikes[:, None] * pre_trace

    # 5. Update post trace
    post_trace = post_trace + post_spikes

    # 6. Apply structural mask
    dW = dW * mask_e_e

    return pre_trace, post_trace, dW


def timestep_phaseb_plastic(state, static, t_ms, theta_deg, phase, contrast, step_key,
                             ee_A_plus_eff, ee_A_minus_eff):
    """Timestep with Phase B plasticity: E→E STDP only, NO feedforward STDP.

    Parameters
    ----------
    state : SimState
    static : StaticConfig
    t_ms : float, current time within element
    theta_deg : float, grating orientation
    phase : float, grating phase
    contrast : float, stimulus contrast (0.0 for blank/omission)
    step_key : JAX PRNGKey
    ee_A_plus_eff : float, effective LTP rate (includes ramp factor)
    ee_A_minus_eff : float, effective LTD rate (includes ramp factor)

    Returns
    -------
    (new_state, v1_spk) where v1_spk is (M,) float32
    """
    s = static
    new_state, v1_spk, _arrivals_tc, _pv_spk, ee_arrivals = timestep(
        state, static, t_ms, theta_deg, phase, contrast, step_key)

    # E→E STDP (the ONLY plasticity in Phase B)
    pre_trace, post_trace, dW_ee = delay_aware_ee_stdp_update(
        new_state.ee_pre_trace, new_state.ee_post_trace,
        ee_arrivals, v1_spk,
        new_state.W_e_e, s.mask_e_e,
        s.ee_stdp_decay_pre, s.ee_stdp_decay_post,
        ee_A_plus_eff, ee_A_minus_eff,
        s.w_e_e_min, s.w_e_e_max,
        s.ee_stdp_weight_dep)

    W_e_e_new = new_state.W_e_e + dW_ee
    W_e_e_new = jnp.clip(W_e_e_new, s.w_e_e_min, s.w_e_e_max)
    W_e_e_new = W_e_e_new * (1.0 - s.eye_M)  # zero diagonal

    # Homeostatic rate estimate
    instant_rate = v1_spk * (1000.0 / s.dt_ms)
    rate_avg_new = s.homeostasis_decay * new_state.rate_avg + (1.0 - s.homeostasis_decay) * instant_rate

    return new_state._replace(
        ee_pre_trace=pre_trace,
        ee_post_trace=post_trace,
        W_e_e=W_e_e_new,
        rate_avg=rate_avg_new,
    ), v1_spk


def reset_state_jax(state, static):
    """Reset dynamic state between evaluation trials, preserving weights.

    Ports reset_state() from biologically_plausible_v1_stdp.py.
    Zeros all conductances, currents, delay buffers, traces, and resets
    membrane potentials to -65 mV. Weight matrices are preserved.

    Parameters
    ----------
    state : SimState
    static : StaticConfig

    Returns
    -------
    SimState with reset dynamic variables
    """
    s = static
    v_init = -65.0
    return state._replace(
        lgn_v=jnp.full(s.n_lgn, v_init, dtype=jnp.float32),
        lgn_u=jnp.full(s.n_lgn, s.lgn_b * v_init, dtype=jnp.float32),
        v1_v=jnp.full(s.M, v_init, dtype=jnp.float32),
        v1_u=jnp.full(s.M, s.v1_b * v_init, dtype=jnp.float32),
        pv_v=jnp.full(s.n_pv, v_init, dtype=jnp.float32),
        pv_u=jnp.full(s.n_pv, s.pv_b * v_init, dtype=jnp.float32),
        som_v=jnp.full(s.n_som, v_init, dtype=jnp.float32),
        som_u=jnp.full(s.n_som, s.som_b * v_init, dtype=jnp.float32),
        I_lgn=jnp.zeros(s.n_lgn, dtype=jnp.float32),
        g_exc_ff=jnp.zeros(s.M, dtype=jnp.float32),
        g_exc_ee=jnp.zeros(s.M, dtype=jnp.float32),
        g_v1_inh_pv_rise=jnp.zeros(s.M, dtype=jnp.float32),
        g_v1_inh_pv_decay=jnp.zeros(s.M, dtype=jnp.float32),
        g_v1_inh_som=jnp.zeros(s.M, dtype=jnp.float32),
        g_v1_apical=jnp.zeros(s.M, dtype=jnp.float32),
        I_pv=jnp.zeros(s.n_pv, dtype=jnp.float32),
        I_pv_inh=jnp.zeros(s.n_pv, dtype=jnp.float32),
        I_som=jnp.zeros(s.n_som, dtype=jnp.float32),
        I_som_inh=jnp.zeros(s.n_som, dtype=jnp.float32),
        delay_buf=jnp.zeros((s.L, s.n_lgn), dtype=jnp.float32),
        ptr=jnp.int32(0),
        delay_buf_ee=jnp.zeros((s.L_ee, s.M), dtype=jnp.float32),
        ptr_ee=jnp.int32(0),
        stdp_x_pre=jnp.zeros((s.M, s.n_lgn), dtype=jnp.float32),
        stdp_x_pre_slow=jnp.zeros((s.M, s.n_lgn), dtype=jnp.float32),
        stdp_x_post=jnp.zeros(s.M, dtype=jnp.float32),
        stdp_x_post_slow=jnp.zeros(s.M, dtype=jnp.float32),
        pv_istdp_x_post=jnp.zeros(s.M, dtype=jnp.float32),
        ee_pre_trace=jnp.zeros((s.M, s.M), dtype=jnp.float32),
        ee_post_trace=jnp.zeros(s.M, dtype=jnp.float32),
        prev_v1_spk=jnp.zeros(s.M, dtype=jnp.float32),
        lgn_rgc_drive=jnp.zeros(s.n_lgn, dtype=jnp.float32),
        drive_acc_ff=jnp.zeros(s.M, dtype=jnp.float32),
        drive_acc_ee=jnp.zeros(s.M, dtype=jnp.float32),
        drive_acc_steps=jnp.int32(0),
    )


def build_trial_stimulus(thetas, n_elem, element_steps, iti_steps, contrast, phases,
                          omit_index, dt_ms):
    """Precompute per-step stimulus arrays for one full sequence trial.

    Called OUTSIDE JIT with concrete Python values for n_elem, element_steps, etc.
    Returns JAX arrays that are passed as inputs to the JIT-compiled runner.

    Parameters
    ----------
    thetas : array-like of float, orientations per element
    n_elem : int, number of sequence elements
    element_steps : int, timesteps per element
    iti_steps : int, timesteps for ITI blank
    contrast : float, stimulus contrast
    phases : array-like of float, random phase per element
    omit_index : int, index of element to omit (-1 = no omission)
    dt_ms : float, timestep in ms

    Returns
    -------
    (theta_arr, contrast_arr, phase_arr, t_ms_arr) — all (total_steps,) float32
    """
    total_steps = n_elem * element_steps + iti_steps
    theta_arr = jnp.zeros(total_steps, dtype=jnp.float32)
    contrast_arr = jnp.zeros(total_steps, dtype=jnp.float32)
    phase_arr = jnp.zeros(total_steps, dtype=jnp.float32)
    t_ms_arr = jnp.zeros(total_steps, dtype=jnp.float32)

    for i in range(n_elem):  # Python loop (concrete n_elem)
        start = i * element_steps
        end = (i + 1) * element_steps
        t_local = jnp.arange(element_steps, dtype=jnp.float32) * dt_ms
        theta_arr = theta_arr.at[start:end].set(float(thetas[i]))
        c_val = 0.0 if i == omit_index else contrast
        contrast_arr = contrast_arr.at[start:end].set(c_val)
        phase_arr = phase_arr.at[start:end].set(float(phases[i]))
        t_ms_arr = t_ms_arr.at[start:end].set(t_local)

    # ITI (blank) — theta=0, contrast=0, phase=0
    iti_start = n_elem * element_steps
    t_ms_arr = t_ms_arr.at[iti_start:].set(
        jnp.arange(iti_steps, dtype=jnp.float32) * dt_ms)

    return theta_arr, contrast_arr, phase_arr, t_ms_arr


# ---------------------------------------------------------------------------
# Sequence trial runner — closure-based JIT (same pattern as segment runners)
# ---------------------------------------------------------------------------

_sequence_runners: dict = {}


def _make_sequence_trial_runners(static, n_elem, element_steps, iti_steps):
    """Create JIT-compiled sequence trial runners for a specific config + trial shape.

    The ``static`` config is captured by the closures so its scalar fields
    remain concrete Python values during JAX tracing (same pattern as
    ``_make_segment_runners``).

    Parameters
    ----------
    static : StaticConfig (closed over)
    n_elem : int, number of sequence elements
    element_steps : int, timesteps per element
    iti_steps : int, timesteps for ITI
    """
    s = static  # closed over — concrete during tracing
    total_steps = n_elem * element_steps + iti_steps

    @jax.jit
    def run_trial_ee_plastic(state, theta_arr, contrast_arr, phase_arr, t_ms_arr,
                              step_keys, ee_A_plus_eff, ee_A_minus_eff):
        def scan_body(carry, inputs):
            st = carry
            t_ms, theta, contrast_val, phase_val, key = inputs
            st_new, v1_spk = timestep_phaseb_plastic(
                st, s, t_ms, theta, phase_val, contrast_val, key,
                ee_A_plus_eff, ee_A_minus_eff)
            return st_new, v1_spk

        final, v1_spks = jax.lax.scan(
            scan_body, state,
            (t_ms_arr, theta_arr, contrast_arr, phase_arr, step_keys))

        v1_counts = v1_spks.astype(jnp.int32).sum(axis=0)
        return final, v1_counts

    @jax.jit
    def run_trial_nonplastic(state, theta_arr, contrast_arr, phase_arr, t_ms_arr,
                              step_keys):
        def scan_body(carry, inputs):
            st = carry
            t_ms, theta, contrast_val, phase_val, key = inputs
            st_new, v1_spk = timestep_nonplastic(
                st, s, t_ms, theta, phase_val, contrast_val, key)
            return st_new, (v1_spk, st_new.g_exc_ee)

        final, (v1_spks, g_exc_ee_all) = jax.lax.scan(
            scan_body, state,
            (t_ms_arr, theta_arr, contrast_arr, phase_arr, step_keys))

        v1_counts = v1_spks.astype(jnp.int32).sum(axis=0)
        # Per-element counts
        elem_spks = v1_spks[:n_elem * element_steps].reshape(
            n_elem, element_steps, -1)
        element_counts = elem_spks.astype(jnp.int32).sum(axis=1)  # (n_elem, M)
        # Per-element conductance traces — shape (n_elem, element_steps, M)
        g_exc_ee_traces = g_exc_ee_all[:n_elem * element_steps].reshape(
            n_elem, element_steps, -1)

        return final, v1_counts, element_counts, g_exc_ee_traces

    return run_trial_ee_plastic, run_trial_nonplastic


def run_sequence_trial_jax(state, static, thetas, element_ms, iti_ms, contrast,
                            plastic_mode, step_keys=None, phases=None,
                            omit_index=-1, ee_A_plus_eff=None, ee_A_minus_eff=None):
    """Run one full sequence trial (elements + ITI) in JAX.

    Parameters
    ----------
    state : SimState
    static : StaticConfig
    thetas : list/array of float, orientations for each sequence element
    element_ms : float, duration of each element (ms)
    iti_ms : float, inter-trial interval duration (ms)
    contrast : float, stimulus contrast
    plastic_mode : str, 'ee' for Phase B training (E→E STDP only), 'none' for evaluation
    step_keys : optional (total_steps, 2) PRNGKey array
    phases : optional array of float, random phase per element
    omit_index : int, element index to omit (-1 = no omission)
    ee_A_plus_eff : float, effective LTP rate (required if plastic_mode='ee')
    ee_A_minus_eff : float, effective LTD rate (required if plastic_mode='ee')

    Returns
    -------
    (final_state, info_dict) where info_dict contains:
        'v1_counts': (M,) int32 total spike counts
        'element_counts': (n_elem, M) int32 per-element counts (only for 'none' mode)
        'g_exc_ee_traces': (n_elem, element_steps, M) float32 per-element conductance
            traces (only for 'none' mode) — continuous signal for omission response metric
    """
    thetas = list(thetas)
    n_elem = len(thetas)
    element_steps = int(round(element_ms / static.dt_ms))
    iti_steps = int(round(iti_ms / static.dt_ms))
    total_steps = n_elem * element_steps + iti_steps

    # Get / create cached JIT-compiled runners for this config + trial shape
    cache_key = (id(static), n_elem, element_steps, iti_steps)
    if cache_key not in _sequence_runners:
        _sequence_runners[cache_key] = _make_sequence_trial_runners(
            static, n_elem, element_steps, iti_steps)
    run_ee_plastic, run_nonplastic = _sequence_runners[cache_key]

    # Generate phases and keys if not provided
    key = state.rng_key
    if phases is None:
        key, *phase_keys = jax.random.split(key, n_elem + 1)
        phases = jnp.array([jax.random.uniform(pk, (), minval=0.0, maxval=2.0 * jnp.pi)
                            for pk in phase_keys])
    if step_keys is None:
        key, subkey = jax.random.split(key)
        step_keys = jax.random.split(subkey, total_steps)

    # Build stimulus arrays (outside JIT — concrete Python values)
    theta_arr, contrast_arr, phase_arr, t_ms_arr = build_trial_stimulus(
        thetas, n_elem, element_steps, iti_steps, contrast, phases,
        omit_index, static.dt_ms)

    if plastic_mode == 'ee':
        final, v1_counts = run_ee_plastic(
            state, theta_arr, contrast_arr, phase_arr, t_ms_arr, step_keys,
            ee_A_plus_eff, ee_A_minus_eff)
        final = final._replace(rng_key=key)
        return final, {"v1_counts": v1_counts}
    else:
        final, v1_counts, element_counts, g_exc_ee_traces = run_nonplastic(
            state, theta_arr, contrast_arr, phase_arr, t_ms_arr, step_keys)
        final = final._replace(rng_key=key)
        return final, {"v1_counts": v1_counts, "element_counts": element_counts,
                        "g_exc_ee_traces": g_exc_ee_traces}


def evaluate_omission_response(state, static, seq_thetas, element_ms, iti_ms,
                                contrast=1.0, n_eval_trials=10, omit_index=1):
    """Evaluate omission response using g_exc_ee conductance traces.

    Runs n_eval_trials for each of two conditions:
    - Trained context: seq_thetas with omit_index element omitted (contrast=0)
    - Control context: replace pre-omission element with novel orientation (22.5 deg)

    This follows Gavornik & Bear (2014): the omission response is the difference
    in neural activity during the blank (omitted) window when preceded by a trained
    vs novel context element.

    Parameters
    ----------
    state : SimState — network state (weights from training, dynamics will be reset)
    static : StaticConfig
    seq_thetas : list of float — trained sequence orientations (e.g. [0, 45, 90, 135])
    element_ms : float — duration of each element in ms
    iti_ms : float — inter-trial interval in ms
    contrast : float — stimulus contrast
    n_eval_trials : int — number of evaluation trials to average (default 10)
    omit_index : int — which element to omit (default 1)

    Returns
    -------
    dict with keys:
        'omr_conductance': float — mean g_exc_ee difference (trained - control)
        'omr_spikes': float — spike count difference (trained - control)
        'trained_g_mean': float — mean g_exc_ee in trained omission window
        'control_g_mean': float — mean g_exc_ee in control omission window
        'trained_spk_mean': float — mean spikes in trained omission window
        'control_spk_mean': float — mean spikes in control omission window
    """
    seq_thetas = list(seq_thetas)

    # Control condition: replace pre-omission element with novel orientation
    # 22.5 deg is not in {0, 45, 90, 135}, matching G&B (2014) design
    novel_theta = 22.5
    ctrl_thetas = seq_thetas.copy()
    pre_omit_idx = omit_index - 1 if omit_index > 0 else len(seq_thetas) - 1
    ctrl_thetas[pre_omit_idx] = novel_theta

    trained_g_vals = []
    control_g_vals = []
    trained_spk_vals = []
    control_spk_vals = []

    for trial in range(n_eval_trials):
        # Trained context omission trial
        st_eval = reset_state_jax(state, static)
        _, info_trained = run_sequence_trial_jax(
            st_eval, static, seq_thetas, element_ms, iti_ms, contrast,
            'none', omit_index=omit_index,
        )
        g_traces = np.array(info_trained['g_exc_ee_traces'])  # (n_elem, elem_steps, M)
        elem_counts = np.array(info_trained['element_counts'])  # (n_elem, M)

        # Mean g_exc_ee in omission window (averaged over timesteps and neurons)
        trained_g_vals.append(float(g_traces[omit_index].mean()))
        trained_spk_vals.append(float(elem_counts[omit_index].sum()))

        # Control context omission trial
        st_eval = reset_state_jax(state, static)
        _, info_ctrl = run_sequence_trial_jax(
            st_eval, static, ctrl_thetas, element_ms, iti_ms, contrast,
            'none', omit_index=omit_index,
        )
        g_traces_ctrl = np.array(info_ctrl['g_exc_ee_traces'])
        elem_counts_ctrl = np.array(info_ctrl['element_counts'])

        control_g_vals.append(float(g_traces_ctrl[omit_index].mean()))
        control_spk_vals.append(float(elem_counts_ctrl[omit_index].sum()))

    trained_g_mean = float(np.mean(trained_g_vals))
    control_g_mean = float(np.mean(control_g_vals))
    trained_spk_mean = float(np.mean(trained_spk_vals))
    control_spk_mean = float(np.mean(control_spk_vals))

    return {
        'omr_conductance': trained_g_mean - control_g_mean,
        'omr_spikes': trained_spk_mean - control_spk_mean,
        'trained_g_mean': trained_g_mean,
        'control_g_mean': control_g_mean,
        'trained_spk_mean': trained_spk_mean,
        'control_spk_mean': control_spk_mean,
    }


def segment_boundary_updates(state, static, v1_counts):
    """Apply slow plasticity at segment boundary.

    Ports _segment_boundary_updates() from the numpy code.
    """
    s = static
    W = state.W

    # --- Homeostatic synaptic scaling (disabled by default: rate=0) ---
    # Only computed if homeostasis_rate > 0
    error = s.target_rate_hz - state.rate_avg
    scale = 1.0 + s.homeostasis_rate * error
    lo = 1.0 - s.homeostasis_clip
    hi = 1.0 + s.homeostasis_clip
    scale = jnp.clip(scale, lo, hi)
    # Apply scaling only if homeostasis_rate > 0 (otherwise scale == 1.0, which is a no-op)
    W = W * scale[:, None]
    W = jnp.clip(W, 0.0, s.w_max)
    W = jnp.minimum(W, s.w_max * s.lgn_mask_e)
    W = W * s.tc_mask_e_f32

    # --- ON/OFF split constraint ---
    # Derive n_pix from array shape (concrete during tracing)
    n_pix = W.shape[1] // 2
    # Compute current ON/OFF sums
    sum_on = W[:, :n_pix].sum(axis=1)
    sum_off = W[:, n_pix:].sum(axis=1)

    err_on = (s.split_target_on - sum_on) / (s.split_target_on + 1e-12)
    err_off = (s.split_target_off - sum_off) / (s.split_target_off + 1e-12)

    scale_on = 1.0 + s.split_constraint_rate * err_on
    scale_off = 1.0 + s.split_constraint_rate * err_off
    lo_s = 1.0 - s.split_constraint_clip
    hi_s = 1.0 + s.split_constraint_clip
    scale_on = jnp.clip(scale_on, lo_s, hi_s)
    scale_off = jnp.clip(scale_off, lo_s, hi_s)

    W_on = W[:, :n_pix] * scale_on[:, None]
    W_off = W[:, n_pix:] * scale_off[:, None]
    W = jnp.concatenate([W_on, W_off], axis=1)

    W = jnp.clip(W, 0.0, s.w_max)
    W = jnp.minimum(W, s.w_max * s.lgn_mask_e)
    W = W * s.tc_mask_e_f32

    # --- Intrinsic bias homeostasis ---
    seg_rate_hz = v1_counts.astype(jnp.float32) / (s.segment_ms / 1000.0)
    I_v1_bias = state.I_v1_bias + s.v1_bias_eta * (s.target_rate_hz - seg_rate_hz)
    I_v1_bias = jnp.clip(I_v1_bias, -s.v1_bias_clip, s.v1_bias_clip)

    return state._replace(W=W, I_v1_bias=I_v1_bias)


# ---------------------------------------------------------------------------
# Segment runner — closure-based JIT to keep StaticConfig scalars concrete
# ---------------------------------------------------------------------------

_segment_runners: dict = {}


def _make_segment_runners(static):
    """Create JIT-compiled segment runners for a specific StaticConfig.

    The ``static`` config is captured by the closures so its scalar fields
    (steps, n_pix, L, …) remain concrete Python values during JAX tracing.
    Arrays in the closure become embedded constants in the compiled graph.
    """
    s = static  # closed over — concrete during tracing
    steps = int(s.steps)

    @jax.jit
    def run_nonplastic(state, theta_deg, contrast, phase, step_keys):
        # Reset drive accumulators
        state = state._replace(
            drive_acc_ff=jnp.zeros(s.M, dtype=jnp.float32),
            drive_acc_ee=jnp.zeros(s.M, dtype=jnp.float32),
            drive_acc_steps=jnp.int32(0),
        )

        def scan_body(carry, inputs):
            st = carry
            k, step_key = inputs
            t_ms = k * s.dt_ms
            st_new, v1_spk = timestep_nonplastic(st, s, t_ms, theta_deg, phase, contrast, step_key)
            return st_new, v1_spk

        final, v1_spks = jax.lax.scan(
            scan_body, state,
            (jnp.arange(steps, dtype=jnp.float32), step_keys))

        v1_counts = v1_spks.astype(jnp.int32).sum(axis=0)
        return final, v1_counts

    @jax.jit
    def run_plastic(state, theta_deg, contrast, phase, step_keys):
        # Reset drive accumulators
        state = state._replace(
            drive_acc_ff=jnp.zeros(s.M, dtype=jnp.float32),
            drive_acc_ee=jnp.zeros(s.M, dtype=jnp.float32),
            drive_acc_steps=jnp.int32(0),
        )

        def scan_body(carry, inputs):
            st = carry
            k, step_key = inputs
            t_ms = k * s.dt_ms
            st_new, v1_spk = timestep_plastic(st, s, t_ms, theta_deg, phase, contrast, step_key)
            return st_new, v1_spk

        final, v1_spks = jax.lax.scan(
            scan_body, state,
            (jnp.arange(steps, dtype=jnp.float32), step_keys))

        v1_counts = v1_spks.astype(jnp.int32).sum(axis=0)
        final = segment_boundary_updates(final, s, v1_counts)
        return final, v1_counts

    return (run_nonplastic, run_plastic)


def run_segment_jax(state, static, theta_deg, contrast, plastic):
    """Run one grating stimulus segment.

    Uses closure-based JIT: ``StaticConfig`` is closed over so its scalar
    fields (steps, n_pix, L, …) stay concrete during tracing.  The
    compiled kernels are cached per ``StaticConfig`` identity.

    Parameters
    ----------
    state : SimState
    static : StaticConfig
    theta_deg : float, grating orientation in degrees
    contrast : float, stimulus contrast
    plastic : bool

    Returns
    -------
    (new_state, v1_counts) where v1_counts is (M,) int32 spike counts
    """
    # Get / create cached JIT-compiled runners for this config
    sid = id(static)
    if sid not in _segment_runners:
        _segment_runners[sid] = _make_segment_runners(static)
    run_nonplastic, run_plastic = _segment_runners[sid]

    steps = int(static.steps)

    # Pre-split RNG keys (needs concrete ``steps`` for shape)
    key, phase_key = jax.random.split(state.rng_key)
    phase = jax.random.uniform(phase_key, (), minval=0.0, maxval=2.0 * jnp.pi)
    step_keys = jax.random.split(key, steps + 1)

    runner = run_plastic if plastic else run_nonplastic
    final_state, v1_counts = runner(
        state, theta_deg, contrast, phase, step_keys[:steps])

    # Advance rng_key for next segment
    final_state = final_state._replace(rng_key=step_keys[steps])
    return final_state, v1_counts


# ---------------------------------------------------------------------------
# Convenience: evaluate tuning (non-plastic multi-orientation)
# ---------------------------------------------------------------------------

def evaluate_tuning_jax(state, static, thetas_deg, repeats=1, contrast=1.0):
    """Evaluate orientation tuning without plasticity (no JIT on this wrapper).

    Parameters
    ----------
    state : SimState (will be saved/restored)
    static : StaticConfig
    thetas_deg : array-like of orientations to test
    repeats : int, number of repeats per orientation
    contrast : float

    Returns
    -------
    rates_hz : (M, K) firing rates in Hz
    """
    thetas = np.asarray(thetas_deg, dtype=np.float64)
    K = len(thetas)
    M = int(static.M)
    segment_ms = float(static.segment_ms)

    # Save state for non-destructive evaluation
    saved_state = state

    rates = np.zeros((M, K), dtype=np.float64)
    for rep in range(repeats):
        for ki, th in enumerate(thetas):
            _state_eval, v1_counts = run_segment_jax(saved_state, static,
                                                      float(th), float(contrast), False)
            counts_np = np.array(v1_counts, dtype=np.float64)
            rates[:, ki] += counts_np / (segment_ms / 1000.0)

    rates /= float(repeats)
    return rates.astype(np.float32)


def calibrate_ee_drive_jax(
    state: SimState,
    static: StaticConfig,
    target_frac: float = 0.15,
    reference_theta: float = 90.0,
    osi_floor: float = 0.30,
    contrast: float = 1.0,
    scales: list = None,
) -> Tuple[float, float]:
    """JAX-native binary-search W_e_e scale to hit target E→E drive fraction.

    Mirrors ``calibrate_ee_drive`` from biologically_plausible_v1_stdp.py but
    runs entirely on JAX (GPU-accelerated).  Uses ``run_segment_jax`` for
    evaluation probes and reads the drive accumulators from the returned state.

    The original state is not mutated — each measurement probe resets dynamic
    state (via ``reset_state_jax``) while preserving weights, matching the
    numpy version's save/restore pattern.

    Parameters
    ----------
    state : SimState
        Network state after Phase A (W_e_e should be the initial values —
        only feedforward STDP is applied during Phase A, so W_e_e is unchanged).
    static : StaticConfig
        Network configuration (scalar parameters, connectivity matrices).
    target_frac : float
        Desired fraction of excitatory drive from recurrent E→E connections.
    reference_theta : float
        Grating orientation for measurement probes (default 90.0°).
    osi_floor : float
        Reject scales that push mean OSI below this threshold (default 0.30).
    contrast : float
        Stimulus contrast for measurement probes (default 1.0).
    scales : list of float or None
        Override coarse sweep scales. Default covers 1–5000.

    Returns
    -------
    best_scale : float
        Scale factor for W_e_e that achieves the target drive fraction.
    best_frac : float
        Achieved drive fraction at best_scale.
    """
    from biologically_plausible_v1_stdp import compute_osi

    if scales is None:
        scales = [1.0, 5.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0]

    W_e_e_orig = state.W_e_e
    M = int(static.M)
    eye_M = jnp.eye(M, dtype=jnp.float32)

    # Use multiple orientations for robust drive fraction measurement
    # (avoids sensitivity to a single RNG-derived grating phase)
    probe_thetas = [0.0, 45.0, 90.0, 135.0]

    def _measure_drive_frac(W_e_e_scaled):
        """Run probes at multiple orientations and return mean drive fraction."""
        total_ff = 0.0
        total_ee = 0.0
        for i, theta in enumerate(probe_thetas):
            probe_key = jax.random.fold_in(state.rng_key, i)
            probe_state = reset_state_jax(
                state._replace(W_e_e=W_e_e_scaled, rng_key=probe_key), static)
            probe_after, _ = run_segment_jax(
                probe_state, static, theta, contrast, False)
            total_ff += float(jnp.sum(probe_after.drive_acc_ff))
            total_ee += float(jnp.sum(probe_after.drive_acc_ee))
        denom = total_ff + total_ee
        return total_ee / denom if denom > 0 else 0.0

    # --- Coarse sweep ---
    # Track ALL probed (scale, frac) pairs for OSI back-off
    all_probes = []
    for s in scales:
        W_scaled = W_e_e_orig * s * (1.0 - eye_M)
        frac = _measure_drive_frac(W_scaled)
        all_probes.append((s, frac))
        print(f"  [calibrate_jax] scale={s:.0f} → drive_frac={frac:.4f}")
        if frac > target_frac * 3:
            break  # no point going higher

    # Find bracket: last scale below target and first scale above
    lo_scale, lo_frac = 1.0, 0.0
    hi_scale, hi_frac = scales[-1], all_probes[-1][1]
    for s, f in all_probes:
        if f <= target_frac:
            lo_scale, lo_frac = s, f
        else:
            hi_scale, hi_frac = s, f
            break

    # --- Binary search refinement (8 iterations) ---
    for _ in range(8):
        mid_scale = math.sqrt(lo_scale * hi_scale)  # geometric midpoint
        if abs(hi_scale - lo_scale) / max(1e-8, hi_scale) < 0.05:
            break
        W_scaled = W_e_e_orig * mid_scale * (1.0 - eye_M)
        mid_frac = _measure_drive_frac(W_scaled)
        all_probes.append((mid_scale, mid_frac))
        print(f"  [calibrate_jax] refine scale={mid_scale:.1f} → "
              f"drive_frac={mid_frac:.4f}")
        if mid_frac < target_frac:
            lo_scale, lo_frac = mid_scale, mid_frac
        else:
            hi_scale, hi_frac = mid_scale, mid_frac

    # Pick the scale closest to target
    best_scale = (lo_scale
                  if abs(lo_frac - target_frac) < abs(hi_frac - target_frac)
                  else hi_scale)
    best_frac = lo_frac if best_scale == lo_scale else hi_frac

    # --- OSI safety check ---
    thetas_check = np.linspace(0, 180, 8, endpoint=False)

    def _check_osi(scale_val):
        W_test = W_e_e_orig * scale_val * (1.0 - eye_M)
        state_test = state._replace(W_e_e=W_test)
        rates_test = evaluate_tuning_jax(
            state_test, static, thetas_check, repeats=2, contrast=contrast)
        osi_test, _ = compute_osi(rates_test, thetas_check)
        return float(osi_test.mean())

    # Small tolerance for float32→float64 rounding in OSI comparison
    osi_tol = 1e-3

    osi_mean = _check_osi(best_scale)
    if osi_mean < osi_floor - osi_tol:
        print(f"  [calibrate_jax] WARNING: OSI={osi_mean:.3f} < "
              f"floor={osi_floor:.2f} at scale={best_scale:.1f}")
        # Sort all probed scales descending and try the highest that passes OSI
        all_probes_sorted = sorted(all_probes, key=lambda x: x[0], reverse=True)
        backed_off = False
        for s, f in all_probes_sorted:
            if s >= best_scale:
                continue
            osi_s = _check_osi(s)
            print(f"  [calibrate_jax] back-off scale={s:.1f} → "
                  f"OSI={osi_s:.3f}, drive_frac={f:.4f}")
            if osi_s >= osi_floor - osi_tol:
                best_scale = s
                best_frac = f
                osi_mean = osi_s
                backed_off = True
                break
        if not backed_off:
            print(f"  [calibrate_jax] WARNING: No scale passes OSI floor, "
                  f"using lowest probed scale")
            lowest = min(all_probes, key=lambda x: x[0])
            best_scale = lowest[0]
            best_frac = lowest[1]

    print(f"  [calibrate_jax] Final: scale={best_scale:.1f} → "
          f"drive_frac={best_frac:.4f}, OSI={osi_mean:.3f}")
    return best_scale, best_frac
