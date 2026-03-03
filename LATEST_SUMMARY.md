# V1 Spiking Network — Project Summary

**Last updated**: 2026-03-04
**Branch**: `fix-inhibition-biology`
**Validation**: 42/42 tests pass across 5 suites (biology, omission, multi-HC, inter-HC, multiconfig)

---

## What This Project Is

A biologically plausible spiking neural network that models orientation selectivity (OSI) formation in primary visual cortex (V1) through experience-dependent learning. The network implements the retina-to-cortex pathway (RGC → LGN → V1) with biologically grounded plasticity rules, and extends to sequence learning following the Gavornik & Bear (2014) protocol.

The project has two phases:
- **Phase A**: Orientation selectivity emerges via feedforward STDP (LGN → V1)
- **Phase B**: Temporal sequence learning via recurrent E→E STDP within V1

The network scales from a single hypercolumn (M=16, 16 neurons) to a retinotopic grid of 64 hypercolumns (n_hc=64, M=64, 4096 neurons) with biologically accurate inter-HC inhibition.

---

## File Map

| File | Lines | Purpose |
|------|-------|---------|
| `biologically_plausible_v1_stdp.py` | ~8,500 | Core numpy simulation — all mechanisms, `Params`, `RgcLgnV1Network` |
| `network_jax.py` | ~4,500 | JAX GPU port — 25x speedup, Phase A + Phase B, multi-HC vmap |
| `validate_inhibition_biology.py` | ~650 | 12 strict biological tests (SOM firing, SSI, cascade, sparsity, F>R, OMR) |
| `validate_omission_fix.py` | ~640 | Phase B omission response validation (5 tests) |
| `validate_multi_hc.py` | ~500 | Multi-HC structural tests (13 tests) |
| `validate_inter_hc_inhibition.py` | ~400 | Inter-HC surround suppression (8 tests) |
| `validate_l4_inhibition_multiconfig.py` | ~500 | Multi-config F>R + OMR across n_hc={1,4,16,64} (4×5 tests) |
| `validate_jax_port.py` | ~490 | JAX vs numpy correctness tests (15 tests) |
| `osi_investigation_harness.py` | ~670 | Standardized ablation/dose-response framework |
| `docs/l4_inhibition_research.md` | ~600 | L4 inhibitory circuit literature review (23+ citations) |
| `docs/recurrent_selectivity_research.md` | | Recurrent connectivity + selectivity mechanisms |
| `docs/multi_hc_research_log.md` | | Multi-hypercolumn biological justification |
| `investigation_results/` | | OSI mechanism investigation (JSON, npz, plots, reports) |

---

## Key APIs

### Numpy (`biologically_plausible_v1_stdp.py`)

```python
p = Params(M=16, N=8, seed=42, ...)           # ~300 configurable fields
net = RgcLgnV1Network(p)                        # Line 1202
net.run_segment(theta_deg, plastic, contrast)   # Line 2743 — one 300ms grating segment
net.evaluate_tuning(thetas, repeats, contrast)  # Line 3218 — non-destructive tuning eval
compute_osi(rates, thetas)                      # Line 115  — doubled-angle vector method
```

### JAX GPU (`network_jax.py`)

```python
state, static = numpy_net_to_jax_state(net)     # Convert numpy → JAX pytree
state, counts = run_segment_jax(state, static, theta_deg, contrast, plastic)  # Phase A
state, counts = run_sequence_trial_jax(          # Phase B
    state, static, thetas, element_ms, iti_ms, contrast,
    plastic_mode='ee',  # or 'none' for eval
    omit_index=-1,      # which element to omit (-1 = none)
    ee_A_plus_eff=0.005, ee_A_minus_eff=0.006,
    phases=jnp.array([0.0, 0.0, 0.0, 0.0]))    # Fixed spatial phases for multi-HC
scale, frac = calibrate_ee_drive_jax(state, static, target_frac=0.05, osi_floor=0.30)
state_pb, static_pb, _, _ = prepare_phaseb_ee(state, static, scale)  # Auto headroom
omr = evaluate_omission_response(state, static, thetas, elem_ms, iti_ms, ...)  # OMR eval
```

**Important**: Reuse the same `static` object across calls to avoid JIT recompilation (cache key is `id(static)`).

---

## Network Architecture

### Sensory Input Pipeline

**RGC (Retinal Ganglion Cells)**
- Difference-of-Gaussians center-surround filtering
- ON and OFF channels, independently sampled
- Position jitter (0.15) breaks lattice artifacts
- Firing: Poisson spikes at `base_rate + gain_rate * stimulus`

**LGN (Lateral Geniculate Nucleus)**
- Izhikevich thalamocortical (TC) neurons: a=0.02, b=0.25, c=-65, d=0.05
- Short-term depression (STP): u=0.05 depletion/spike, tau_rec=50ms
- N=8 patch → 2 × 8² = 128 LGN neurons (ON + OFF)

### V1 Excitatory Population

- M=16–64 ensembles per hypercolumn, Izhikevich regular-spiking (RS): a=0.02, b=0.2, c=-65, d=8
- LGN→E: dense with spatial envelope (sigma=2.0), 75% anatomical sparsity
- E→E: all-to-all (Phase B) with heterogeneous conduction delays (1–6ms)
- E→E short-term depression (STD): U=0.25 (Thomson & Lamy 2007: PPR=0.58)
- dt = 0.5ms (Izhikevich stability requirement)

### Inhibitory Interneurons

**PV (Parvalbumin, fast-spiking)**
- Feedforward (LGN→PV) + feedback (E→PV→E) circuit
- PV→PV mutual inhibition (w_pv_pv=0.5, Gouwens et al. 2019)
- Inhibitory STDP (iSTDP): homeostatic, targets ~8 Hz firing rate
- tau_gaba=8ms (Bacci et al. 2003)

**SOM (Somatostatin, low-threshold spiking)**
- E→SOM→E lateral inhibition within and between hypercolumns
- **w_e_som=0.3**, w_som_e=0.3 (Scala 2019: 21.1% connectivity)
- **som_bias=0.5**: tonic background current models in vivo thalamocortical + neuromodulatory input (Urban-Ciecko & Barth 2016), SOM fires ~3.3 Hz evoked
- SOM GABA: tau=15ms, dual-exponential kinetics (dendritic targeting, Kapfer et al. 2007)
- SOM→PV cross-inhibition (w_som_pv=0.3, Pfeffer et al. 2013)
- E→SOM facilitating STP: U=0.30, tau_fac=200ms (Silberberg & Markram 2007: PPR ~2.5)
- **phaseb_som_gain=0.5**: VIP/cholinergic gating during learning (Sarkar et al. 2024; Fu et al. 2014)

### Multi-Hypercolumn Architecture (n_hc > 1)

- Retinotopic grid of hypercolumns (tested up to n_hc=64, 8×8 grid, 4096 neurons)
- Each HC has independent LGN receptive field (rf_spacing_pix=1.0)
- Block-diagonal feedforward: each HC's E neurons only see own LGN
- Block-diagonal vmap: W_e_e_hc (n_hc, M_per_hc, M_per_hc) for efficient Phase B

**Inter-HC E→E connections:**
- Distance-dependent Gaussian decay (inter_hc_w_e_e=0.005)
- **Sparse** (inter_hc_ee_sparsity=0.15, Stettler et al. 2002; Bosking et al. 1997)
- Conduction delay: 8ms base + 4ms/HC distance (0.1–0.3 m/s unmyelinated)
- Population-specific STDP ceilings (w_e_e_max_intra, w_e_e_max_inter)

**Inter-HC surround suppression (E→E→SOM cascade):**
- **Biologically accurate pathway** (Scala 2019: 0% monosynaptic E→SOM in L4):
  E(remote) → pop rate → leaky integrator → delay → W_hc_lateral → E(local) → SOM(local) → E suppression
- Leaky integrator (NOT EMA): `y=(1-a)*y + x`, steady-state amplification ~40x
- SSI=0.36–0.65 (biology: 0.3–0.7, Adesnik et al. 2012)
- Onset ~22–42ms (biology: ~25ms)
- inter_hc_e_lateral_gain=0.002 scales lateral signal for modest E excitation

---

## Plasticity Rules

### 1. Triplet STDP (LGN→E, Phase A) — **ESSENTIAL**

From Pfister & Gerstner (2006). Drives orientation selectivity formation.

```
LTP: dW = +A2_plus * post_spike * x_pre * (w_max - W)    # pair term
         +A3_plus * post_spike * x_pre * x_post_slow      # triplet enhancement
LTD: dW = -A2_minus * pre_arrival * x_post * W            # pair term
```

Parameters: A2_plus=0.008, A2_minus=0.010, tau_plus=tau_minus=20ms, w_max=1.0

### 2. Heterosynaptic Depression — **ESSENTIAL ENABLER**

Resource-like competition: depresses inactive synapses on postsynaptic spike.

```
dW = -A_het * post_spike * (1 - arrivals) * W
```

A_het=0.032 (default). Prevents all-weights-saturate-to-w_max failure mode.

### 3. ON/OFF Split Competition — **IMPORTANT** (redundant with Het)

Cross-channel depression: ON activity depresses OFF synapses and vice versa. Strongly redundant with Het (interaction I = −0.653).

### 4. E→E Delay-Aware STDP (Phase B) — Sequence Learning

Weight-dependent pair-based STDP on recurrent excitatory connections.

```
LTP: dW = +A_plus * post_spike * pre_trace * (w_max - W)
LTD: dW = -A_minus * delayed_arrival * post_trace * (W - w_min)
```

A_plus=0.005, A_minus=0.006. Headroom: 3× for n_hc=1, 5× for n_hc>1.

**Self-regulating**: As W → w_max, LTP → 0. This creates a ceiling on F>R asymmetry. This is a property of the weight-dependent learning rule, not a bug.

---

## Phase B Protocol (Gavornik & Bear 2014)

### Experimental Design

| Parameter | Value | Reference |
|-----------|-------|-----------|
| Sequence | [0°, 45°, 90°, 135°] | 4-element orientation sequence |
| Element duration | 150ms | Matches G&B 2014 |
| ITI | 1500ms | Inter-trial interval (~1-2s in biology) |
| Presentations | 800 | ~200/day × 4 days |
| STDP mode | E→E only | Feedforward frozen |
| Spatial phases | Fixed [0,0,0,0] | Same physical gratings across trials |
| Omission metric | g_exc_ee conductance | Continuous, low-noise |

### Calibration Parameters (M-dependent)

| Config | target_frac | headroom | Rationale |
|--------|-------------|----------|-----------|
| n_hc=1, M=16 | 0.15 | 3× | Weak recurrent cascade (15 E→E connections) |
| n_hc=1, M=64 | 0.10 | 3× | Moderate cascade (63 connections), balances F>R + OMR |
| n_hc>1 | 0.05 | 5× | Strong cascade, preserve per-HC OSI |

### Pipeline

1. **Phase A** (100 segments, JAX): Feedforward STDP develops OSI → mean OSI ≈ 0.81
2. **Calibrate E→E**: Scale weights so target_frac of excitatory drive is recurrent, with OSI floor=0.30
   - Uses som_bias=0.0 in calibration probe (prevents SOM interference)
3. **Set w_e_e_max** = headroom × calibrated mean
4. **Record pre-calibration preferred orientations** (post-cal pref distorted by strong E→E)
5. **Phase B training**: 800 presentations with E→E STDP
6. **Evaluate**: F>R asymmetry + omission response (conductance-based)

### Results by Configuration

| Config | F>R | OMR | OSI (post-cal) | Time |
|--------|-----|-----|-----------------|------|
| n_hc=1, M=16 | **2.262** | +0.001112 | 0.886 | ~30s |
| n_hc=1, M=64 | **1.085** | +0.000208 | 0.762 | 2.3 min |
| n_hc=4, M=64 | **1.092** | +0.004960 | 0.791 | 3.5 min |
| n_hc=16, M=64 | **1.066** | +0.001877 | 0.783 | 4.0 min |
| n_hc=64, M=64 | **1.146** | +0.003317 | 0.805 | 4.9 min |

All F>R values are monotonically increasing over 800 presentations.

---

## OSI Mechanism Investigation Results

Systematic ablation study across 66+ conditions × 3 seeds (~184 simulation runs).
Full report: `investigation_results/synthesis/FINAL_REPORT.md`

### Ablation from Full Model (baseline OSI = 0.846)

| Rank | Mechanism | OSI after ablation | Delta | Role |
|------|-----------|-------------------|-------|------|
| 1 | Triplet STDP | 0.200 | −0.646 | **Essential** |
| 2 | ON/OFF Split | 0.522 | −0.324 | **Important** |
| 3 | Heterosynaptic | 0.745 | −0.101 | **Facilitating** |
| 4 | PV Inhibition | 0.824 | −0.022 | Modulatory |
| 5 | SOM Inhibition | 0.837 | −0.009 | Modulatory |
| 6 | TC STP | 0.844 | −0.002 | Neutral |

### Key Insight

**STDP + weight competition = necessary and sufficient for OSI.**
STDP alone → weights saturate → OSI 0.014. Add heterosynaptic depression → sharp selectivity (OSI 0.792).

---

## Performance

| Operation | n_hc=1 | n_hc=64, M=64 |
|-----------|--------|----------------|
| Phase A (100 seg) | 3.1s (31ms/seg) | 7.5s (75ms/seg) |
| Phase B (800 pres) | 121s (151ms/pres) | 257s (321ms/pres) |
| Full pipeline | ~2.3 min | ~4.9 min |

### Block-Diagonal vmap Optimization

The n_hc=64 M=64 (4096 neurons) Phase B was optimized from 93 min to 4.9 min via:
1. **E→E vmap**: W_e_e → W_e_e_hc (n_hc, M_per_hc, M_per_hc)
2. **PV vmap**: per_hc_pv_step vmapped over HC dim
3. **SOM vmap**: per_hc_som_step vmapped (when w_e_som > 0)

### JAX Architecture

- **SimState** (~72 fields): mutable NamedTuple — voltages, currents, weights, traces, delay buffers, RNG key
- **StaticConfig** (~145 fields): immutable NamedTuple — connectivity matrices, masks, delays, decay constants, all scalar parameters
- **JIT strategy**: Closure-based caching. StaticConfig captured by closure in `_make_segment_runners()` factory. Cache key = `id(static)`. Reuse same object to avoid recompilation.
- **Phase A RNG**: Must use numpy RNG (not JAX) to get well-distributed preferred orientations

---

## Validation Test Suites

### Biological Fidelity (`validate_inhibition_biology.py`) — 12/12 PASS

| Test | What it checks | Result |
|------|----------------|--------|
| 1. SOM firing rate | 1–40 Hz evoked (Ma et al. 2010) | 3.3 Hz |
| 2. Phase A OSI | > 0.50 | 0.886 |
| 3. F>R (n_hc=1, M=16) | > 1.50 (800 pres) | 2.262 |
| 4. VIP gating | phaseb_som_gain configurable, SOM active, F>R > 1.15 | PASS |
| 5. SSI | > 0.05 (Adesnik et al. 2012) | 0.364 |
| 6. SSI onset | 10–60 ms | 42.5 ms |
| 7. Sparse inter-HC E→E | < 30% connectivity (Stettler et al. 2002) | 15.0% |
| 8. Calibration integrity | OSI > 0.30, target_frac in range | PASS |
| 9. F>R regression (M=16) | > 1.50 | 2.262 |
| 10. F>R regression (n_hc=4) | median > 1.05 | 1.173 |
| 11. OMR | > 0 (positive omission response) | +0.001112 |
| 12. Performance | < 10 min total at n_hc=64 M=64 | est 5.2 min |

### Phase B Omission (`validate_omission_fix.py`) — 5/5 PASS

| Test | What it checks | Result |
|------|----------------|--------|
| 1. Trace recording | g_exc_ee shape, non-negative, omission gap | PASS |
| 2. F>R ratio | > 1.15, monotonically increasing | F>R = 1.811 |
| 3. Omission response | Conductance positive (trained > control) | +0.001112 |
| 4. Bio audit | Weight-dep STDP, no global renorm, local plasticity | PASS |
| 5. Benchmark | 800 pres in < 5 min on GPU | PASS |

### Multi-HC Structural (`validate_multi_hc.py`) — 13/13 PASS

Tests retinotopic grid layout, block-diagonal connectivity, inter-HC delays, per-HC OSI, E→E weight structure.

### Inter-HC Inhibition (`validate_inter_hc_inhibition.py`) — 8/8 PASS

| Test | Result |
|------|--------|
| SSI magnitude | 0.552 (biology: 0.3–0.7) |
| SSI onset | ~22 ms (biology: ~25 ms) |
| Ablation ratio | 2.86× (disabling inter-HC → more activity) |
| + F>R, OMR, OSI regressions | All PASS |

### Multi-Config (`validate_l4_inhibition_multiconfig.py`) — 4/4 PASS

Tests n_hc={1,4,16,64} × M=64. Each config: F>R > 1.05, F>R monotonic, OMR positive, post-cal OSI > 0.30.

---

## L4 Inhibitory Circuit — 9 Mechanisms

All enabled by default. Grounded in peer-reviewed literature (23+ citations in `docs/l4_inhibition_research.md`).

| # | Mechanism | Key Params | Citation |
|---|-----------|------------|----------|
| 1 | PV→PV mutual inhibition | w_pv_pv=0.5 | Gouwens et al. 2019 |
| 2 | Fast GABA kinetics | tau_gaba=8ms | Bacci et al. 2003 |
| 3 | Intra-HC SOM | w_e_som=0.3, w_som_e=0.3, som_bias=0.5 | Ma et al. 2010; Scala 2019 |
| 4 | SOM GABA (dual-exp) | tau=15ms | Kapfer et al. 2007 |
| 5 | SOM→PV cross-inhibition | w_som_pv=0.3 | Pfeffer et al. 2013 |
| 6 | Phase A E→E STDP | A+=0.005, A−=0.006 | Song et al. 2000 |
| 7 | E→E STD | U=0.25 | Thomson & Lamy 2007 |
| 8 | E→SOM facilitating STP | U=0.30, tau_fac=200ms | Silberberg & Markram 2007 |
| 9 | Inter-HC E→E→SOM cascade | gain=0.002 | Scala 2019; Adesnik et al. 2012 |

---

## Environment

```
Python:    3.9 (miniconda3/envs/habitat)
NumPy:     1.26.4
JAX:       GPU-enabled (CUDA)
SciPy:     1.8.1
Matplotlib: 3.9.2
```

---

## References

- Izhikevich (2003, 2007): Spiking neuron models
- Pfister & Gerstner (2006): Triplet STDP
- Bi & Poo (1998): Spike-timing dependent plasticity
- Song, Miller & Abbott (2000): Weight-dependent STDP
- Gavornik & Bear (2014): Learned spatiotemporal sequence recognition in V1
- Ma et al. (2010): SOM interneuron firing rates in visual cortex
- Scala et al. (2019): L4 connectivity — 0% monosynaptic E→SOM
- Adesnik et al. (2012): Surround suppression in V1 via lateral circuits
- Stettler et al. (2002): Sparse, patchy horizontal E→E connections
- Sarkar et al. (2024): M2 muscarinic/cholinergic gating of SOM during learning
- Urban-Ciecko & Barth (2016): SOM spontaneous activity in vivo
- Pfeffer et al. (2013): SOM→PV cross-inhibition
- Thomson & Lamy (2007): E→E short-term depression
- Silberberg & Markram (2007): E→SOM facilitating synapses
- Kapfer et al. (2007): SOM dendritic-targeting GABA kinetics
- Fu et al. (2014): VIP→SOM disinhibition during locomotion/attention
