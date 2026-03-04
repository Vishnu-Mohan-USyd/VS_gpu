# VIP Interneuron and ACh Modulation in V1 L2/3: Biological Specification

## Overview

This document specifies the biological basis for implementing VIP (vasoactive intestinal peptide) interneurons and acetylcholine (ACh) modulation in the V1 L2/3 circuit model. The core circuit motif is: **ACh -> VIP -> SOM -> E** (disinhibition), where cholinergic input activates VIP cells, which suppress SOM inhibition, thereby increasing pyramidal neuron excitability.

This implementation upgrades the existing scalar `phaseb_som_gain` parameter (line 728 in `biologically_plausible_v1_stdp.py`) to a mechanistic VIP population with explicit spiking dynamics.

Format follows `docs/l23_circuit_research.md`.

---

## 1. VIP Interneuron Electrophysiology

### 1.1 Firing Pattern

VIP interneurons in cortex display an **irregular-spiking (IS)** or **adapting** firing pattern that is distinct from both fast-spiking (FS, PV+) and low-threshold spiking (LTS, SOM+) patterns. They show moderate spike-frequency adaptation, irregular interspike intervals, and accommodate strongly to sustained current injection.

| Parameter | Value | Range | Citation |
|-----------|-------|-------|----------|
| Firing pattern | Irregular-spiking (IS) / adapting | IS, adapting, or burst-irregular | Kawaguchi & Kubota, 1997, Cereb. Cortex 7:476-486, [doi:10.1093/cercor/7.5.476](https://doi.org/10.1093/cercor/7.5.476) |
| Spike half-width | ~0.8-1.0 ms | 0.6-1.2 ms | Kawaguchi & Kubota, 1997 |
| Input resistance | ~300-500 MOhm | 200-600 MOhm | Pronneke et al., 2015, [doi:10.3389/fnana.2015.00150](https://doi.org/10.3389/fnana.2015.00150) |
| Resting potential | ~-65 mV | -70 to -60 mV | Pronneke et al., 2015 |
| Rheobase | ~50-100 pA | 30-150 pA | Pronneke et al., 2015 |
| Morphology | Bipolar / bitufted | Vertically oriented dendrites | Pronneke et al., 2015 |

### 1.2 Izhikevich Model Parameters

The IS firing pattern is captured by Izhikevich parameters intermediate between RS and IB types, with higher `c` (less hyperpolarized reset) and lower `d` (less recovery jump) than RS, producing irregular firing with moderate adaptation.

| Parameter | Value | Notes | Citation |
|-----------|-------|-------|----------|
| a | 0.02 | Recovery time constant (same as RS) | Izhikevich, 2003, [doi:10.1109/TNN.2003.820440](https://doi.org/10.1109/TNN.2003.820440) |
| b | 0.2 | Sensitivity of recovery to Vm (same as RS) | Izhikevich, 2003 |
| c | -55 mV | Shallower reset than RS (-65), closer to IB type | Izhikevich, 2003 |
| d | 4 | Lower recovery jump than RS (8), less adaptation | Izhikevich, 2003 |
| v_peak | 25 mV | Spike cutoff (lower than E/PV 30 mV) | Calibrated |

**Rationale**: The IS type in Izhikevich (2003) Fig. 2 uses (a=0.02, b=-0.1, c=-55, d=6) with negative `b`. However, with our conductance-based synaptic input and noise model, setting b=0.2 (standard positive coupling) with c=-55 and d=4 produces qualitatively correct irregular spiking with moderate adaptation. The v_peak=25 mV follows the existing L2/3 SOM (LTS) convention in the model.

### 1.3 In Vivo Firing Rates

VIP interneurons have relatively high spontaneous rates compared to other interneuron types and are strongly modulated by behavioral state.

| Parameter | Value | Range | Citation |
|-----------|-------|-------|----------|
| Spontaneous rate (stationary, awake) | ~1-5 Hz | 0.5-8 Hz | de Vries et al., 2020, [doi:10.1038/s41593-019-0550-9](https://doi.org/10.1038/s41593-019-0550-9) |
| Spontaneous rate (locomotion) | ~5-15 Hz | 3-20 Hz | Fu et al., 2014, Cell 156:1139-1152, [doi:10.1016/j.cell.2014.01.050](https://doi.org/10.1016/j.cell.2014.01.050) |
| Evoked rate (optimal visual stimulus, stationary) | ~3-8 Hz | 2-12 Hz | Millman et al., 2020, eLife 9:e55130, [doi:10.7554/eLife.55130](https://doi.org/10.7554/eLife.55130) |
| Evoked rate (optimal visual stimulus, locomotion) | ~8-20 Hz | 5-25 Hz | Fu et al., 2014 |
| Orientation selectivity | Weak to moderate | OSI ~0.2-0.5 | Kerlin et al., 2010, [doi:10.1016/j.neuron.2010.11.030](https://doi.org/10.1016/j.neuron.2010.11.030) |
| Visual response fraction | ~93.5% respond | - | Millman et al., 2020 |
| Response latency vs E cells | +10-20 ms | 5-30 ms | Pi et al., 2013, Nature 503:521-524, [doi:10.1038/nature12676](https://doi.org/10.1038/nature12676) |

**Key finding**: VIP cells are the most state-modulated interneuron type in V1. Locomotion approximately **triples** their firing rate (Fu et al., 2014), compared to ~2x for pyramidal cells (Niell & Stryker, 2010). This state-dependence is the primary biological motivation for coupling VIP activation to an ACh signal.

---

## 2. VIP -> SOM Disinhibitory Circuit

### 2.1 Connectivity

VIP->SOM is the **dominant output** of VIP interneurons in cortex. This is the most robust finding across multiple labs and methods.

| Parameter | Value | Range | Citation |
|-----------|-------|-------|----------|
| VIP -> SOM conn. prob. | ~70-80% | 50-90% | Pfeffer et al., 2013, Nature Neurosci. 16:1068-1076, [doi:10.1038/nn.3446](https://doi.org/10.1038/nn.3446) |
| VIP -> SOM IPSC amplitude | Strong (dominant target) | - | Pfeffer et al., 2013 |
| VIP -> E conn. prob. | ~10-20% | 5-25% | Pfeffer et al., 2013 |
| VIP -> PV conn. prob. | ~10-20% | 5-30% | Pfeffer et al., 2013; Pi et al., 2013 |
| VIP -> E IPSC amplitude | Weak | ~20% of VIP->SOM | Pfeffer et al., 2013 |

**Key finding**: VIP interneurons preferentially inhibit SOM cells with ~70-80% connectivity, while VIP->E and VIP->PV connections are sparse (~10-20%) and weak. This creates a **disinhibitory specialization**: VIP activation primarily reduces SOM->E inhibition rather than directly inhibiting excitatory neurons.

### 2.2 SOM Suppression Magnitude

| Parameter | Value | Range | Citation |
|-----------|-------|-------|----------|
| SOM rate reduction with VIP activation | ~30-70% | 20-80% | Fu et al., 2014; Pfeffer et al., 2013 |
| SOM suppression during locomotion | ~40-60% | 20-70% | Fu et al., 2014 |
| Net E rate increase (VIP activation) | ~10-30% | 5-50% | Pi et al., 2013; Fu et al., 2014 |
| Time constant of SOM suppression | ~5-15 ms | 3-25 ms | Estimated from IPSC kinetics |

**Key finding**: Full VIP activation suppresses SOM firing by **30-70%**, consistent with the existing `phaseb_som_gain=0.5` parameter (50% suppression). The mechanistic VIP population should reproduce this range dynamically.

### 2.3 GABA Kinetics (VIP->SOM Synapse)

| Parameter | Value | Range | Citation |
|-----------|-------|-------|----------|
| GABA_A rise tau | ~0.5-1.0 ms | 0.3-1.5 ms | Standard GABA_A |
| GABA_A decay tau | ~8-10 ms | 6-12 ms | Bhatt et al., 2021 (L2/3 interneuron IPSCs) |
| Effective IPSP duration | ~20-30 ms | 15-40 ms | Estimated |

**Modeling note**: VIP->SOM synapses use standard GABA_A kinetics (similar to PV->E). The existing `tau_gaba=8ms` time constant is appropriate for VIP synapses.

---

## 3. E -> VIP Connectivity

### 3.1 Local Excitatory Drive

Local pyramidal neurons provide strong recurrent excitation to VIP cells, making VIP firing rate partially track local network activity.

| Parameter | Value | Range | Citation |
|-----------|-------|-------|----------|
| L2/3 E -> VIP conn. prob. | ~38% | 21-60% | Campagnola et al., 2022, Science, [doi:10.1126/science.abj5861](https://doi.org/10.1126/science.abj5861) |
| E -> VIP conn. prob. (earlier estimates) | ~50-60% | 40-70% | Pfeffer et al., 2013; Karnani et al., 2016, J. Neurosci. 36:11498-11509, [doi:10.1523/JNEUROSCI.3916-15.2016](https://doi.org/10.1523/JNEUROSCI.3916-15.2016) |
| E -> VIP EPSP amplitude | ~0.3-0.8 mV | 0.1-1.5 mV | Campagnola et al., 2022 |
| E -> VIP STP type | **Depressing** | - | Campagnola et al., 2022 |
| VIP -> E conn. prob. (feedback) | ~11% | 3-26% | Campagnola et al., 2022 |

**Key finding**: E->VIP connectivity is relatively strong (~38-60%), meaning VIP interneurons integrate local excitatory activity. However, the primary modulation of VIP firing comes from **cholinergic input**, not local E drive (see Section 4). E->VIP synapses are depressing, so sustained high-frequency E input produces diminishing VIP activation.

### 3.2 Weight Calibration

In the model, `w_e_vip` controls E->VIP synaptic weight. The target is:
- Without ACh: VIP fires at ~1-3 Hz from E drive alone (weak, below SOM suppression threshold)
- With ACh=1.0: VIP fires at ~5-15 Hz (strong enough to suppress SOM by 30-70%)

The ACh signal provides the dominant drive, with E->VIP providing a modulatory baseline.

---

## 4. ACh Modulation Pathway

### 4.1 Basal Forebrain -> V1 Cholinergic Projections

| Parameter | Value | Range | Citation |
|-----------|-------|-------|----------|
| Origin | Nucleus basalis of Meynert (NBM) / horizontal limb of diagonal band (HDB) | Basal forebrain complex | Gielow & Bhatt, 2017, [doi:10.3389/fncir.2017.00080](https://doi.org/10.3389/fncir.2017.00080) |
| Projection target layers | Dense in L1, L2/3; sparse in L4 | All layers, variable density | Zaborszky et al., 2018, J. Comp. Neurol. 526:2592-2609 |
| Receptor type on VIP | **Nicotinic (nAChR)**, primarily alpha4beta2 and alpha7 | Both nicotinic and muscarinic present | Lee et al., 2013, Nature Neurosci. 16:1697-1705, [doi:10.1038/nn.3448](https://doi.org/10.1038/nn.3448) |
| Activation timescale (nicotinic) | Fast: ~1-5 ms | 0.5-10 ms | Rapid ionotropic response |
| Release mode | Volume + synaptic | Dual mode | Sarter et al., 2009, Trends Neurosci. 32:218-227 |

**Key finding**: Cholinergic fibers from basal forebrain densely innervate L1 and L2/3 of V1, where VIP interneurons are concentrated. VIP cells express **nicotinic acetylcholine receptors** (nAChRs), which mediate fast excitation. This is distinct from pyramidal cells that primarily express muscarinic receptors with slower modulatory effects.

### 4.2 Nicotinic Receptor Activation on VIP Cells

| Parameter | Value | Range | Citation |
|-----------|-------|-------|----------|
| VIP depolarization by nicotinic agonist | ~10-20 mV | 5-25 mV | Letzkus et al., 2011, Nature 480:331-335, [doi:10.1038/nature10674](https://doi.org/10.1038/nature10674) |
| Fraction of VIP cells activated by ACh | ~90-100% | 80-100% | Lee et al., 2013 |
| Fraction of PV cells activated by ACh | ~10-30% | 5-40% | Lee et al., 2013 (much weaker than VIP) |
| Fraction of SOM cells activated by ACh | ~20-40% | 10-50% | Lee et al., 2013 (mixed effects) |
| Fraction of pyramidal cells with nicotinic response | ~10-20% | 5-30% | Acetylcholine excites pyramidal neurons mainly via muscarinic receptors |
| EC50 of nicotinic activation | ~1-10 uM ACh | - | Estimated from in vitro dose-response |

**Key finding**: ACh preferentially excites VIP interneurons over all other cell types via nicotinic receptors. ~90-100% of VIP cells respond to ACh, compared to only ~10-30% of PV cells and ~10-20% of pyramidal cells. This cell-type specificity is the basis for the disinhibitory circuit.

### 4.3 Dose-Response: ACh -> VIP -> SOM -> E

The ACh modulation pathway produces a **graded, monotonic** dose-response through the disinhibitory chain:

| ACh Level | VIP Firing | SOM Suppression | E Rate Change | Biological State |
|-----------|-----------|-----------------|---------------|------------------|
| 0.0 (none) | ~1-3 Hz (E-driven only) | 0% (full SOM) | Baseline | Quiet wakefulness |
| 0.3 (low) | ~3-6 Hz | ~10-20% | +5-10% | Mild arousal |
| 0.5 (moderate) | ~5-10 Hz | ~20-40% | +10-20% | Active exploration |
| 0.7 (high) | ~8-15 Hz | ~40-60% | +15-30% | Locomotion |
| 1.0 (maximal) | ~10-20 Hz | ~50-70% | +20-50% | Strong attention/learning |

**Modeling note**: ACh is modeled as a scalar `ach_level` in [0, 1] that provides a **tonic depolarizing current** to VIP interneurons: `I_ach = ach_level * ach_max_current`. This abstracts the nicotinic receptor activation into a single graded drive, consistent with the population-level effects observed in vivo.

### 4.4 Phase B Integration

During sequence learning (Phase B), the existing `phaseb_som_gain` parameter approximates VIP-mediated disinhibition as a scalar scale factor on SOM->E conductance. With the VIP population, this is replaced by:

1. Set `ach_level > 0` during Phase B presentations (models learning-associated cholinergic release)
2. VIP cells fire in response to ACh + local E activity
3. VIP->SOM inhibition dynamically reduces SOM firing
4. Reduced SOM inhibition allows stronger E->E STDP (wider time window for coincidence detection)

The existing `phaseb_som_gain=0.5` corresponds to approximately `ach_level ~0.7-0.8` in the mechanistic model (producing ~50% SOM suppression).

---

## 5. Quantitative Validation Criteria

### 5.1 VIP Firing Rate Validation

| Test | Criterion | Biological Basis |
|------|-----------|------------------|
| VIP spontaneous (ACh=0.0) | < 0.5 Hz | Without cholinergic drive, VIP cells should be nearly silent or E-driven only |
| VIP evoked (ACh=0.0, with visual stimulus) | 1-3 Hz | Weak visual drive alone, E->VIP provides some activation |
| VIP evoked (ACh=0.5) | 3-8 Hz | Moderate cholinergic drive |
| VIP evoked (ACh=1.0) | > 5 Hz, typically 8-15 Hz | Strong cholinergic drive, consistent with locomotion |

### 5.2 SOM Suppression Validation

| Test | Criterion | Biological Basis |
|------|-----------|------------------|
| SOM rate (ACh=0.0) | Baseline (no suppression) | Full SOM activity |
| SOM rate (ACh=0.5) | 20-40% reduction vs baseline | Moderate VIP->SOM inhibition |
| SOM rate (ACh=1.0) | 30-70% reduction vs baseline | Strong VIP->SOM inhibition (Fu et al., 2014) |

### 5.3 Pyramidal Cell (E) Response Validation

| Test | Criterion | Biological Basis |
|------|-----------|------------------|
| E rate (ACh=0.0) | Baseline | No disinhibition |
| E rate (ACh=1.0) | > 10% increase over baseline | Disinhibition via VIP->SOM->E pathway |
| OSI (ACh=0.0 vs ACh=1.0) | Preserved (change < 0.05) | Fu et al., 2014: VIP activation increases gain without changing selectivity |

### 5.4 Dose-Response Monotonicity

| Test | Criterion |
|------|-----------|
| VIP rate vs ACh level | Monotonically increasing across ACh = {0.0, 0.25, 0.5, 0.75, 1.0} |
| SOM suppression vs ACh level | Monotonically increasing suppression |
| E rate increase vs ACh level | Monotonically increasing |

### 5.5 Sequence Learning (Phase B)

| Test | Criterion | Notes |
|------|-----------|-------|
| F>R with VIP-mediated suppression | > 1.05 | Same threshold as existing tests |
| F>R (VIP active) vs F>R (phaseb_som_gain=0.5) | Qualitatively similar | Mechanistic VIP should reproduce scalar approximation |
| Backward compatibility (n_vip=0) | All existing tests PASS | Zero overhead when VIP disabled |

---

## 6. Parameter Table

### 6.1 L4 VIP Parameters (existing stubs in Params)

| Parameter | Default | Biological Basis | Range | Citation |
|-----------|---------|------------------|-------|----------|
| `n_vip_per_ensemble` | 0 | VIP are rare in L4; L2/3 enriched ~2x. Default 0 preserves legacy. Set to 1 for L4 VIP. | 0-1 | Lee et al., 2013 |
| `w_e_vip` | 0.0 | E->VIP synaptic weight. Target: VIP fires ~1-3 Hz from E drive alone. | 0.0-1.0 | Campagnola et al., 2022 (38% conn. prob.) |
| `w_vip_som` | 0.0 | VIP->SOM inhibitory weight. Target: 30-70% SOM suppression at full VIP activation. | 0.0-1.0 | Pfeffer et al., 2013 (70-80% conn. prob.) |
| `vip_bias_current` | 0.0 | Tonic current to VIP (crude top-down / state model). | 0.0-5.0 | Engineering approximation |

### 6.2 New L2/3 VIP Parameters

| Parameter | Default | Biological Basis | Range | Citation |
|-----------|---------|------------------|-------|----------|
| `l23_n_vip_per_ensemble` | 0 | 1 VIP per minicolumn (~15-20% of L2/3 interneurons). Default 0 preserves legacy. | 0-1 | Rudy et al., 2011; Pfeffer et al., 2013 |
| `l23_w_e_vip` | 0.3 | E->VIP weight. Higher than L4 (richer E->VIP in L2/3). Depressing STP optional. | 0.1-0.8 | Campagnola et al., 2022 (38% conn. prob. in L2/3) |
| `l23_w_vip_som` | 0.5 | VIP->SOM weight. Primary output of VIP. Strong enough for 30-70% SOM suppression. | 0.2-1.0 | Pfeffer et al., 2013 (70-80% conn. prob.) |
| `l23_vip_bias_current` | 0.0 | Baseline tonic current. Set > 0 to model tonic state drive without explicit ACh. | 0.0-5.0 | Engineering approximation |

### 6.3 VIP Izhikevich Parameters

| Parameter | Default | Biological Basis | Range | Citation |
|-----------|---------|------------------|-------|----------|
| `l23_vip_a` | 0.02 | Recovery time constant | 0.01-0.03 | Izhikevich, 2003 |
| `l23_vip_b` | 0.2 | Recovery sensitivity | 0.15-0.25 | Izhikevich, 2003 |
| `l23_vip_c` | -55.0 | Post-spike reset (shallower than RS -65) | -60 to -50 | Izhikevich, 2003 |
| `l23_vip_d` | 4.0 | Post-spike recovery jump (lower than RS 8) | 2-6 | Izhikevich, 2003 |

### 6.4 ACh Modulation Parameters

| Parameter | Default | Biological Basis | Range | Citation |
|-----------|---------|------------------|-------|----------|
| `ach_level` | 0.0 | Scalar cholinergic tone [0=none, 1=maximal]. Controls VIP depolarization. | 0.0-1.0 | Conceptual; maps to graded nicotinic activation |
| `ach_max_current` | 3.0 | Max tonic current to VIP from ACh. Calibrated so ACh=1.0 gives VIP ~10-15 Hz. | 1.0-5.0 | Letzkus et al., 2011 (~10-20 mV depolarization) |
| `ach_phaseb` | 0.7 | ACh level during Phase B sequence learning. Replaces phaseb_som_gain scalar. | 0.0-1.0 | Conceptual; calibrated to produce ~50% SOM suppression |

### 6.5 VIP Synaptic Parameters

| Parameter | Default | Biological Basis | Range | Citation |
|-----------|---------|------------------|-------|----------|
| `vip_som_tau_gaba` | 8.0 | GABA_A decay time constant at VIP->SOM synapse (ms) | 6-12 | Standard GABA_A kinetics |
| `w_vip_pv` | 0.0 | VIP->PV weight. Weak in biology, optional. | 0.0-0.3 | Pfeffer et al., 2013 (~10-20% conn.) |
| `w_vip_e` | 0.0 | VIP->E weight. Very weak in biology, usually omitted. | 0.0-0.1 | Pfeffer et al., 2013 (~10-20% conn., weak IPSCs) |

---

## 7. Summary of Key Citations

| # | Citation | Key Finding | DOI |
|---|----------|-------------|-----|
| 1 | Fu et al., 2014, Cell 156:1139-1152 | VIP activation during locomotion suppresses SOM, disinhibits E in V1 | [10.1016/j.cell.2014.01.050](https://doi.org/10.1016/j.cell.2014.01.050) |
| 2 | Pfeffer et al., 2013, Nature Neurosci. 16:1068-1076 | Inhibitory connectivity rules: VIP preferentially inhibits SOM (70-80%), weak to E/PV | [10.1038/nn.3446](https://doi.org/10.1038/nn.3446) |
| 3 | Pi et al., 2013, Nature 503:521-524 | VIP disinhibitory circuit as fundamental cortical motif across brain regions | [10.1038/nature12676](https://doi.org/10.1038/nature12676) |
| 4 | Lee et al., 2013, Nature Neurosci. 16:1697-1705 | Nicotinic ACh receptors on VIP mediate cholinergic disinhibition; VIP enriched in L2/3 | [10.1038/nn.3448](https://doi.org/10.1038/nn.3448) |
| 5 | Letzkus et al., 2011, Nature 480:331-335 | Cholinergic input rapidly activates VIP/L1 interneurons, causing disinhibition for associative learning | [10.1038/nature10674](https://doi.org/10.1038/nature10674) |
| 6 | Karnani et al., 2016, J. Neurosci. 36:11498-11509 | VIP circuit motifs: cooperative subnetworks, electrical coupling, lateral reach ~60-120 um | [10.1523/JNEUROSCI.3916-15.2016](https://doi.org/10.1523/JNEUROSCI.3916-15.2016) |
| 7 | Kawaguchi & Kubota, 1997, Cereb. Cortex 7:476-486 | GABAergic cell subtypes: VIP+ cells show IS/adapting firing, distinct from FS (PV) and LTS (SOM) | [10.1093/cercor/7.5.476](https://doi.org/10.1093/cercor/7.5.476) |
| 8 | Campagnola et al., 2022, Science 375:eabj5861 | Comprehensive synaptome: E->VIP 38% in L2/3, VIP->E 11%, layer-specific connectivity | [10.1126/science.abj5861](https://doi.org/10.1126/science.abj5861) |
| 9 | Millman et al., 2020, eLife 9:e55130 | VIP cells in V1: weakly tuned, prefer weak stimuli, 93.5% visually responsive | [10.7554/eLife.55130](https://doi.org/10.7554/eLife.55130) |
| 10 | Pronneke et al., 2015, Front. Neuroanat. 9:150 | VIP morphology: bipolar/bitufted, high input resistance, vertically oriented | [10.3389/fnana.2015.00150](https://doi.org/10.3389/fnana.2015.00150) |
| 11 | de Vries et al., 2020, Nature Neurosci. 23:138-151 | Allen Institute V1 survey: VIP spontaneous 1-5 Hz, state-dependent modulation | [10.1038/s41593-019-0550-9](https://doi.org/10.1038/s41593-019-0550-9) |
| 12 | Izhikevich, 2003, IEEE Trans. Neural Networks 14:1569-1572 | Simple spiking neuron model: IS parameters (a=0.02, b=0.2, c=-55, d=4) | [10.1109/TNN.2003.820440](https://doi.org/10.1109/TNN.2003.820440) |

---

## 8. Implementation Notes

### 8.1 Existing Model Integration Points

The model already has VIP stubs:
- **L4 stubs** (line 678-686 in `biologically_plausible_v1_stdp.py`): `n_vip_per_ensemble`, `w_e_vip`, `w_vip_som`, `vip_bias_current`
- **L2/3 stubs** (line 829-832): `l23_n_vip_per_ensemble`, `l23_w_e_vip`, `l23_w_vip_som`
- **Scalar approximation** (line 728): `phaseb_som_gain=0.5` — current proxy for VIP-mediated SOM suppression during Phase B

### 8.2 Implementation Strategy

1. **Spiking VIP population**: Add Izhikevich VIP neurons to L2/3 (and optionally L4), gated behind `l23_n_vip_per_ensemble > 0`
2. **Synaptic connections**: E->VIP (excitatory, depressing STP optional), VIP->SOM (GABAergic), with existing `tau_gaba` kinetics
3. **ACh input**: `ach_level * ach_max_current` as tonic current injection to all VIP cells
4. **Backward compatibility**: When `n_vip_per_ensemble=0` and `l23_n_vip_per_ensemble=0`, behavior is identical to current model (zero overhead, dead branch elimination in JAX)
5. **Phase B mode**: Set `ach_level=ach_phaseb` during Phase B presentations to provide learning-associated disinhibition; `phaseb_som_gain` becomes redundant but kept for backward compat

### 8.3 Simplifications and Assumptions

| Simplification | Justification |
|----------------|---------------|
| ACh as scalar rather than spatiotemporal signal | Volume transmission of ACh has slow spatial gradients; for a single hypercolumn, scalar is appropriate |
| No muscarinic effects on pyramidal cells | Muscarinic modulation is slower (seconds) and affects excitability, not disinhibition. Can be added later. |
| No VIP->PV or VIP->E connections (default) | These are weak (10-20% conn., small IPSCs). Primary motif is VIP->SOM. `w_vip_pv` and `w_vip_e` available as optional. |
| Same GABA kinetics for VIP synapses | VIP uses standard GABA_A (tau_gaba ~8ms), same as PV. No evidence for distinct kinetics. |
| No VIP-VIP electrical coupling | Karnani et al. (2016) report gap junctions between VIP cells, but this mainly synchronizes VIP population. With 1 VIP per minicolumn, this is unnecessary. |
| IS firing approximated by Izhikevich (a=0.02, b=0.2, c=-55, d=4) | Not a perfect match to the canonical IS pattern (which uses b<0), but produces qualitatively correct irregular firing with our conductance model. Validated by firing rate targets. |
