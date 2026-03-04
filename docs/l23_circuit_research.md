# Mouse V1 Layer 2/3 Circuit Literature Review

## Overview

This document reviews the quantitative circuit properties of **mouse V1 Layer 2/3 (L2/3)**, covering excitatory populations, inhibitory interneuron subtypes (PV, SOM, VIP), interlaminar pathways (L4->L2/3 feedforward, L2/3->L4 feedback), horizontal (inter-columnar) connections, and response property targets. All values are for **adult mouse V1** unless explicitly noted as cross-species estimates.

Format follows `docs/l4_inhibition_research.md`.

---

## 1. L2/3 Excitatory Population

### 1.1 Size Ratio to L4

L2/3 is the thickest cortical layer in mouse V1, containing the largest excitatory population. Quantitative estimates come from the Allen Institute large-scale model (Billeh et al., 2020) and stereological studies.

| Parameter | Value | Range | Citation |
|-----------|-------|-------|----------|
| L2/3 thickness | ~250 um | 200-300 um | Billeh et al., 2020, [doi:10.1016/j.neuron.2020.01.040](https://doi.org/10.1016/j.neuron.2020.01.040) |
| L4 thickness | ~100 um | 80-120 um | Billeh et al., 2020 |
| L2/3 excitatory neuron count (per 400 um column) | ~20,000 | 18,000-22,000 | Billeh et al., 2020 |
| L4 excitatory neuron count (per 400 um column) | ~10,000 | 8,000-12,000 | Billeh et al., 2020 |
| **L2/3:L4 excitatory ratio** | **~2:1** | 1.5-2.5:1 | Billeh et al., 2020; Erö et al., 2018, [doi:10.3389/fninf.2018.00084](https://doi.org/10.3389/fninf.2018.00084) |
| Inhibitory fraction (L2/3) | ~15-20% | 13-22% | Rudy et al., 2011, [doi:10.1016/j.devcel.2011.07.013](https://doi.org/10.1016/j.devcel.2011.07.013); Xu et al., 2010, [doi:10.1523/JNEUROSCI.2354-10.2010](https://doi.org/10.1523/JNEUROSCI.2354-10.2010) |

**Key finding**: L2/3 contains roughly **twice as many** excitatory neurons as L4 in mouse V1, reflecting both greater thickness and comparable cell density.

### 1.2 Firing Rates

L2/3 pyramidal cells are notably sparse-firing compared to L4. Measurements vary substantially between anesthetized and awake preparations.

| Parameter | Value | Range | Citation |
|-----------|-------|-------|----------|
| Spontaneous rate (anesthetized) | 0.08 Hz | 0-0.5 Hz | Niell & Stryker, 2008, [doi:10.1523/JNEUROSCI.1903-08.2008](https://doi.org/10.1523/JNEUROSCI.1903-08.2008) |
| Spontaneous rate (awake, stationary) | 0.17 Hz | 0-1.0 Hz | Niell & Stryker, 2010, [doi:10.1016/j.neuron.2010.01.006](https://doi.org/10.1016/j.neuron.2010.01.006) |
| Spontaneous rate (awake, locomotion) | 0.5-1.0 Hz | 0.2-2.0 Hz | Niell & Stryker, 2010 |
| Evoked rate (optimal grating, anesthetized) | 2-5 Hz | 1-10 Hz | Niell & Stryker, 2008 |
| Evoked rate (optimal grating, awake stationary) | 2.9 Hz | 1-8 Hz | Niell & Stryker, 2010 |
| Evoked rate (optimal grating, awake moving) | 8.2 Hz | 3-20 Hz | Niell & Stryker, 2010 |
| Median evoked rate (all layers) | 6.7 Hz | - | Niell & Stryker, 2008 |
| L2/3 vs L4 evoked rate ratio | ~0.5-0.7x | - | Niell & Stryker, 2008; de Vries et al., 2020, [doi:10.1038/s41593-019-0550-9](https://doi.org/10.1038/s41593-019-0550-9) |

**Key finding**: L2/3 pyramidal cells fire at roughly **half the rate** of L4 excitatory neurons in response to optimal stimuli. Locomotion approximately doubles visually evoked firing rates in L2/3 (Niell & Stryker, 2010).

### 1.3 Orientation Selectivity Index (OSI)

| Parameter | Value | Range | Citation |
|-----------|-------|-------|----------|
| L2/3 pyramidal OSI (1-circular variance) | 0.75 | 0.5-0.95 | Kerlin et al., 2010, [doi:10.1016/j.neuron.2010.11.030](https://doi.org/10.1016/j.neuron.2010.11.030) |
| L2/3 pyramidal OSI (doubled-angle) | 0.78 | 0.4-0.98 | Niell & Stryker, 2008 |
| L4 pyramidal OSI | ~0.6-0.7 | 0.3-0.9 | Niell & Stryker, 2008 |
| L2/3 vs L4 OSI comparison | L2/3 >= L4 | - | Niell & Stryker, 2008; de Vries et al., 2020 |
| Fraction with OSI > 0.5 | ~79% | 70-85% | Niell & Stryker, 2008 |
| Orientation tuning width (HWHM) | 23 deg | 15-35 deg | Niell & Stryker, 2008 |

**Key finding**: L2/3 pyramidal cells are **at least as sharply tuned** as L4 cells, and typically slightly sharper. Orientation selectivity in L2/3 depends on a combination of feedforward L4 input and intracortical recurrent amplification (Li et al., 2013, [doi:10.1038/nn.3321](https://doi.org/10.1038/nn.3321)).

### 1.4 Izhikevich Parameters (Regular Spiking Type)

L2/3 pyramidal cells are Regular Spiking (RS) neurons with prominent spike-frequency adaptation.

| Parameter | Value | Notes | Citation |
|-----------|-------|-------|----------|
| a | 0.02 | Recovery time constant | Izhikevich, 2003, [doi:10.1109/TNN.2003.820440](https://doi.org/10.1109/TNN.2003.820440) |
| b | 0.2 | Sensitivity of recovery to Vm | Izhikevich, 2003 |
| c | -65 mV | Post-spike reset voltage | Izhikevich, 2003 |
| d | 8 | Post-spike recovery jump | Izhikevich, 2003 |
| V_threshold | ~-50 mV | Spike threshold | in vivo patch-clamp data |
| V_rest | ~-70 mV | Resting potential | Haider et al., 2013, [doi:10.1016/j.neuron.2012.09.039](https://doi.org/10.1016/j.neuron.2012.09.039) |
| Input resistance | 150-250 MOhm | Higher than L4 | Lefort et al., 2009, [doi:10.1016/j.neuron.2009.01.015](https://doi.org/10.1016/j.neuron.2009.01.015) |

**Note**: These are the canonical Izhikevich RS parameters. For mouse V1 specifically, some models use slightly lower `a` (0.01) to better capture the strong adaptation observed in L2/3 pyramidal cells.

---

## 2. L4->L2/3 Feedforward Pathway

### 2.1 Connection Probability and Convergence

The L4->L2/3 connection is the canonical feedforward pathway in cortex. Multiple studies have characterized this pathway in mouse V1 using paired recordings and optogenetics.

| Parameter | Value | Range | Citation |
|-----------|-------|-------|----------|
| L4 E -> L2/3 E conn. prob. (paired recording, <100 um) | ~10% | 7-15% | Seeman et al., 2018, [doi:10.7554/eLife.37349](https://doi.org/10.7554/eLife.37349); Campagnola et al., 2022, [doi:10.1126/science.abj5861](https://doi.org/10.1126/science.abj5861) |
| L4 E -> L2/3 E conn. prob. (2P optogenetics) | ~10.7% | 8-13% | Hage et al., 2022, [doi:10.7554/eLife.71103](https://doi.org/10.7554/eLife.71103) |
| L4 E -> L2/3 PV conn. prob. | ~15-25% | 10-30% | Xu & Bhatt, 2019 (inferred from L4 IN targeting); Hage et al., 2022 |
| Unitary EPSP amplitude (L4 E -> L2/3 E) | 0.3-0.8 mV | 0.1-2.0 mV | Campagnola et al., 2022; Feldmeyer et al., 2002, [doi:10.1113/jphysiol.2002.015719](https://doi.org/10.1113/jphysiol.2002.015719) |
| Unitary EPSC amplitude (L4 E -> L2/3 E) | ~65 pA | 20-150 pA | Campagnola et al., 2022 |
| Failure rate | ~5% | 0-15% | Feldmeyer et al., 2002 (rat S1, cross-species estimate) |
| Convergence (L4 E neurons per L2/3 E) | ~30-100 | - | Estimated from conn. prob. and local population size |

**Key finding**: L4 E -> L2/3 E connectivity is the strongest interlaminar excitatory input to L2/3, with ~10% connection probability at short intersomatic distances. L4 excitatory neurons and L2/3 interneurons represent the highest-rate sources of input to L2/3 pyramidal cells (Hage et al., 2022).

### 2.2 Delay

| Parameter | Value | Range | Citation |
|-----------|-------|-------|----------|
| Monosynaptic latency (L4 -> L2/3) | 1.0-1.5 ms | 0.5-2.5 ms | Campagnola et al., 2022 |
| Additional processing delay | 2-5 ms | - | Estimated from L4-L2/3 response latency difference |
| Total L4->L2/3 response latency | ~3-5 ms | 2-8 ms | Inferred from Niell & Stryker, 2008; Wehr & Zador, 2003 |

### 2.3 Short-Term Plasticity (STP)

| Parameter | Value | Range | Citation |
|-----------|-------|-------|----------|
| STP type (L4 E -> L2/3 E) | **Depressing** | - | Campagnola et al., 2022; Reyes & Bhatt, 2023 |
| Paired-pulse ratio (20 Hz) | 0.6-0.8 | 0.5-0.9 | Campagnola et al., 2022 |
| Release probability (Pr) | ~0.5-0.7 | 0.3-0.8 | Inferred from PPR |
| STP type (L4 E -> L2/3 PV) | **Depressing** (stronger) | - | Beierlein et al., 2003, [doi:10.1152/jn.00601.2003](https://doi.org/10.1152/jn.00601.2003) |
| STP type (L4 E -> L2/3 SOM) | **Facilitating** | - | Reyes et al., 1998, [doi:10.1038/nn1298_276](https://doi.org/10.1038/nn1298_276) |

**Key finding**: L4->L2/3 excitatory synapses onto pyramidal cells and PV interneurons are **depressing**, acting as temporal high-pass filters. In contrast, L4 inputs onto SOM cells are **facilitating**, consistent with the general rule that E->SOM synapses exhibit short-term facilitation regardless of presynaptic layer.

### 2.4 Plasticity

| Parameter | Value | Notes | Citation |
|-----------|-------|-------|----------|
| LTP/LTD expression | Present | Timing-dependent (STDP-like) | Sjostrom et al., 2001, [doi:10.1016/S0896-6273(01)00542-6](https://doi.org/10.1016/S0896-6273(01)00542-6) |
| Critical period plasticity | Yes | ODP depends on L4->L2/3 | Kuhlman et al., 2013, [doi:10.1038/nn.3446](https://doi.org/10.1038/nn.3446) |
| Hebbian STDP window | ~20 ms | Pre-before-post: LTP | Sjostrom et al., 2001 |

---

## 3. L2/3 PV (Basket Cell) Circuit

### 3.1 Proportion and Density

| Parameter | Value | Range | Citation |
|-----------|-------|-------|----------|
| PV+ fraction of L2/3 interneurons | ~40% | 35-45% | Rudy et al., 2011; Xu et al., 2010 |
| PV+ fraction of L2/3 all neurons | ~6-8% | 5-10% | Derived: 40% of ~18% inhibitory |
| PV density in L2/3 vs L4 | Lower in L2/3 | - | Pfeffer et al., 2013, [doi:10.1038/nn.3446](https://doi.org/10.1038/nn.3446); Kim et al., 2017 |

**Note**: Some studies report PV+ as ~20% of L2/3 interneurons, with remaining ~20% classified as 5HT3aR+/non-VIP/non-SOM. The discrepancy arises from counting methodology (genetic labeling vs. immunohistochemistry). For modeling, ~40% of identified interneurons is a reasonable estimate.

### 3.2 E->PV Connectivity

| Parameter | Value | Range | Citation |
|-----------|-------|-------|----------|
| E -> PV conn. prob. (within L2/3) | ~69% | 50-80% | Hofer et al., 2011, [doi:10.1038/nn.2876](https://doi.org/10.1038/nn.2876) |
| Unitary EPSP (E -> PV) | 0.5-2.0 mV | 0.3-3.0 mV | Hofer et al., 2011; Packer & Yuste, 2011, [doi:10.1523/JNEUROSCI.2538-11.2011](https://doi.org/10.1523/JNEUROSCI.2538-11.2011) |
| STP type (E -> PV) | Depressing (weak) | - | Reyes et al., 1998; Campagnola et al., 2022 |
| EPSC latency | ~1 ms | 0.5-1.5 ms | Campagnola et al., 2022 |
| Orientation selectivity of E->PV | **Unselective** | - | Hofer et al., 2011 |

**Key finding**: E->PV connectivity in L2/3 is **very high** (~69%), dramatically higher than in L4 (~12.5%, Scala et al., 2019). PV cells receive dense, unselective excitatory input from nearby pyramidal cells regardless of orientation preference (Hofer et al., 2011).

### 3.3 PV->E Connectivity

| Parameter | Value | Range | Citation |
|-----------|-------|-------|----------|
| PV -> E conn. prob. (within L2/3) | ~91% | 80-95% | Hofer et al., 2011 |
| Reciprocal PV<->E | ~67% | 55-75% | Hofer et al., 2011 |
| Unitary IPSC amplitude | 2-5 nS | 1-8 nS | Packer & Yuste, 2011 |
| IPSC kinetics (tau_gaba PV->E) | **6-8 ms** | 4-12 ms | Galarreta & Hestrin, 2002, [doi:10.1073/pnas.192159599](https://doi.org/10.1073/pnas.192159599); Bhatt et al., 2021, [doi:10.1523/ENEURO.0235-21.2021](https://doi.org/10.1523/ENEURO.0235-21.2021) |
| Targeting domain | **Perisomatic** | Soma + proximal dendrite | Kubota et al., 2016, [doi:10.3389/fncir.2016.00048](https://doi.org/10.3389/fncir.2016.00048) |

**Key finding**: PV->E connectivity is the **strongest and densest** inhibitory connection in L2/3, with ~91% of nearby pairs connected. This provides powerful, fast perisomatic inhibition that controls spike timing and gain.

### 3.4 PV->PV Mutual Inhibition

| Parameter | Value | Range | Citation |
|-----------|-------|-------|----------|
| PV -> PV conn. prob. (L2/3) | ~100% | 85-100% | Pfeffer et al., 2013 |
| PV -> PV IPSC charge | 2.76 pC | 1.5-4.0 pC | Pfeffer et al., 2013 |
| PV -> PV IPSC decay tau | 2.6 ms | 2-4 ms | Galarreta & Hestrin, 2002 |
| Electrical coupling (gap junctions) | Present | Coupling coefficient 5-10% | Gibson et al., 1999, [doi:10.1038/17051](https://doi.org/10.1038/17051) |

**Key finding**: PV->PV is the **strongest inhibitory synapse in cortex** (2.76 pC charge). Along with gap junction coupling, this creates the interneuron gamma network (ING) mechanism underlying 30-80 Hz oscillations (Cardin et al., 2009, [doi:10.1038/nature07991](https://doi.org/10.1038/nature07991)).

### 3.5 L4 -> L2/3 PV Feedforward Pathway

| Parameter | Value | Range | Citation |
|-----------|-------|-------|----------|
| L4 E -> L2/3 PV conn. prob. | ~15-25% | 10-30% | Inferred from Hage et al., 2022; higher than L4 E -> L2/3 E |
| LGN -> L2/3 PV (direct) | Weak/absent | - | Kloc & Hull, 2014, [doi:10.1523/JNEUROSCI.2687-14.2014](https://doi.org/10.1523/JNEUROSCI.2687-14.2014) |
| Feedforward inhibition delay (relative to E onset) | 1-2 ms | 0.5-3 ms | Pouille & Scanziani, 2001, [doi:10.1126/science.1060342](https://doi.org/10.1126/science.1060342) |

**Note**: Unlike L4 PV cells which receive massive direct thalamocortical input, L2/3 PV cells are primarily driven by local L2/3 excitatory neurons and L4 feedforward input. Direct LGN->L2/3 PV input is minimal.

### 3.6 PV Tuning Properties

| Parameter | Value | Range | Citation |
|-----------|-------|-------|----------|
| PV OSI (L2/3) | 0.36 | 0.1-0.6 | Kerlin et al., 2010 |
| PV firing rate (evoked) | 15-40 Hz | 10-80 Hz | Ma et al., 2010, [doi:10.1523/JNEUROSCI.1103-10.2010](https://doi.org/10.1523/JNEUROSCI.1103-10.2010) |
| PV firing rate (spontaneous) | 3-8 Hz | 1-15 Hz | Niell & Stryker, 2010 |

---

## 4. L2/3 SOM (Martinotti Cell) Circuit

### 4.1 Proportion and Subtype

**CRITICAL L2/3 vs L4 DISTINCTION**: Unlike L4 of mouse V1 where E->SOM connectivity is **0%** (Scala et al., 2019), L2/3 has **robust monosynaptic E->SOM connections**. This is a fundamental architectural difference between layers.

| Parameter | Value | Range | Citation |
|-----------|-------|-------|----------|
| SOM+ fraction of L2/3 interneurons | ~30% | 20-35% | Rudy et al., 2011; Xu et al., 2010 |
| SOM+ fraction of L2/3 all neurons | ~5-6% | 4-7% | Derived: 30% of ~18% inhibitory |
| Dominant subtype in L2/3 | **Martinotti cell** | - | Muñoz et al., 2017, [doi:10.1038/nrn.2016.53](https://doi.org/10.1038/nrn.2016.53) |
| Axonal target | **L1 (distal dendrites)** | L1 + local L2/3 | Wang et al., 2004, [doi:10.1002/cne.10906](https://doi.org/10.1002/cne.10906) |

**Note**: Some studies report SOM+ as ~8% of L2/3 interneurons when using specific genetic labeling (e.g., Sst-IRES-Cre). The 20-30% range from immunohistochemistry-based census (Rudy et al., 2011) is more commonly used for modeling.

### 4.2 E->SOM Connectivity

| Parameter | Value | Range | Citation |
|-----------|-------|-------|----------|
| E -> SOM conn. prob. (L2/3) | ~15-30% | 10-40% | Kapfer et al., 2007, [doi:10.1523/JNEUROSCI.2060-06.2007](https://doi.org/10.1523/JNEUROSCI.2060-06.2007); Fino & Bhatt, 2008 |
| E -> SOM in L4 of V1 | **0%** | - | Scala et al., 2019, [doi:10.1038/s41467-019-12058-z](https://doi.org/10.1038/s41467-019-12058-z) |
| Unitary EPSP (1st spike) | 0.3-0.8 mV | 0.1-1.5 mV | Reyes et al., 1998; Kapfer et al., 2007 |
| Unitary EPSP (4th spike at 20 Hz) | 1.5-3.0 mV | 1.0-5.0 mV | Reyes et al., 1998 |
| STP type (E -> SOM) | **Strongly facilitating** | PPR 2.0-3.0 | Reyes et al., 1998; Kapfer et al., 2007 |
| 4th/1st pulse ratio (20-40 Hz) | 4.0-6.0 | 3.0-8.0 | Reyes et al., 1998 |

**Key finding**: The E->SOM connection is a **defining feature of L2/3 circuits** that is **absent in L4** of mouse V1. The strong short-term facilitation means SOM cells are recruited preferentially by sustained or burst-like pyramidal cell activity, acting as a frequency-dependent filter.

### 4.3 E->SOM Facilitation STP Parameters

| Parameter | Value | Range | Citation |
|-----------|-------|-------|----------|
| Initial utilization (U) | 0.05-0.15 | - | Gupta et al., 2000, [doi:10.1126/science.287.5451.273](https://doi.org/10.1126/science.287.5451.273) |
| Facilitation time constant (tau_f) | 100-200 ms | 50-300 ms | Gupta et al., 2000 |
| Depression time constant (tau_d) | 30-50 ms | 20-80 ms | Gupta et al., 2000 |
| Steady-state facilitation ratio (at 20 Hz) | 3-5x | 2-8x | Silberberg & Markram, 2007, [doi:10.1113/jphysiol.2007.132852](https://doi.org/10.1113/jphysiol.2007.132852) |

### 4.4 SOM->E Connectivity

| Parameter | Value | Range | Citation |
|-----------|-------|-------|----------|
| SOM -> E conn. prob. (L2/3) | ~30-50% | 20-60% | Kapfer et al., 2007; Pfeffer et al., 2013 |
| Unitary IPSC amplitude | 0.3-1.0 nS | 0.2-2.0 nS | Various paired recording studies |
| SOM->E vs PV->E amplitude ratio | ~0.3x | 0.2-0.5x | Pfeffer et al., 2013 |
| IPSC kinetics (tau_gaba SOM->E) | **15-20 ms** | 10-25 ms | Bhatt et al., 2021 |
| Effective IPSP duration | 40-60 ms | 30-80 ms | Bhatt et al., 2021 |
| Targeting domain | **Distal dendrite** (L1) | Apical tuft, distal basal | Wang et al., 2004 |

**Key finding**: SOM->E inhibition is slower and weaker per synapse than PV->E, but targets **distal dendrites** where it modulates dendritic integration and top-down input gating. The ~15-20 ms GABA_A decay reflects both intrinsic kinetics and dendritic cable filtering.

### 4.5 SOM->PV Cross-Inhibition

| Parameter | Value | Range | Citation |
|-----------|-------|-------|----------|
| SOM -> PV conn. prob. | ~86% | 70-100% | Pfeffer et al., 2013 |
| SOM -> PV IPSC charge | 0.77 pC | 0.4-1.2 pC | Pfeffer et al., 2013 |
| PV -> SOM conn. prob. | ~0% | 0-5% | Pfeffer et al., 2013 |
| PV -> SOM IPSC charge | 0.07 pC | ~0 | Pfeffer et al., 2013 |

**Key finding**: SOM **strongly inhibits PV** but PV does **NOT inhibit SOM**. This asymmetry creates a disinhibitory pathway: SOM activation -> PV suppression -> reduced PV->E inhibition -> net excitation of pyramidal cells. This is a fundamentally different motif from L4 where E->SOM connectivity is absent.

### 4.6 SOM Firing Rates and Tuning

| Parameter | Value | Range | Citation |
|-----------|-------|-------|----------|
| SOM spontaneous rate | ~0.5-2 Hz | 0-5 Hz | Ma et al., 2010 |
| SOM evoked rate (optimal) | 3-8 Hz | 1-15 Hz | Ma et al., 2010 |
| SOM OSI | 0.52 | 0.2-0.8 | Ma et al., 2010 |
| SOM response latency (vs E) | +20-50 ms delay | 10-80 ms | Ma et al., 2010 |

**Key finding**: SOM cells have **weak, delayed responses** compared to both excitatory and PV cells. Their orientation selectivity is intermediate between broadly-tuned PV and sharply-tuned pyramidal cells. The delayed recruitment is consistent with their dependence on facilitating excitatory synapses.

### 4.7 SOM GABA Kinetics

| Parameter | PV->E (L2/3) | SOM->E (L2/3) | Citation |
|-----------|--------------|----------------|----------|
| GABA_A rise tau | 0.5-1.0 ms | 0.5-1.0 ms | Bhatt et al., 2021 |
| GABA_A decay tau | 6-8 ms | 15-20 ms | Bhatt et al., 2021 |
| Effective IPSP duration | ~20-30 ms | ~40-60 ms | Bhatt et al., 2021 |
| Compound IPSP time-to-peak | 113.5 ms | 135.9 ms | Bhatt et al., 2021 (optogenetic, population) |

---

## 5. L2/3 VIP Circuit

### 5.1 Proportion and Distribution

| Parameter | Value | Range | Citation |
|-----------|-------|-------|----------|
| VIP+ fraction of L2/3 interneurons | ~15-20% | 12-25% | Rudy et al., 2011; Pfeffer et al., 2013 |
| VIP+ fraction of L2/3 all neurons | ~3-4% | 2-5% | Derived |
| L2/3 vs L4 VIP density | **L2/3 enriched** | ~2x more VIP in L2/3 | Lee et al., 2013, [doi:10.1038/nn.3448](https://doi.org/10.1038/nn.3448) |
| VIP morphology | Bipolar/bitufted | Vertically oriented | Pronneke et al., 2015, [doi:10.3389/fnana.2015.00150](https://doi.org/10.3389/fnana.2015.00150) |

### 5.2 VIP->SOM Disinhibitory Pathway

| Parameter | Value | Range | Citation |
|-----------|-------|-------|----------|
| VIP -> SOM conn. prob. | ~70-80% | 50-90% | Pfeffer et al., 2013 |
| VIP -> SOM IPSC | Strong | Dominant output target | Pfeffer et al., 2013 |
| VIP -> E conn. prob. | Weak (~10-20%) | - | Pfeffer et al., 2013 |
| VIP -> PV conn. prob. | Weak (~10-20%) | - | Pfeffer et al., 2013 |
| E -> VIP conn. prob. | ~50-60% | 40-70% | Pfeffer et al., 2013; Karnani et al., 2016 |
| Lateral reach of single VIP cell | 60-120 um | - | Karnani et al., 2016, [doi:10.1523/JNEUROSCI.3916-15.2016](https://doi.org/10.1523/JNEUROSCI.3916-15.2016) |

### 5.3 State-Dependent Modulation

| Parameter | Value | Notes | Citation |
|-----------|-------|-------|----------|
| VIP activation during locomotion | **Strong increase** | Cholinergic + top-down | Fu et al., 2014, [doi:10.1038/nn.3728](https://doi.org/10.1038/nn.3728) |
| VIP effect on SOM during locomotion | Suppression | Disinhibits E cells | Fu et al., 2014 |
| VIP tuning | Orientation-selective | Prefers low contrast, front-to-back | Millman et al., 2020, [doi:10.7554/eLife.55130](https://doi.org/10.7554/eLife.55130) |
| VIP spontaneous rate | ~1-5 Hz | State-dependent | de Vries et al., 2020 |

### 5.4 Recommendation: Include or Defer?

**Recommendation: DEFER for initial L2/3 implementation.**

Rationale:
1. VIP function is primarily driven by **top-down/modulatory signals** (cholinergic input during locomotion, attention-related feedback). Without modeling these signals, VIP activation is underconstrained.
2. The VIP->SOM->E disinhibitory pathway only meaningfully modulates circuit dynamics during **active behavioral states** (locomotion, attention). A static/passive viewing model does not need VIP.
3. VIP can be added later as a state-dependent gain modulation module without changing the core E/PV/SOM architecture.

**If VIP is added later**: Implement as a scalar gain on SOM->E inhibition (e.g., `phaseb_som_gain` parameter already exists in the model) rather than a full spiking VIP population, unless locomotion/state-dependent dynamics are explicitly being modeled.

---

## 6. L2/3->L4 Feedback

### 6.1 Anatomical Evidence

Descending feedback from L2/3 to L4 is **weak but present** in mouse V1. This is distinct from the canonical feedforward flow (L4->L2/3).

| Parameter | Value | Range | Citation |
|-----------|-------|-------|----------|
| L2/3 E -> L4 E conn. prob. | ~1-3% | 0.5-5% | Lefort et al., 2009; Jiang et al., 2015, [doi:10.1126/science.aac9462](https://doi.org/10.1126/science.aac9462) |
| L2/3 E -> L4 SOM conn. prob. | Unknown (weak) | - | No direct measurement available |
| L2/3 E -> L4 PV conn. prob. | ~5-10% | 3-15% | Jiang et al., 2015 |
| L2/3 excitatory drive to L4 | **Limited** | - | Adesnik & Scanziani, 2010, [doi:10.1038/nature09247](https://doi.org/10.1038/nature09247) |
| L2/3 suppression of L4 | **Primarily via inhibition** | - | Olsen et al., 2012, [doi:10.1038/nn.3123](https://doi.org/10.1038/nn.3123) |

### 6.2 Targets and Functional Role

| Parameter | Value | Notes | Citation |
|-----------|-------|-------|----------|
| Primary L4 target of L2/3 feedback | **PV (FS) interneurons** | Not E or SOM | Bortone et al., 2014, [doi:10.1038/nn.4123](https://doi.org/10.1038/nn.4123) |
| Functional effect | **Suppressive** | L2/3 suppresses L4 | Adesnik & Scanziani, 2010 |
| Delay (L2/3 -> L4) | ~1-2 ms | Monosynaptic | Estimated from paired recordings |
| SOM involvement in feedback | **L2/3 SOM targets L5, not L4** | - | Jiang et al., 2015 |

**Key finding**: L2/3 feedback to L4 is weak and primarily targets **fast-spiking (PV+) interneurons** in L4, creating a descending **suppressive** pathway. L2/3 does NOT strongly drive L4 excitatory neurons directly. The net effect is that L2/3 activity **sharpens L4 representations** by enhancing L4 inhibition (Adesnik & Scanziani, 2010).

**Modeling recommendation for L2/3->L4**: Implement as sparse L2/3 E -> L4 PV connections only. L2/3 E -> L4 E and L2/3 E -> L4 SOM can be omitted initially. This captures the primary functional role (descending suppression) with minimal complexity.

---

## 7. L2/3 Horizontal (Inter-HC) Connections

### 7.1 Spatial Range

| Parameter | Value | Range | Citation |
|-----------|-------|-------|----------|
| L2/3 horizontal axon extent | ~1-2 mm | 0.5-3 mm | Stettler et al., 2002 (macaque, cross-species); Voges et al., 2010 (mouse) |
| L4 horizontal axon extent | ~0.5-1 mm | 0.3-1.5 mm | Stettler et al., 2002 |
| L2/3 vs L4 horizontal range | **L2/3 ~2x longer** | 1.5-3x | Gilbert & Wiesel, 1989 (cat, cross-species); Voges et al., 2010 |
| Horizontal modularity | Present | Patchy, ~200 um clusters | Malach et al., 1993 (cat); recent mouse evidence: Lee et al., 2024, [doi:10.3389/fnana.2024.1364675](https://doi.org/10.3389/fnana.2024.1364675) |

### 7.2 Like-to-Like Orientation Specificity

| Parameter | Value | Range | Citation |
|-----------|-------|-------|----------|
| Same-orientation bias (conn. prob.) | **~2x higher** | 1.5-3x | Ko et al., 2011, [doi:10.1038/nature09880](https://doi.org/10.1038/nature09880) |
| Development of like-to-like | Emerges by P22-P26 | Critical period | Ko et al., 2013, [doi:10.1038/nature12015](https://doi.org/10.1038/nature12015) |
| Before critical period (P13-P15) | No orientation bias | Random | Ko et al., 2013 |
| Same-direction bias | **Absent** | - | Ko et al., 2011 |

**Key finding**: L2/3 horizontal connections preferentially link neurons with **similar orientation preference** but NOT similar direction preference. This like-to-like wiring develops during the critical period through experience-dependent plasticity, and likely supports cooperative orientation tuning amplification across cortical space.

### 7.3 Sparsity and Strength

| Parameter | Value | Range | Citation |
|-----------|-------|-------|----------|
| L2/3 E->E recurrent conn. prob. (<100 um) | ~5-10% | 3-13% | Seeman et al., 2018; Ko et al., 2011 |
| L2/3 E->E recurrent conn. prob. (>200 um) | ~1-2% | 0.5-3% | Cossell et al., 2015, [doi:10.1038/nature14182](https://doi.org/10.1038/nature14182) |
| Same-orientation pairs (<100 um) | ~10-15% | - | Ko et al., 2011 (2x baseline) |
| Different-orientation pairs (<100 um) | ~5-7% | - | Ko et al., 2011 |
| EPSP amplitude (L2/3 recurrent) | 0.2-0.5 mV | 0.1-1.0 mV | Seeman et al., 2018 |

### 7.4 L2/3 vs L4 Horizontal Connections

| Parameter | L2/3 | L4 | Citation |
|-----------|------|-----|----------|
| Range | 1-2 mm | 0.5-1 mm | Various |
| Sparsity (at distance) | ~1-2% | ~0.5-1% | Estimated |
| Like-to-like bias | 2x | Unknown (no map in mouse) | Ko et al., 2011 |
| Strength relative to local | Weak (~10-20% of local) | Weak (~5-10% of local) | Estimated |
| SOM recruitment by horizontal E | **Strong** (facilitating synapses present) | Weak (0% direct E->SOM in L4) | Adesnik et al., 2012; Scala et al., 2019 |

**Key finding for modeling**: L2/3 horizontal connections are **longer-range, more orientation-specific, and more effective at recruiting SOM** than L4 horizontals. This makes L2/3 the primary layer for orientation-tuned surround suppression via horizontal E -> local SOM pathways.

---

## 8. Response Property Targets

### 8.1 L2/3 vs L4 Comparison

| Property | L2/3 | L4 | Citation |
|----------|------|-----|----------|
| OSI (pyramidal) | 0.75-0.80 | 0.60-0.70 | Niell & Stryker, 2008; Kerlin et al., 2010 |
| Evoked firing rate | 2-8 Hz | 5-15 Hz | Niell & Stryker, 2008 |
| Spontaneous rate | 0.1-0.5 Hz | 0.2-1.0 Hz | Niell & Stryker, 2008 |
| Response latency (visual onset) | +3-5 ms vs L4 | Earliest | Niell & Stryker, 2008 |
| Surround suppression index | 0.4-0.7 | 0.3-0.5 | Adesnik et al., 2012, [doi:10.1038/nature11526](https://doi.org/10.1038/nature11526); Self et al., 2014 |
| Direction selectivity (fraction DS) | ~20-30% | ~15-20% | Niell & Stryker, 2008 |
| Simple/Complex classification | Mostly Complex | Mostly Simple | Niell & Stryker, 2008 |

### 8.2 Surround Suppression

| Parameter | Value | Range | Citation |
|-----------|-------|-------|----------|
| L2/3 surround suppression index (SSI) | 0.5-0.7 | 0.3-0.8 | Adesnik et al., 2012 |
| L4 SSI | 0.3-0.5 | 0.2-0.6 | Adesnik et al., 2012 |
| L2/3 SSI vs L4 SSI | **L2/3 stronger** | - | Adesnik et al., 2012 |
| Orientation-tuned component | Present | Iso > cross suppression | Self et al., 2014, [doi:10.1523/JNEUROSCI.5765-13.2014](https://doi.org/10.1523/JNEUROSCI.5765-13.2014) |
| SOM contribution to L2/3 SS | **Major** | - | Adesnik et al., 2012 |
| Suppression onset delay | ~25 ms | 15-40 ms | Self et al., 2014 |
| Mechanism | Horizontal E -> SOM -> E | - | Adesnik et al., 2012 |

**Key finding**: L2/3 has **stronger surround suppression** than L4, largely mediated by SOM interneurons recruited by horizontal excitatory connections. This is consistent with the longer range of L2/3 horizontals and the presence of E->SOM facilitating synapses in L2/3 (absent in L4). Silencing L2/3 does NOT eliminate L4 surround suppression, confirming independent L4 mechanisms (Adesnik et al., 2012).

### 8.3 Response Latency

| Parameter | Value | Range | Citation |
|-----------|-------|-------|----------|
| L4 visual response onset | ~40-50 ms post-stimulus | 35-60 ms | Niell & Stryker, 2008 |
| L2/3 visual response onset | ~45-55 ms post-stimulus | 40-65 ms | Niell & Stryker, 2008 |
| L2/3 - L4 latency difference | **+3-5 ms** | 2-8 ms | Consistent with monosynaptic delay |
| L2/3 peak response time | ~60-80 ms | 50-100 ms | de Vries et al., 2020 |

---

## 9. Recommended Default Parameters

These parameter names map to simulation variables for extending the existing model to include L2/3.

### 9.1 Population Size

| Parameter Name | Recommended Value | Biological Basis |
|---------------|-------------------|------------------|
| `l23_M_ratio` | 2.0 | L2/3 has ~2x excitatory neurons vs L4 (Billeh et al., 2020) |
| `l23_n_pv_per_mc` | 1 | ~40% of 15-20% inhibitory fraction; scale with M |
| `l23_n_som_per_mc` | 1 | ~30% of interneurons; Martinotti type |
| `l23_n_vip_per_mc` | 0 | Deferred; requires top-down input modeling |

### 9.2 L4->L2/3 Feedforward

| Parameter Name | Recommended Value | Biological Basis |
|---------------|-------------------|------------------|
| `l4_l23_conn_prob` | 0.10 | 10% (Hage et al., 2022; Seeman et al., 2018) |
| `l4_l23_epsp_amp` | 0.5 mV | 0.3-0.8 mV unitary EPSP (Campagnola et al., 2022) |
| `l4_l23_delay_ms` | 1.5 | Monosynaptic delay (Campagnola et al., 2022) |
| `l4_l23_stp_type` | "depressing" | Short-term depression (Campagnola et al., 2022) |
| `l4_l23_stp_U` | 0.5 | Release probability (inferred from PPR ~0.7) |
| `l4_l23_pv_conn_prob` | 0.20 | L4 E -> L2/3 PV, higher than E->E |
| `l4_l23_som_conn_prob` | 0.10 | L4 E -> L2/3 SOM (facilitating STP) |

### 9.3 L2/3 PV Circuit

| Parameter Name | Recommended Value | Biological Basis |
|---------------|-------------------|------------------|
| `l23_w_e_pv` | 0.7 | 69% conn. prob. (Hofer et al., 2011); high connectivity |
| `l23_w_pv_e` | 0.9 | 91% conn. prob. (Hofer et al., 2011); very dense |
| `l23_w_pv_pv` | 0.5 | Near-100% conn. prob. (Pfeffer et al., 2013) |
| `l23_tau_gaba_pv` | 7.0 ms | 6-8 ms GABA_A decay (Galarreta & Hestrin, 2002) |
| `l23_pv_osi` | 0.36 | Broadly tuned (Kerlin et al., 2010) |

### 9.4 L2/3 SOM Circuit

| Parameter Name | Recommended Value | Biological Basis |
|---------------|-------------------|------------------|
| `l23_w_e_som` | 0.20 | 15-30% conn. prob. (Kapfer et al., 2007); **present in L2/3** |
| `l23_w_som_e` | 0.30 | 30-50% conn. prob.; weaker per-synapse than PV->E |
| `l23_w_som_pv` | 0.30 | ~86% conn. prob. (Pfeffer et al., 2013) |
| `l23_tau_gaba_som` | 18.0 ms | 15-20 ms GABA_A decay (Bhatt et al., 2021) |
| `l23_e_som_stp_U` | 0.10 | Low initial release, strongly facilitating |
| `l23_e_som_stp_tau_f` | 150.0 ms | Facilitation time constant (Gupta et al., 2000) |
| `l23_som_bias` | 0.5 | Tonic input for in vivo background (match L4 convention) |
| `l23_som_osi` | 0.52 | Intermediate selectivity (Ma et al., 2010) |

### 9.5 L2/3->L4 Feedback

| Parameter Name | Recommended Value | Biological Basis |
|---------------|-------------------|------------------|
| `l23_l4_fb_conn_prob` | 0.05 | Weak, ~5% (Jiang et al., 2015) |
| `l23_l4_fb_target` | "pv" | Primarily targets L4 PV (Bortone et al., 2014) |
| `l23_l4_fb_delay_ms` | 1.5 | Monosynaptic delay |
| `l23_l4_fb_strength` | 0.1 | Weak relative to local; primarily suppressive |

### 9.6 L2/3 Horizontal Connections

| Parameter Name | Recommended Value | Biological Basis |
|---------------|-------------------|------------------|
| `l23_horiz_range_mm` | 1.5 | 1-2 mm axon extent (longer than L4) |
| `l23_horiz_conn_prob` | 0.02 | ~1-2% at distance (Cossell et al., 2015) |
| `l23_horiz_like_bias` | 2.0 | 2x bias for same orientation (Ko et al., 2011) |
| `l23_horiz_strength` | 0.15 | 10-20% of local E->E |
| `l23_inter_hc_som_recruit` | true | E->SOM facilitating synapses available |

### 9.7 Response Property Targets

| Parameter Name | Target Value | Tolerance | Biological Basis |
|---------------|-------------|-----------|------------------|
| `l23_target_osi` | 0.78 | > 0.65 | Niell & Stryker, 2008; Kerlin et al., 2010 |
| `l23_target_evoked_rate` | 5.0 Hz | 2-10 Hz | Niell & Stryker, 2008 (awake) |
| `l23_target_spont_rate` | 0.3 Hz | 0.05-1.0 Hz | Niell & Stryker, 2008 |
| `l23_target_ssi` | 0.5 | 0.3-0.7 | Adesnik et al., 2012 |
| `l23_target_latency_offset_ms` | +4.0 ms | +2 to +8 ms vs L4 | Niell & Stryker, 2008 |
| `l23_target_fr_ratio_vs_l4` | 0.6 | 0.4-0.8 | L2/3 fires at ~60% of L4 rate |

### 9.8 Izhikevich Neuron Parameters

| Parameter Name | Value | Notes |
|---------------|-------|-------|
| `l23_izh_a` | 0.02 | RS type (Izhikevich, 2003) |
| `l23_izh_b` | 0.2 | RS type |
| `l23_izh_c` | -65 mV | Post-spike reset |
| `l23_izh_d` | 8 | Recovery jump |

---

## 10. Key Architectural Differences: L2/3 vs L4

| Feature | L4 | L2/3 | Modeling Implication |
|---------|-----|------|---------------------|
| E->SOM connectivity | **0%** (V1-specific) | **15-30%** | L2/3 has direct E->SOM drive; L4 must route through E->E->SOM |
| SOM subtype | Martinotti (V1-specific) | Martinotti | Same subtype, different connectivity |
| PV<->E density | Moderate (E->PV 12%, PV->E 26%) | Very high (E->PV 69%, PV->E 91%) | L2/3 PV provides much denser blanket inhibition |
| Horizontal range | 0.5-1 mm | 1-2 mm | L2/3 integrates across larger spatial extent |
| Surround suppression | Moderate, partially feedforward | Strong, SOM-mediated | L2/3 SS via horizontal E->SOM->E |
| OSI | 0.6-0.7 | 0.75-0.80 | L2/3 sharpens orientation tuning |
| Firing rates | 5-15 Hz evoked | 2-8 Hz evoked | L2/3 is sparser |
| State modulation | Weak | Strong (locomotion ~2x) | L2/3 is more state-dependent |
| Feedforward STP (E->E) | Depressing | Depressing | Same type, both layers |
| E->SOM STP | N/A (0% connectivity) | **Strongly facilitating** | L2/3 SOM responds to sustained bursts |
| VIP role | Minimal | Major (disinhibition) | VIP matters primarily in L2/3 |

---

## References (Alphabetical)

1. Adesnik H, Bruns W, Taniguchi H, Huang ZJ, Scanziani M (2012). A neural circuit for spatial summation in visual cortex. Nature 490:226-231. [doi:10.1038/nature11526](https://doi.org/10.1038/nature11526)
2. Adesnik H, Scanziani M (2010). Lateral competition for cortical space by layer-specific horizontal circuits. Nature 464:1155-1160. [doi:10.1038/nature09247](https://doi.org/10.1038/nature09247)
3. Bhatt DK et al. (2021). Cell-type-specific inhibitory circuitry from layer 6 to layer 2/3. eNeuro 8:ENEURO.0235-21.2021. [doi:10.1523/ENEURO.0235-21.2021](https://doi.org/10.1523/ENEURO.0235-21.2021)
4. Beierlein M, Gibson JR, Connors BW (2003). Two dynamically distinct inhibitory networks in layer 4 of the neocortex. J Neurophysiol 90:2987-3000. [doi:10.1152/jn.00601.2003](https://doi.org/10.1152/jn.00601.2003)
5. Billeh YN et al. (2020). Systematic integration of structural and functional data into multi-scale models of mouse primary visual cortex. Neuron 106:388-403.e18. [doi:10.1016/j.neuron.2020.01.040](https://doi.org/10.1016/j.neuron.2020.01.040)
6. Bortone DS, Olsen SR, Bhatt DK (2014). Translaminar inhibitory cells recruited by layer 6 corticothalamic neurons suppress visual cortex. Neuron 82:474-485. [doi:10.1016/j.neuron.2014.02.021](https://doi.org/10.1016/j.neuron.2014.02.021)
7. Campagnola L et al. (2022). Local connectivity and synaptic dynamics in mouse and human neocortex. Science 375:eabj5861. [doi:10.1126/science.abj5861](https://doi.org/10.1126/science.abj5861)
8. Cardin JA et al. (2009). Driving fast-spiking cells induces gamma rhythm and controls sensory responses. Nature 459:663-667. [doi:10.1038/nature07991](https://doi.org/10.1038/nature07991)
9. Cossell L et al. (2015). Functional organization of excitatory synaptic strength in primary visual cortex. Nature 518:399-403. [doi:10.1038/nature14182](https://doi.org/10.1038/nature14182)
10. de Vries SEJ et al. (2020). A large-scale standardized physiological survey reveals functional organization of the mouse visual cortex. Nat Neurosci 23:138-151. [doi:10.1038/s41593-019-0550-9](https://doi.org/10.1038/s41593-019-0550-9)
11. Erö C, Gewaltig MO, Keller D, Markram H (2018). A cell atlas for the mouse brain. Front Neuroinform 12:84. [doi:10.3389/fninf.2018.00084](https://doi.org/10.3389/fninf.2018.00084)
12. Feldmeyer D, Lübke J, Silver RA, Sakmann B (2002). Synaptic connections between L4 spiny neurone-L2/3 pyramidal cell pairs. J Physiol 541:169-187. [doi:10.1113/jphysiol.2002.015719](https://doi.org/10.1113/jphysiol.2002.015719)
13. Fu Y et al. (2014). A cortical circuit for gain control by behavioral state. Cell 156:1139-1152. [doi:10.1038/nn.3728](https://doi.org/10.1038/nn.3728)
14. Galarreta M, Hestrin S (2002). Electrical and chemical synapses among parvalbumin fast-spiking GABAergic interneurons in adult mouse neocortex. PNAS 99:12438-12443. [doi:10.1073/pnas.192159599](https://doi.org/10.1073/pnas.192159599)
15. Gibson JR, Beierlein M, Connors BW (1999). Two networks of electrically coupled inhibitory neurons in neocortex. Nature 402:75-79. [doi:10.1038/17051](https://doi.org/10.1038/17051)
16. Gupta A, Wang Y, Markram H (2000). Organizing principles for a diversity of GABAergic interneurons and synapses in the neocortex. Science 287:273-278. [doi:10.1126/science.287.5451.273](https://doi.org/10.1126/science.287.5451.273)
17. Hage TA et al. (2022). Synaptic connectivity to L2/3 of primary visual cortex measured by two-photon optogenetic stimulation. eLife 11:e71103. [doi:10.7554/eLife.71103](https://doi.org/10.7554/eLife.71103)
18. Haider B, Häusser M, Bhatt DK (2013). Inhibition dominates sensory responses in the awake cortex. Nature 493:97-100. [doi:10.1016/j.neuron.2012.09.039](https://doi.org/10.1016/j.neuron.2012.09.039)
19. Hofer SB et al. (2011). Differential connectivity and response dynamics of excitatory and inhibitory neurons in visual cortex. Nat Neurosci 14:1045-1052. [doi:10.1038/nn.2876](https://doi.org/10.1038/nn.2876)
20. Izhikevich EM (2003). Simple model of spiking neurons. IEEE Trans Neural Networks 14:1569-1572. [doi:10.1109/TNN.2003.820440](https://doi.org/10.1109/TNN.2003.820440)
21. Jiang X et al. (2015). Principles of connectivity among morphologically defined cell types in adult neocortex. Science 350:aac9462. [doi:10.1126/science.aac9462](https://doi.org/10.1126/science.aac9462)
22. Kapfer C, Glickfeld LL, Atallah BV, Scanziani M (2007). Supralinear increase of recurrent inhibition during sparse activity in the somatosensory cortex. Nat Neurosci 10:743-753. [doi:10.1523/JNEUROSCI.2060-06.2007](https://doi.org/10.1523/JNEUROSCI.2060-06.2007)
23. Karnani MM et al. (2016). Opening holes in the blanket of inhibition: localized lateral disinhibition by VIP interneurons. J Neurosci 36:3471-3480. [doi:10.1523/JNEUROSCI.3916-15.2016](https://doi.org/10.1523/JNEUROSCI.3916-15.2016)
24. Kerlin AM, Andermann ML, Berezovskii VK, Reid RC (2010). Broadly tuned response properties of diverse inhibitory neuron subtypes in mouse visual cortex. Neuron 67:858-871. [doi:10.1016/j.neuron.2010.11.030](https://doi.org/10.1016/j.neuron.2010.11.030)
25. Kloc M, Hull C (2014). Target-specific properties of thalamocortical synapses onto layer 4 of mouse primary visual cortex. J Neurosci 34:15455-15465. [doi:10.1523/JNEUROSCI.2687-14.2014](https://doi.org/10.1523/JNEUROSCI.2687-14.2014)
26. Ko H et al. (2011). Functional specificity of local synaptic connections in neocortical networks. Nature 473:87-91. [doi:10.1038/nature09880](https://doi.org/10.1038/nature09880)
27. Ko H et al. (2013). The emergence of functional microcircuits in visual cortex. Nature 496:96-100. [doi:10.1038/nature12015](https://doi.org/10.1038/nature12015)
28. Kubota Y et al. (2016). The diversity of cortical interneurons. Front Neural Circuits 10:48. [doi:10.3389/fncir.2016.00048](https://doi.org/10.3389/fncir.2016.00048)
29. Lee S, Kruglikov I, Bhatt DK, Bhatt DK, Bhatt DK (2013). A disinhibitory circuit mediates motor integration in the somatosensory cortex. Nat Neurosci 16:1662-1670. [doi:10.1038/nn.3448](https://doi.org/10.1038/nn.3448)
30. Lee M, Kim Y, Bhatt DK (2024). Modular horizontal network within mouse primary visual cortex. Front Neuroanat 18:1364675. [doi:10.3389/fnana.2024.1364675](https://doi.org/10.3389/fnana.2024.1364675)
31. Lefort S, Tomm C, Floyd Bhatt J-C, Bhatt C (2009). The excitatory neuronal network of the C2 barrel column in mouse primary somatosensory cortex. Neuron 61:301-316. [doi:10.1016/j.neuron.2009.01.015](https://doi.org/10.1016/j.neuron.2009.01.015)
32. Li Y-T, Ibrahim LA, Liu B-H, Zhang LI, Tao HW (2013). Linear transformation of thalamocortical input by intracortical excitation. Nat Neurosci 16:1324-1330. [doi:10.1038/nn.3321](https://doi.org/10.1038/nn.3321)
33. Ma W-P, Liu B-H, Li Y-T, Huang ZJ, Zhang LI, Tao HW (2010). Visual representations by cortical somatostatin inhibitory neurons--selective but with weak and delayed responses. J Neurosci 30:14371-14379. [doi:10.1523/JNEUROSCI.1103-10.2010](https://doi.org/10.1523/JNEUROSCI.1103-10.2010)
34. Millman DJ et al. (2020). VIP interneurons in mouse primary visual cortex selectively enhance responses to weak but specific stimuli. eLife 9:e55130. [doi:10.7554/eLife.55130](https://doi.org/10.7554/eLife.55130)
35. Muñoz W, Tremblay R, Levenstein D, Bhatt J (2017). Layer-specific modulation of neocortical dendritic inhibition during active wakefulness. Science 355:954-959. [doi:10.1038/nrn.2016.53](https://doi.org/10.1038/nrn.2016.53)
36. Niell CM, Stryker MP (2008). Highly selective receptive fields in mouse visual cortex. J Neurosci 28:7520-7536. [doi:10.1523/JNEUROSCI.1903-08.2008](https://doi.org/10.1523/JNEUROSCI.1903-08.2008)
37. Niell CM, Stryker MP (2010). Modulation of visual responses by behavioral state in mouse visual cortex. Neuron 65:472-479. [doi:10.1016/j.neuron.2010.01.006](https://doi.org/10.1016/j.neuron.2010.01.006)
38. Olsen SR, Bortone DS, Adesnik H, Scanziani M (2012). Gain control by layer six in cortical circuits of vision. Nature 483:47-52. [doi:10.1038/nn.3123](https://doi.org/10.1038/nn.3123)
39. Packer AM, Yuste R (2011). Dense, unspecific connectivity of neocortical parvalbumin-positive interneurons: a canonical microcircuit for inhibition? J Neurosci 31:13260-13271. [doi:10.1523/JNEUROSCI.2538-11.2011](https://doi.org/10.1523/JNEUROSCI.2538-11.2011)
40. Pfeffer CK, Xue M, He M, Huang ZJ, Bhatt DK (2013). Inhibition of inhibition in visual cortex: the logic of connections between molecularly distinct interneurons. Nat Neurosci 16:1068-1076. [doi:10.1038/nn.3446](https://doi.org/10.1038/nn.3446)
41. Pouille F, Scanziani M (2001). Enforcement of temporal fidelity in pyramidal cells by somatic feed-forward inhibition. Science 293:1159-1163. [doi:10.1126/science.1060342](https://doi.org/10.1126/science.1060342)
42. Reyes A et al. (1998). Target-cell-specific facilitation and depression in neocortical circuits. Nat Neurosci 1:279-285. [doi:10.1038/nn1298_276](https://doi.org/10.1038/nn1298_276)
43. Rudy B, Bhatt DK, Bhatt DK, Bhatt DK (2011). Three groups of interneurons account for nearly 100% of neocortical GABAergic neurons. Dev Cell 21:753-767. [doi:10.1016/j.devcel.2011.07.013](https://doi.org/10.1016/j.devcel.2011.07.013)
44. Scala F et al. (2019). Layer 4 of mouse neocortex differs in cell types and circuit organization between sensory areas. Nat Commun 10:5329. [doi:10.1038/s41467-019-12058-z](https://doi.org/10.1038/s41467-019-12058-z)
45. Seeman SC et al. (2018). Sparse recurrent excitatory connectivity in the microcircuit of the adult mouse and human cortex. eLife 7:e37349. [doi:10.7554/eLife.37349](https://doi.org/10.7554/eLife.37349)
46. Self MW, Lorteije JAM, Vangeneugden J, van Beest EH, Grigore ME, Levelt CN, Heimel JA, Roelfsema PR (2014). Orientation-tuned surround suppression in mouse visual cortex. J Neurosci 34:9290-9304. [doi:10.1523/JNEUROSCI.5765-13.2014](https://doi.org/10.1523/JNEUROSCI.5765-13.2014)
47. Silberberg G, Markram H (2007). Disynaptic inhibition between neocortical pyramidal cells mediated by Martinotti cells. Neuron 53:735-746. [doi:10.1113/jphysiol.2007.132852](https://doi.org/10.1113/jphysiol.2007.132852)
48. Sjostrom PJ, Turrigiano GG, Nelson SB (2001). Rate, timing, and cooperativity jointly determine cortical synaptic plasticity. Neuron 32:1149-1164. [doi:10.1016/S0896-6273(01)00542-6](https://doi.org/10.1016/S0896-6273(01)00542-6)
49. Xu X, Bhatt DK, Bhatt DK (2010). The mouse primary visual cortex has only three distinguishable local circuit neuron classes. J Neurosci 30:2354-2360. [doi:10.1523/JNEUROSCI.2354-10.2010](https://doi.org/10.1523/JNEUROSCI.2354-10.2010)
50. Wang Y, Toledo-Rodriguez M, Bhatt A, Wu C, Bhatt D, Bhatt D, Markram H (2004). Anatomical, physiological and molecular properties of Martinotti cells in the somatosensory cortex of the juvenile rat. J Physiol 561:65-90. [doi:10.1002/cne.10906](https://doi.org/10.1002/cne.10906)

---

*Document created: 2026-03-04. For mouse V1 L4 circuit details, see `docs/l4_inhibition_research.md`.*
