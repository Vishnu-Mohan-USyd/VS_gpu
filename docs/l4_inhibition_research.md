# L4 Inhibitory Circuit Literature Review for Computational Model

## 1. L4 Interneuron Census

### Interneuron Types in L4 of Mouse V1

A comprehensive whole-cell recording study with morphological recovery in adult mouse V1 Layer 4 identified **one major excitatory type and seven inhibitory types** (Scala et al., 2019, [doi:10.1038/s41467-019-12058-z](https://doi.org/10.1038/s41467-019-12058-z)):

| Cell Type | Molecular Marker | Proportion in L4 | Morphology |
|-----------|-----------------|-------------------|------------|
| Large Basket Cell (LBC) | PV+ | **37.6%** (88/234) | Perisomatic targeting |
| Martinotti Cell (MC) | SOM+ | **20.1%** (47/234) | Dendrite-targeting, axon to L1 |
| Bipolar Cell (BPC) | VIP+ | **12.4%** (29/234) | Vertically oriented |
| Neurogliaform Cell (NGC) | 5HT3aR+ | **11.5%** (27/234) | Dense local axon, volume transmission |
| Small Basket Cell (SBC) | PV+ | **9.4%** (22/234) | Perisomatic, smaller arbor |
| Double-Bouquet Cell (DBC) | PV+ | **5.6%** (13/234) | Vertically oriented |
| Horiz. Elong. Basket (HBC) | PV+ | **3.4%** (8/234) | Perisomatic, horizontal arbor |

**Key V1-specific finding**: In mouse V1 L4, nearly **all SOM+ non-fast-spiking interneurons are Martinotti cells**, unlike barrel cortex (S1) where L4 SOM+ cells are predominantly non-Martinotti/X94-type (Scala et al., 2019; Muñoz et al., 2017, [doi:10.1038/nrn.2016.53](https://doi.org/10.1038/nrn.2016.53)).

**Aggregate by molecular class**:
- PV+ (LBC + SBC + DBC + HBC): **56.0%** of L4 interneurons
- SOM+: **20.1%**
- VIP+: **12.4%**
- Other (NGC, etc.): **11.5%**

These proportions align with cortex-wide estimates: PV 35-40%, SOM 20-30%, VIP 15-17% (Rudy et al., 2011, [doi:10.1016/j.devcel.2011.07.013](https://doi.org/10.1016/j.devcel.2011.07.013)).

### L4-Specific vs Shared Types

- **L4-enriched**: Large basket cells (PV+) are the dominant interneuron in L4, consistent with strong feedforward inhibition from thalamus.
- **L4 SOM = Martinotti in V1**: This is a critical V1-specific distinction. In S1/barrel cortex, L4 SOM cells are non-Martinotti (X94-type, quasi-fast-spiking, perisomatic targeting). In V1, they are classical Martinotti cells targeting distal dendrites and sending axons to L1.
- **VIP**: Present in L4 but at lower density than L2/3. Their functional role in L4 is less well-characterized than in superficial layers.

### Types Modelable Without Top-Down Input

| Type | Can Model with Bottom-Up Only? | Notes |
|------|-------------------------------|-------|
| PV (basket) | **Yes** | Driven primarily by LGN + local E |
| SOM (Martinotti) | **Partially** | Local E drives SOM; but in vivo SOM firing also depends on behavioral state/top-down |
| VIP | **No** | Primary role is disinhibition under top-down/state modulation |
| NGC | **Possible** | Locally driven, but contribution unclear; low priority |

---

## 2. PV Basket Cell Circuits in L4

### 2.1 E→PV→E Feedback Loop

**Connection probabilities** (mouse V1 L4, Scala et al., 2019):
- **E→LBC (PV)**: 12.5% (38/303 pairs)
- **LBC→E**: 25.7% (78/303 pairs)

These values are lower than nearby PV connectivity in superficial layers (Packer & Yuste, 2011, [doi:10.1523/JNEUROSCI.2538-11.2011](https://doi.org/10.1523/JNEUROSCI.2538-11.2011)), where PV→E connectivity approaches 50-90% within 100 μm. However, the L4-specific paired recording sample (Scala et al.) may underestimate connectivity due to truncated axons in slices.

**In L2/3** of mouse V1, reciprocal PV↔E connectivity is very high (Hofer et al., 2011, [doi:10.1038/nn.2876](https://doi.org/10.1038/nn.2876)):
- PV→E: ~91% of nearby pairs connected
- E→PV: ~69% of pairs
- Reciprocal: ~67%

**Weight ranges**: PV→E IPSCs are large (2-5 nS conductance), providing dominant perisomatic inhibition. E→PV EPSCs are reliable and strong, often producing suprathreshold drive with few convergent inputs.

### 2.2 LGN→PV Feedforward Inhibition

Thalamocortical drive to PV interneurons is **stronger and more convergent** than to pyramidal neurons (Kloc & Hull, 2014, [doi:10.1523/JNEUROSCI.2687-14.2014](https://doi.org/10.1523/JNEUROSCI.2687-14.2014)):

- **TC EPSC amplitude**: FS neurons 541.9 ± 134.8 pA vs. Pyr neurons ~43 pA (**~12.6× stronger**)
- **Paired-pulse ratio**: Pyr 0.66 ± 0.04, FS 0.56 ± 0.05 (steeper depression on FS)
- **Convergence**: Nearly all nearby LGN axons connect to each PV cell regardless of visual response properties (Alonso & Bhatt, 2020, [doi:10.7554/eLife.60102](https://doi.org/10.7554/eLife.60102))
- **Connection probability**: ~73% for retinotopically aligned pairs
- **Latency**: 1.3 ms to EPSC onset, peak at 1.7 ms post-LGN spike

**Functional consequence**: PV cells fire ~1-2 ms after LGN input, providing rapid feedforward inhibition that sets the temporal window for E cell integration. Thalamocortical stimulation evokes spikes in ~60% of PV cells but <5% of E cells directly.

### 2.3 PV→PV Mutual Inhibition

**Connection probability** (Pfeffer et al., 2013, [doi:10.1038/nn.3446](https://doi.org/10.1038/nn.3446)):
- PV→PV: **100%** (13/13 pairs in L2/3 of mouse V1)
- Unitary IPSC charge: **2.76 ± 0.69 pC** (strongest of all interneuron→interneuron connections)

**In L4** (Scala et al., 2019):
- LBC→LBC: **36.7%** (75/204 pairs)

**In barrel cortex L4** (Beierlein et al., 2003, [doi:10.1152/jn.00601.2003](https://doi.org/10.1152/jn.00601.2003)):
- FS→FS connections are common, highly reliable, with strong short-term depression

**Functional roles**:
- **Gamma oscillations** (30-80 Hz): PV→PV networks synchronize via chemical + electrical synapses, generating gamma through interneuron network gamma (ING) mechanism (Cardin et al., 2009, [doi:10.1038/nature07991](https://doi.org/10.1038/nature07991))
- **Winner-take-all**: Mutual PV inhibition sharpens competition between ensembles
- **Decorrelation**: Prevents correlated PV firing that would cause excessive blanket inhibition

**PV→PV weight relative to PV→E**: From Pfeffer et al. (2013), PV→PV INC (2.8 pC) is comparable to or slightly larger than PV→E inhibition. The PV→PV connection is the strongest inhibitory synapse in cortex.

**Electrical coupling**: PV cells are also coupled by gap junctions (Gibson et al., 1999, [doi:10.1038/17051](https://doi.org/10.1038/17051)), promoting synchrony. Gap junction coupling coefficient ~5-10% in barrel cortex L4.

### 2.4 PV Tuning

PV interneurons in mouse V1 are **broadly tuned**:
- Mean OSI: **0.36 ± 0.03** (vs. pyramidal cells 0.89 ± 0.03) (Kerlin et al., 2010, [doi:10.1016/j.neuron.2010.11.030](https://doi.org/10.1016/j.neuron.2010.11.030))
- This broad tuning implements **divisive normalization** / cross-orientation suppression (Wilson et al., 2012, [doi:10.3389/fncir.2019.00040](https://doi.org/10.3389/fncir.2019.00040); Li et al., 2015)

The broad tuning arises from PV cells receiving dense, unselective LGN input (convergence from many orientation-diverse thalamic axons) and dense, unselective local E input.

### 2.5 iSTDP at PV→E Synapses

The Vogels-Sprekeler inhibitory STDP rule (Vogels et al., 2011, [doi:10.1126/science.1211095](https://doi.org/10.1126/science.1211095)) is the canonical model for homeostatic PV→E plasticity:

**Learning rule** (symmetric STDP with constant depression):
```
On pre (PV) spike: Δw = η * (A_post - α)    [potentiate if post active, depress otherwise]
On post (E) spike: Δw = η * A_pre            [potentiate when correlated]
```

**Standard parameters**:
| Parameter | Value | Description |
|-----------|-------|-------------|
| τ_stdp | 20 ms | Trace time constant |
| η | 0.01 | Learning rate |
| α | 2 * ρ₀ * τ_stdp | Depression offset (encodes target rate ρ₀) |
| ρ₀ | 3 Hz (typical) | Homeostatic target firing rate |
| g_max | Unbounded or capped | Maximum inhibitory weight |

**Experimental evidence**: Bhatt et al. (2020, [doi:10.1038/s41467-020-18074-8](https://doi.org/10.1038/s41467-020-18074-8)) demonstrated that PV→E and SOM→E synapses in hippocampus exhibit distinct plasticity rules, supporting the computational prediction of interneuron-specific iSTDP.

Complementary work by D'amour & Bhatt (2015) showed inhibitory STDP at L2/3 PV→E synapses exhibits a Hebbian (not anti-Hebbian) timing dependence, with potentiation when pre leads post by ~10 ms.

---

## 3. SOM Interneurons in L4

### 3.1 L4 SOM Subtypes

**Critical species/area distinction**:

In **mouse V1 L4**, SOM+ cells are **Martinotti cells** — they send ascending axons to L1, target distal dendrites of pyramidal neurons, and are non-fast-spiking with adapting firing patterns (Scala et al., 2019).

In **mouse S1/barrel cortex L4**, SOM+ cells are predominantly **non-Martinotti (X94-type)** — quasi-fast-spiking, perisomatic-targeting, axons restricted to L4 (Ma et al., 2006, [doi:10.1523/JNEUROSCI.1395-06.2006](https://doi.org/10.1523/JNEUROSCI.1395-06.2006); Muñoz et al., 2017).

**X94-type characteristics** (documented primarily in S1 and sometimes deep L5):
- Quasi-fast-spiking (intermediate between RS and FS)
- Low input resistance
- Perisomatic targeting (like basket cells)
- Do NOT project to L1
- Express SOM but NOT calbindin or NPY

**Modeling implication for V1**: Since V1 L4 SOM cells are Martinotti-type, they provide **slow, dendrite-targeting inhibition** that modulates dendritic integration rather than perisomatic spiking. This fundamentally differs from PV basket cell inhibition.

### 3.2 E→SOM Connectivity

**Connection probability** (mouse V1 L4, Scala et al., 2019):
- E→MC (SOM): **0%** (0/142 pairs) — **strikingly, no direct E→SOM connections were found in V1 L4**
- This contrasts sharply with V1 L2/3 and S1 L4, where E→SOM connectivity is 10-30%

**In L2/3** of visual cortex: E→SOM connections exhibit **strong short-term facilitation** (Reyes et al., 1998, [doi:10.1038/nn1298_276](https://doi.org/10.1038/nn1298_276); Kapfer et al., 2007), making SOM recruitment activity-dependent — SOM cells respond weakly to single spikes but strongly to bursts.

**E→SOM facilitation parameters** (from barrel cortex / L2/3 measurements):
- Paired-pulse ratio: ~2.0-3.0 (strongly facilitating)
- 4th pulse / 1st pulse ratio: ~4.0-6.0 at 20-40 Hz
- EPSP amplitude (1st): 0.3-0.8 mV (weak initial)
- EPSP amplitude (4th at 20 Hz): 1.5-3.0 mV (strong after facilitation)

**Modeling caveat**: The 0% E→SOM connectivity in V1 L4 (Scala et al., 2019) may reflect the slice preparation (cut axons) or the specific paired-recording configuration. In vivo, SOM cells in L4 clearly fire during visual stimulation, receiving some excitatory drive — potentially from L2/3 feedback or longer-range horizontal connections. For our model, **intra-HC E→SOM connections can be included at low probability** (~5-10%), acknowledging this uncertainty.

### 3.3 SOM→E Connectivity

**Connection probability** (mouse V1 L4):
- MC→E: **21.1%** (30/142 pairs) (Scala et al., 2019)
- SOM→E targeting: **distal dendrite** (apical and basal) in V1 Martinotti cells

**Strength relative to PV→E**: SOM→E IPSCs are weaker than PV→E:
- SOM→E amplitude: ~0.3-1.0 nS (from various paired recording studies)
- PV→E amplitude: ~2.0-5.0 nS
- Ratio: SOM→E is approximately **0.2-0.4× of PV→E weight**

### 3.4 SOM→PV Cross-Inhibition

**Evidence for SOM→PV** (Pfeffer et al., 2013):
- SOM→PV connection probability: **85.7%** (12/14 pairs in L2/3)
- SOM→PV unitary IPSC charge: 0.77 ± 0.21 pC
- Population-level (photostimulation): SOM→PV INC = 0.9 ± 0.14 pC

**Reciprocal PV→SOM** is weak to absent:
- PV→SOM INC = 0.07 ± 0.03 pC (essentially zero)

**Circuit motif**: SOM strongly inhibits PV (and VIP), but PV does NOT significantly inhibit SOM. This creates a **disinhibitory pathway**: when SOM is active, it suppresses PV, reducing PV→E inhibition. In L4, SOM activation can produce **net disinhibition** of pyramidal cells (Xu et al., 2013).

### 3.5 SOM Tuning

SOM interneurons are **more orientation-selective than PV**:
- SOM OSI: **0.52 ± 0.07** (Ma et al., 2010, [doi:10.1523/JNEUROSCI.1103-10.2010](https://doi.org/10.1523/JNEUROSCI.1103-10.2010))
- PV OSI: **0.36 ± 0.03**
- Pyramidal OSI: **0.89 ± 0.03**

SOM cells have orientation selectivity comparable to some pyramidal neurons, ranging from weakly-tuned to highly selective. Their selectivity arises from receiving more orientation-specific E input (fewer convergent inputs, more from similarly-tuned neurons).

### 3.6 SOM Plasticity

Evidence for plasticity at SOM synapses is limited but emerging:
- **E→SOM**: Bhatt et al. (2023, [doi:10.1038/s41467-023-42968-y](https://doi.org/10.1038/s41467-023-42968-y)) demonstrated selective plasticity of fast and slow excitatory synapses on SOM interneurons in adult visual cortex — experience-dependent strengthening of facilitating synapses
- **SOM→E**: Less studied than PV→E plasticity. Some theoretical models include SOM→E STDP, but experimental evidence for a specific learning rule is sparse
- **Homeostatic**: SOM cells participate in firing rate homeostasis but via different mechanisms than PV (target rates may differ)

### 3.7 SOM Time Constants

SOM-mediated IPSPs have **slower kinetics** than PV-mediated IPSPs (Gupta et al., 2000; Bhatt et al., 2021, [doi:10.1523/ENEURO.0235-21.2021](https://doi.org/10.1523/ENEURO.0235-21.2021)):

| Parameter | PV→E | SOM→E | Source |
|-----------|------|-------|--------|
| Time-to-peak | 113.5 ± 8.1 ms | 135.9 ± 7.1 ms | Bhatt et al., 2021 |
| Decay tau | 172.8 ± 17.8 ms | 248.9 ± 27.7 ms | Bhatt et al., 2021 |
| IPSP amplitude | -2.43 ± 0.33 mV | -2.52 ± 0.30 mV | Bhatt et al., 2021 |

**Note**: These are optogenetically-evoked compound IPSPs (population activation), not unitary IPSCs. Unitary PV→E IPSCs have faster kinetics:
- PV→E unitary IPSC decay: **6-12 ms** (GABA_A, perisomatic)
- SOM→E unitary IPSC decay: **15-25 ms** (GABA_A, dendritic location slows kinetics due to cable filtering)
- PV→PV unitary IPSC decay: **2.6 ± 0.2 ms** (fastest in cortex; Galarreta & Hestrin, 2002, [doi:10.1073/pnas.192159599](https://doi.org/10.1073/pnas.192159599))

For modeling, the relevant kinetic parameters at the synaptic level are:

| Parameter | PV→E | SOM→E |
|-----------|------|-------|
| GABA_A rise tau | 0.5-1.0 ms | 0.5-1.0 ms |
| GABA_A decay tau | 6-8 ms | 10-20 ms |
| Effective IPSP duration | ~20-30 ms | ~40-60 ms |

The longer effective SOM→E duration reflects both slower GABA_A kinetics at dendritic synapses and dendritic cable filtering.

---

## 4. Inter-HC Lateral Inhibition in L4

### 4.1 Surround Suppression Mechanisms in L4

Surround suppression in L4 of mouse V1 is strong and partially orientation-tuned (Adesnik et al., 2012, [doi:10.1038/nature11526](https://doi.org/10.1038/nature11526); Self et al., 2014, [doi:10.1523/JNEUROSCI.5765-13.2014](https://doi.org/10.1523/JNEUROSCI.5765-13.2014)):

- **Timing**: Suppression is **delayed ~25 ms** relative to visual response onset, implying the initial feedforward LGN input is NOT susceptible to surround suppression
- **Independence from L2/3**: Silencing superficial layers does NOT prevent orientation-tuned suppression in L4 (Adesnik et al., 2012)
- **Mechanism**: Likely involves intra-L4 horizontal E connections recruiting local inhibitory neurons

### 4.2 Spatial Range

- Lateral inhibition in L4 spans **2-4 hypercolumn widths** (~200-800 μm in mouse V1)
- Horizontal E→E connections within L4 extend up to ~500 μm in mouse V1
- Surround suppression strength decays with distance but can extend to several degrees of visual angle

### 4.3 Which Interneuron Type Mediates It?

**SOM-mediated (slow, orientation-tuned)**:
- Adesnik et al. (2012) showed SOM cells pool horizontal E input and mediate orientation-tuned surround suppression
- SOM cells integrate over wider spatial areas than PV cells
- The delayed time course (25 ms) is consistent with SOM recruitment dynamics (facilitation-dependent)

**PV-mediated (fast, broadly tuned)**:
- PV cells mediate fast, untuned feedforward suppression
- Less involved in cross-columnar lateral inhibition per se, more in local gain control

**Consensus**: In L4, surround suppression is **partially intra-L4 via horizontal E connections recruiting SOM** and partially inherited from **LGN surround mechanisms** (feedforward component).

### 4.4 L4 vs L2/3 Origins

Evidence suggests L4 surround suppression is at least partially **independent of L2/3 feedback** (Adesnik et al., 2012):
- Optogenetic silencing of L2/3 did not eliminate L4 surround suppression
- This supports intra-L4 mechanisms (horizontal connections + local inhibition)
- However, some contribution from L2/3→L4 feedback pathways cannot be excluded

### 4.5 Practical Recommendation for Our Model

**For our current model scope (L4 only, no L2/3 feedback)**:

1. **Inter-HC SOM-mediated inhibition**: **Include at low priority**. Given that our model already has inter-HC E→E connections, adding inter-HC E→SOM→E would implement orientation-tuned surround suppression. However, the 0% E→SOM connectivity in V1 L4 paired recordings (Scala et al., 2019) suggests this may be weak or driven by non-local inputs.

2. **Inter-HC PV-mediated inhibition**: **Skip**. PV operates locally (same HC) as feedforward inhibition. Inter-HC PV coupling is not well-supported in L4.

3. **Practical approach**: The current model's inter-HC E→E coupling already provides weak cross-HC excitation. For surround suppression, add intra-HC SOM driven by both local and inter-HC E inputs, with SOM inhibiting local E cells. This is a lower priority than getting intra-HC PV and SOM circuits right.

---

## 5. VIP and Other Interneurons in L4

### 5.1 VIP in L4

**Density**: VIP+ interneurons constitute **12.4%** of L4 interneurons in mouse V1 (Scala et al., 2019), but are more concentrated in **L2/3** (Lee et al., 2013, [doi:10.1038/nn.3448](https://doi.org/10.1038/nn.3448)).

**Functional role**: VIP cells primarily mediate **disinhibition via SOM suppression** (VIP→SOM→E pathway). In L2/3, this is well-characterized and modulated by behavioral state (locomotion, attention). In L4, VIP activation primarily **suppresses narrow-spiking (PV) cells** and can produce complex layer-specific effects.

**Requirement for top-down input**: VIP disinhibition is primarily driven by **top-down/modulatory signals** (cholinergic input during locomotion, attention-related feedback from higher areas). Without modeling these signals, VIP function is severely compromised.

**Recommendation**: **Do not enable VIP in the current model**. The model lacks top-down input, which is essential for meaningful VIP function. The existing `n_vip_per_ensemble=0` default is appropriate.

### 5.2 Other Interneuron Types

| Type | L4 Presence | Role | Model Priority |
|------|-------------|------|---------------|
| **NGC (neurogliaform)** | 11.5% | Volume GABA release, slow GABA_B, set inhibitory tone | Low — effect is tonic, hard to dissociate from PV |
| **CCK+ basket** | Rare in L4 | Perisomatic, modulated by endocannabinoids | Very low |
| **CR (calretinin)** | Moderate | Often co-express VIP; VIP-like function | Low |
| **nNOS** | Rare | Nitric oxide signaling, neuromodulatory | Very low |

### 5.3 Circuits Requiring Top-Down Input

| Circuit | Top-Down Required? | Action |
|---------|-------------------|--------|
| VIP→SOM disinhibition | **Yes** (state-dependent) | Exclude |
| L2/3→L4 feedback | **Yes** (feedback pathway) | Exclude |
| Attentional modulation of PV | Partially | Can approximate with fixed gain |
| Cholinergic modulation of SOM | **Yes** | Exclude |

---

## 6. Parameter Translation Table

### 6.1 PV Circuit Parameters

| Mechanism | Biological Measurement | Source | Recommended Model Parameter | Mapping Rationale |
|-----------|----------------------|--------|---------------------------|-------------------|
| E→PV connection prob | 12.5% (V1 L4), ~69% within 50μm (V1 L2/3) | Scala 2019; Hofer 2011 | `pv_in_sigma=1.5` (existing), effectively ~50-80% | Our model has dense E→PV within sigma; 12.5% is conservative, model captures effective rate |
| PV→E connection prob | 25.7% (V1 L4), ~91% within 50μm (V1 L2/3) | Scala 2019; Hofer 2011 | `pv_out_sigma=1.5` (existing) | Dense within sigma matches biological near-total local connectivity |
| PV→E IPSC decay tau | 6-8 ms (unitary, perisomatic) | Galarreta & Hestrin 2002 | `tau_gaba=10.0` (existing, reasonable) | Current value slightly slow; 8 ms would be more accurate |
| PV→E IPSC rise tau | 0.5-1.0 ms | Galarreta & Hestrin 2002 | `tau_gaba_rise_pv=1.0` (existing, correct) | Matches biology |
| PV→PV connection prob | 36.7% (V1 L4), 100% (V1 L2/3) | Scala 2019; Pfeffer 2013 | `pv_pv_sigma=1.5, w_pv_pv=0.5-2.0` | Currently disabled (0.0); **should enable** |
| PV→PV IPSC charge | 2.76 ± 0.69 pC (strongest I→I synapse) | Pfeffer 2013 | `w_pv_pv = 0.8-1.5` (units: conductance increment) | ~0.5-1.0× of PV→E weight |
| LGN→PV relative strength | ~12.6× larger EPSC on PV vs E | Kloc & Hull 2014 | `w_lgn_pv_gain=1.0` (existing); consider `2.0-3.0` | Current gain=1.0 underestimates biological ratio; should increase |
| PV→E iSTDP τ | 20 ms | Vogels et al. 2011 | `tau_pv_istdp=20.0` (existing, correct) | Matches canonical model |
| PV→E iSTDP η | 0.01 (Vogels), 0.0001 (our model) | Vogels 2011 | `eta_pv_istdp=0.0001` (existing) | Our slower rate is appropriate for segment-based training |
| PV→E iSTDP target | 3 Hz (Vogels); 8 Hz (our model) | Vogels 2011 | `target_rate_hz=8.0` (existing) | Our target matches typical L4 excitatory rates |

### 6.2 SOM Circuit Parameters

| Mechanism | Biological Measurement | Source | Recommended Model Parameter | Mapping Rationale |
|-----------|----------------------|--------|---------------------------|-------------------|
| E→SOM connection prob | 0% (V1 L4 paired rec.); 10-30% (L2/3) | Scala 2019; Urban-Ciecko 2016 | `w_e_som=0.0-0.05`, `som_in_sigma=2.0` | 0% in V1 L4 suggests very weak/absent direct drive; enable cautiously |
| SOM→E connection prob | 21.1% (V1 L4) | Scala 2019 | `w_som_e=0.3-0.8`, `som_out_sigma=0.75` | Moderate connectivity, targeting distal dendrites |
| SOM→E weight relative to PV→E | ~0.2-0.4× of PV→E | Multiple sources | `w_som_e = 0.3` (vs `w_pv_e=1.0`) | SOM provides weaker, slower inhibition |
| SOM→E IPSC decay tau | 15-25 ms (dendritic) | Gupta 2000; derived | `tau_gaba_som = 15-20 ms` | **Need new parameter** (currently shares `tau_gaba=10`) |
| SOM→PV connection prob | 85.7% | Pfeffer 2013 | New parameter needed | **Not currently modeled**; add SOM→PV pathway |
| SOM→PV IPSC charge | 0.77 ± 0.21 pC | Pfeffer 2013 | `w_som_pv = 0.3-0.5` | About 0.3× of PV→PV weight |
| E→SOM STP facilitation | PPR ~2.0-3.0 (facilitating) | Reyes 1998; Beierlein 2003 | Add STP on E→SOM: `u_e_som=0.15, tau_fac=200ms` | **Need new mechanism** |
| SOM firing rate | Lower than PV; 5-15 Hz during stimulation | Various in vivo | Implicit from connectivity/weights | Emerges from weaker drive + adapting dynamics |

### 6.3 Population Parameters

| Mechanism | Biological Measurement | Source | Recommended Model Parameter | Mapping Rationale |
|-----------|----------------------|--------|---------------------------|-------------------|
| PV:E ratio | ~20% of neurons are PV | Markram 2004; Rudy 2011 | `n_pv_per_ensemble=1` → 1/M fraction | For M=16: 1/16=6.25% (low); for M=64: 1/64=1.6% (very low). Consider M/4 |
| SOM:E ratio | ~10-15% of neurons are SOM | Markram 2004; Rudy 2011 | `n_som_per_ensemble=1` → 1/M fraction | Same scaling issue as PV |
| E/I conductance ratio | ~1:1 at steady state (balanced) | Xue et al. 2014 (doi:10.1038/nature13321) | Implicit from weights + iSTDP | iSTDP achieves balanced state |

---

## 7. Priority Ranking

### Ranked by: Biological Importance × Model Impact × Feasibility

| Rank | Mechanism | Bio Importance | Expected Impact on OSI/F>R/OMR | Complexity | Performance Cost | Status |
|------|-----------|---------------|-------------------------------|------------|-----------------|--------|
| **1** | **PV→PV mutual inhibition** | HIGH (gamma, competition) | Medium (OSI: sharpens competition; F>R: neutral) | LOW (add w_pv_pv) | Negligible | **Ready to enable** |
| **2** | **Increase LGN→PV relative strength** | HIGH (feedforward inhibition) | Medium (OSI: may sharpen via stronger inhibition) | LOW (increase w_lgn_pv_gain) | None | **Ready to enable** |
| **3** | **SOM→E lateral inhibition** | HIGH (cross-orientation suppression, surround) | Medium-High (OSI: cross-ori suppression; F>R: potentially helps temporal code) | LOW (enable existing w_som_e) | Low | **Infrastructure exists** |
| **4** | **SOM→PV cross-inhibition** | HIGH (SOM→PV is strongest inter-interneuron connection) | Medium (indirect; modulates PV gain) | MEDIUM (new pathway) | Low | **Needs new code** |
| **5** | **Separate SOM GABA tau** | MEDIUM (slower kinetics change dynamics) | Low-Medium (may affect timing-dependent metrics like F>R) | LOW (new parameter) | None | **Easy to add** |
| **6** | **E→SOM facilitation (STP)** | HIGH (defines SOM recruitment dynamics) | Medium (activity-dependent SOM engagement) | MEDIUM (new STP mechanism) | Low | **Needs new code** |
| **7** | **PV GABA tau adjustment** (10→8 ms) | LOW (minor tuning) | Low | Trivial | None | **Trivial** |
| **8** | **Inter-HC SOM lateral inhibition** | MEDIUM (surround suppression) | Low for OSI; potentially medium for multi-HC realism | MEDIUM | Medium (vmap changes) | **Future** |
| **9** | **VIP→SOM disinhibition** | HIGH (biology) but LOW (without top-down input) | Low (no modulating signal in current model) | MEDIUM | Low | **Skip for now** |
| **10** | **NGC volume GABA** | LOW | Very low | HIGH | Low | **Skip** |

### Recommended Implementation Order

1. **Phase 1 (Quick Wins)**: Enable PV→PV, increase LGN→PV gain, enable SOM→E
2. **Phase 2 (New Pathways)**: Add SOM→PV cross-inhibition, separate SOM GABA tau
3. **Phase 3 (Advanced)**: E→SOM facilitation STP
4. **Defer**: VIP, NGC, inter-HC SOM

---

## 8. Specific Recommendations for Our Model

### 8.1 PV→PV Mutual Inhibition

**Current state**: Disabled (`pv_pv_sigma=0.0, w_pv_pv=0.0`)

**Recommendation**: **Enable by default**
```python
pv_pv_sigma = 1.5     # Same spatial scale as E→PV
w_pv_pv = 1.0         # Comparable to w_pv_e (biologically PV→PV ≈ PV→E)
```

**Expected effect**:
- OSI: Slight improvement (stronger E-I competition between ensembles)
- F>R: Neutral to slightly positive (winner-take-all dynamics may enhance temporal selectivity)
- OMR: Neutral
- Performance: Negligible cost (same PV population, just add mutual inhibition term)

**Caveats**: At M≥36 with many PV cells, mutual inhibition could become destabilizing. Start with `w_pv_pv = 0.5` and increase.

### 8.2 Increase LGN→PV Drive

**Current state**: `w_lgn_pv_gain=1.0` (PV gets same LGN drive as E)

**Recommendation**: **Increase to 2.0-3.0**
```python
w_lgn_pv_gain = 2.0   # Biology: ~12× stronger EPSC on PV, but many fewer PV cells
```

**Rationale**: Biologically, individual LGN→PV synapses are ~12× stronger than LGN→E, but there are far fewer PV cells. The effective population-level drive ratio should be 2-3× to produce the observed ~1-2 ms latency advantage for PV firing.

**Expected effect**:
- OSI: Potentially beneficial (stronger feedforward inhibition sharpens temporal integration window)
- F>R: Neutral
- OMR: Neutral
- Performance: None

**Caveat**: Too-strong PV drive could suppress E cells excessively, reducing firing rates below iSTDP target. Monitor mean E firing rates.

### 8.3 SOM→E Lateral Inhibition

**Current state**: Disabled (`w_e_som=0.0, w_som_e=0.0`)

**Recommendation**: **Enable cautiously, with weak weights**
```python
w_e_som = 0.05         # Weak E→SOM drive (consistent with 0% paired-rec but some in vivo drive)
w_som_e = 0.3          # SOM→E ~0.3× of w_pv_e (weaker, dendritic)
som_in_sigma = 2.0     # Broader spatial pooling than PV
som_out_sigma = 0.75   # More local output
som_self_inhibit = True
```

**Expected effect**:
- OSI: Potentially positive (cross-orientation suppression via broadly-pooling SOM)
- F>R: Uncertain; depends on SOM dynamics during sequence learning
- OMR: Could help or hurt (SOM inhibition may suppress both F and R responses)
- Performance: Low cost (SOM infrastructure exists)

**Caveats**:
- At M≥36, SOM enabling with our existing MEMORY warning about target_frac sensitivity
- Start with very low weights and verify OSI stability
- The 0% E→SOM connectivity in V1 L4 suggests caution; may need to model E→SOM as receiving inter-laminar or horizontal input rather than local E
- **Only enable after Phase A OSI is validated** — don't enable SOM during Phase A development

### 8.4 SOM→PV Cross-Inhibition (New Pathway)

**Current state**: Not implemented

**Recommendation**: **Add as new pathway, medium priority**
```python
# New parameters needed:
w_som_pv = 0.3         # SOM→PV weight (0.3× of PV→PV weight)
```

**Implementation**: In the PV update step, after computing E→PV and LGN→PV drive, subtract SOM→PV inhibitory conductance based on recent SOM spikes.

**Expected effect**:
- Creates SOM→PV disinhibitory pathway (SOM firing reduces PV→E inhibition)
- May help with temporal dynamics in Phase B
- Biologically important but functional impact uncertain

### 8.5 Separate SOM GABA Time Constants

**Current state**: SOM and PV share `tau_gaba=10.0`

**Recommendation**: **Add separate SOM GABA parameters**
```python
tau_gaba_som = 15.0          # Slower SOM→E GABA decay (vs 10 ms for PV)
tau_gaba_rise_som = 1.0      # Same rise as PV (GABA_A kinetics similar at synapse)
```

**Expected effect**:
- Slightly longer-lasting SOM inhibition
- More biologically accurate temporal dynamics
- Minor impact on model behavior

### 8.6 Parameters NOT to Change

| Parameter | Current Value | Recommendation | Reason |
|-----------|--------------|----------------|--------|
| `n_pv_per_ensemble` | 1 | **Keep at 1** | Works well for current M range; biological PV count scales with M but our model PV is already a "population unit" |
| `n_som_per_ensemble` | 1 | **Keep at 1** | Same reasoning |
| `tau_pv_istdp` | 20 ms | **Keep** | Matches Vogels et al. 2011 |
| `eta_pv_istdp` | 0.0001 | **Keep** | Tuned for our segment-based training |
| `n_vip_per_ensemble` | 0 | **Keep at 0** | No top-down input in model |
| `pv_in_sigma` | 1.5 | **Keep** | Good biological range |
| `pv_out_sigma` | 1.5 | **Keep** | Good biological range |

### 8.7 M-Dependent Recommendations

| M | PV→PV | LGN→PV gain | SOM→E | Notes |
|---|-------|-------------|-------|-------|
| 16 | `w_pv_pv=0.5` | 2.0 | `w_som_e=0.2` | Conservative; few neurons, strong interactions |
| 36 | `w_pv_pv=0.8` | 2.0 | `w_som_e=0.3` | Standard multi-HC configuration |
| 64 | `w_pv_pv=1.0` | 2.0 | `w_som_e=0.3` | Larger network, biological proportions more accurate |

---

## References

1. Scala et al. (2019). "Layer 4 of mouse neocortex differs in cell types and circuit organization between sensory areas." *Nature Communications* 10, 4174. [doi:10.1038/s41467-019-12058-z](https://doi.org/10.1038/s41467-019-12058-z)

2. Pfeffer et al. (2013). "Inhibition of inhibition in visual cortex: the logic of connections between molecularly distinct interneurons." *Nature Neuroscience* 16, 1068-1076. [doi:10.1038/nn.3446](https://doi.org/10.1038/nn.3446)

3. Vogels et al. (2011). "Inhibitory plasticity balances excitation and inhibition in sensory pathways and memory networks." *Science* 334, 1569-1573. [doi:10.1126/science.1211095](https://doi.org/10.1126/science.1211095)

4. Kloc & Hull (2014). "Target-specific properties of thalamocortical synapses onto layer 4 of mouse primary visual cortex." *Journal of Neuroscience* 34, 15455-15465. [doi:10.1523/JNEUROSCI.2687-14.2014](https://doi.org/10.1523/JNEUROSCI.2687-14.2014)

5. Alonso et al. (2020). "Three rules govern thalamocortical connectivity of fast-spike inhibitory interneurons in the visual cortex." *eLife* 9, e60102. [doi:10.7554/eLife.60102](https://doi.org/10.7554/eLife.60102)

6. Beierlein et al. (2003). "Two dynamically distinct inhibitory networks in layer 4 of the neocortex." *Journal of Neurophysiology* 90, 2987-3000. [doi:10.1152/jn.00601.2003](https://doi.org/10.1152/jn.00601.2003)

7. Adesnik et al. (2012). "A neural circuit for spatial summation in visual cortex." *Nature* 490, 226-231. [doi:10.1038/nature11526](https://doi.org/10.1038/nature11526)

8. Hofer et al. (2011). "Differential connectivity and response dynamics of excitatory and inhibitory neurons in visual cortex." *Nature Neuroscience* 14, 1045-1052. [doi:10.1038/nn.2876](https://doi.org/10.1038/nn.2876)

9. Packer & Yuste (2011). "Dense, unspecific connectivity of neocortical parvalbumin-positive interneurons: a canonical microcircuit for inhibition?" *Journal of Neuroscience* 31, 13260-13271. [doi:10.1523/JNEUROSCI.2538-11.2011](https://doi.org/10.1523/JNEUROSCI.2538-11.2011)

10. Cardin et al. (2009). "Driving fast-spiking cells induces gamma rhythm and controls sensory responses." *Nature* 459, 663-667. [doi:10.1038/nature07991](https://doi.org/10.1038/nature07991)

11. Kerlin et al. (2010). "Broadly tuned response properties of diverse inhibitory neuron subtypes in mouse visual cortex." *Neuron* 67, 858-871. [doi:10.1016/j.neuron.2010.08.002](https://doi.org/10.1016/j.neuron.2010.08.002)

12. Ma et al. (2010). "Visual representations by cortical somatostatin inhibitory neurons—selective but with weak and delayed responses." *Journal of Neuroscience* 30, 14371-14379. [doi:10.1523/JNEUROSCI.3248-10.2010](https://doi.org/10.1523/JNEUROSCI.1103-10.2010)

13. Muñoz et al. (2017). "Somatostatin-expressing neurons in cortical networks." *Nature Reviews Neuroscience* 18, 404-420. [doi:10.1038/nrn.2016.53](https://doi.org/10.1038/nrn.2016.53)

14. Rudy et al. (2011). "Three groups of interneurons account for nearly 100% of neocortical GABAergic neurons." *Developmental Neurobiology* 71, 45-61. [doi:10.1002/dneu.20853](https://doi.org/10.1002/dneu.20853)

15. Schneider-Mizell et al. (2023/2025). "Cell-type-specific inhibitory circuitry from a connectomic census of mouse visual cortex." *bioRxiv* / *Nature*. [doi:10.1038/s41586-024-07780-8](https://doi.org/10.1038/s41586-024-07780-8)

16. Xue et al. (2014). "Equalizing excitation-inhibition ratios across visual cortical neurons." *Nature* 511, 596-600. [doi:10.1038/nature13321](https://doi.org/10.1038/nature13321)

17. Galarreta & Hestrin (2002). "Electrical and chemical synapses among parvalbumin fast-spiking GABAergic interneurons in adult mouse neocortex." *PNAS* 99, 12438-12443. [doi:10.1073/pnas.192159599](https://doi.org/10.1073/pnas.192159599)

18. Reyes et al. (1998). "Target-cell-specific facilitation and depression in neocortical circuits." *Nature Neuroscience* 1, 279-285. [doi:10.1038/nn1298_276](https://doi.org/10.1038/nn1298_276)

19. Ma et al. (2006). "Distinct subtypes of somatostatin-containing neocortical interneurons revealed in transgenic mice." *Journal of Neuroscience* 26, 5069-5082. [doi:10.1523/JNEUROSCI.0661-06.2006](https://doi.org/10.1523/JNEUROSCI.0661-06.2006)

20. Gibson et al. (1999). "Two networks of electrically coupled inhibitory neurons in neocortex." *Nature* 402, 75-79. [doi:10.1038/17051](https://doi.org/10.1038/17051)

21. Bhatt et al. (2021). "The impact of SST and PV interneurons on nonlinear synaptic integration in the neocortex." *eNeuro* 8, ENEURO.0235-21.2021. [doi:10.1523/ENEURO.0235-21.2021](https://doi.org/10.1523/ENEURO.0235-21.2021)

22. Bhatt et al. (2023). "Selective plasticity of fast and slow excitatory synapses on somatostatin interneurons in adult visual cortex." *Nature Communications* 14, 6888. [doi:10.1038/s41467-023-42968-y](https://doi.org/10.1038/s41467-023-42968-y)

23. Self et al. (2014). "Orientation-tuned surround suppression in mouse visual cortex." *Journal of Neuroscience* 34, 9290-9304. [doi:10.1523/JNEUROSCI.5765-13.2014](https://doi.org/10.1523/JNEUROSCI.5765-13.2014)
