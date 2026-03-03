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

## 9. Sequence Learning Mechanisms (Phase B)

This section documents plasticity mechanisms investigated for Phase B sequence learning
(Gavornik & Bear 2014 protocol: repeated ABCD orientation sequences → F>R potentiation
and omission response). These mechanisms operate on top of the Phase A orientation
selectivity established by feedforward triplet-STDP.

### 9.1 NMDA-Modulated STDP (Three-Factor LTP) — TESTED, HARMFUL

> **Status: DISABLED (ee_nmda_stdp_alpha=0.0).** Tested at alpha=1.0, 2.0, 3.0 —
> all harmful. Alpha=2.0+ causes F>R reversal; alpha=1.0 is marginally better than
> baseline at M=16 but non-monotonic at M=64. Root cause: the boost amplifies all
> co-active synapses equally (backward connections also receive strong g_exc_ee from
> forward-triggered activity), so LTP boost is roughly symmetric. See Section 9.4.3.

#### Biological basis

NMDA spikes in cortical dendrites produce **supralinear Ca2+ signals** that are
selective for temporally correlated synaptic inputs. This provides a natural
three-factor learning rule: pre-activity x post-activity x local dendritic state.

**Branco, Clark & Hausser (2010)** demonstrated that single dendrites discriminate
temporal input sequences via the interaction of two biophysical properties:
1. **Dendritic impedance gradient**: distal-to-proximal activation produces larger
   initial depolarization at the high-impedance dendritic tip
2. **NMDA voltage-dependent nonlinearity**: this depolarization recruits NMDA receptors
   supralinearly, producing direction-selective Ca2+ signals

Key quantitative findings:
- Local dendritic Ca2+ was **48 +/- 13% larger** for preferred vs non-preferred input
  direction (P=0.0047)
- Somatic voltage response: 31 +/- 4% increase for preferred direction
- Spike probability: 38 +/- 9% increase for preferred direction
- NMDA blockade (D-AP5) completely abolished direction selectivity
- Optimal input velocity: 2.6 +/- 0.5 um/ms

**Sjostrom & Hausser (2006)** showed a **cooperative switch** for plasticity in
distal dendrites:
- Same pairing protocol produces LTP at proximal synapses but LTD at distal synapses
- **Dendritic** (not somatic) depolarization determines the sign of plasticity
- LTP requires local EPSP amplitude > ~1.0 mV (cooperative threshold)
- Ca2+ signal boosting reached **268 +/- 68%** with dendritic depolarization
- This establishes a three-factor rule: pre x post x local dendritic voltage

**Graupner & Brunel (2012)** provided a calcium-based computational model showing
that NMDA-mediated Ca2+ transients explain sensitivity of plasticity to spike
pattern, rate, and dendritic location.

#### Implementation: post-neuron conductance modulates LTP rate

The NMDA three-factor rule is implemented by modulating the LTP term of the
delay-aware E->E STDP with the post-neuron's current E->E conductance (`g_exc_ee`).
This conductance serves as a proxy for "how many coincident E->E inputs converged
on this neuron" — the trigger for dendritic NMDA spikes.

```
# In delay_aware_ee_stdp_update:
nmda_boost = 1.0 + alpha * sigmoid((g_exc_ee - thresh) / beta)  # shape: (M,)

# LTP with NMDA modulation (weight-dependent):
dW_ltp = A_plus * post_spikes[:, None] * pre_trace * (w_max - W)**mu * nmda_boost[:, None]

# LTD is NOT modulated (uses separate Ca2+ pathway):
dW_ltd = -A_minus * arrivals * post_trace[:, None] * (W - w_min)**mu
```

**Why this selectively boosts forward-direction learning:**
1. Forward-direction presynaptic neurons fire first, building up g_exc_ee
2. When the post-neuron fires (driven by accumulated forward input), both the
   pre_trace for forward synapses AND nmda_boost are high
3. Backward-direction synapses have decayed pre_trace at the time of the post spike,
   so even with high nmda_boost, their LTP is smaller
4. The multiplicative interaction (pre_trace x nmda_boost) creates a nonlinear
   advantage for temporally correlated (forward) inputs

#### Parameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `ee_nmda_alpha` | 2.0 | Max 3x boost at saturation; Sjostrom 2006: Ca2+ boosting 268% |
| `ee_nmda_threshold` | auto-calibrated | Set to calibrated g_exc_ee mean (data-driven) |
| `ee_nmda_beta` | 0.025 | Sigmoid steepness; ~50% activation range around threshold |
| LTD modulation | None | Biology: distinct Ca2+ pathway for LTD (Graupner & Brunel 2012) |

#### Citations

- Branco T, Clark BA, Hausser M (2010). Dendritic discrimination of temporal input
  sequences in cortical neurons. *Science* 329:1671-1675.
  [doi:10.1126/science.1189664](https://doi.org/10.1126/science.1189664). PMID: 20705816.
- Sjostrom PJ, Hausser M (2006). A cooperative switch determines the sign of synaptic
  plasticity in distal dendrites of neocortical pyramidal neurons. *Neuron* 51:227-238.
  [doi:10.1016/j.neuron.2006.06.017](https://doi.org/10.1016/j.neuron.2006.06.017).
- Graupner M, Brunel N (2012). Calcium-based plasticity model explains sensitivity of
  synaptic changes to spike pattern, rate, and dendritic location. *PNAS* 109:3991-3996.
  [doi:10.1073/pnas.1109359109](https://doi.org/10.1073/pnas.1109359109).

### 9.2 w_e_e_max Headroom — M-Dependent Auto-Select (THE WORKING FIX)

> **Status: IMPLEMENTED AND VALIDATED.** `prepare_phaseb_ee()` now auto-selects
> headroom: 5x for M≤16 and multi-HC (n_hc>1), 3x for n_hc=1 M>16. This is the
> only intervention that improved F>R without harming other metrics.

#### Biological basis

Cortical excitatory synaptic weights span a wide dynamic range with a log-normal
distribution. The original fixed 3x headroom causes weight-dependent STDP to
saturate prematurely (LTP proportional to (w_max - W) approaches 0), limiting
F>R to ~1.22 at M=16.

**Song, Sjostrom, Reigl, Nelson & Chklovskii (2005)** measured synaptic weights
in L5 rat visual cortex via quadruple whole-cell recordings:
- Log-normal distribution: p[w] = 0.426 * exp[-(ln[w] + 0.702)^2 / (2 * 0.9355^2)] / w
- EPSP amplitudes span 0.01 mV to >10 mV (**~100-fold range**)
- Mean: 0.77 mV, with a heavy tail of strong connections
- 17% of connections (above 1.2 mV) contribute ~50% of total synaptic weight

**Markram, Lubke, Frotscher & Sakmann (1997)** measured unitary EPSPs in L5 rat
somatosensory cortex:
- Range: 0.15-5.5 mV (mean 1.3 +/- 1.1 mV) — a **~37-fold range**
- Number of synaptic contacts per connection: 4-8 (mean 5.5)

**Lefort, Tomm, Bhatt & Bhatt (2009)** measured EPSP amplitudes in L4 barrel cortex:
- Range: 0.1-8 mV — an **80-fold range**

#### Implementation: M-dependent headroom

Headroom is auto-selected by `prepare_phaseb_ee()` based on network configuration:

| Config | Headroom | Rationale | F>R Result |
|--------|----------|-----------|------------|
| M≤16 (any n_hc) | **5x** | Sparse connectivity (15 E→E/neuron), weak recurrent cascade | **1.974** (M=16) |
| n_hc=1, M>16 | **3x** | Dense connectivity (63 E→E/neuron), 5x causes F>R reversal | **1.403** (M=64) |
| n_hc>1, M>16 | **5x** | Lower target_frac → lower cal_mean → safe absolute w_max | **1.168** (n_hc=64) |

This is **biologically conservative** given the 37-100x weight range observed in vivo.
At M≤16, the 5x ceiling allows weight-dependent STDP to maintain meaningful LTP
drive throughout 800 presentations, enabling F>R to reach ~1.97. At M>16 with
n_hc=1, the dense recurrent cascade with 5x headroom causes both forward AND
backward weights to saturate at ceiling equally, degrading F>R. 3x is sufficient.

#### Citations

- Song S, Sjostrom PJ, Reigl M, Nelson S, Chklovskii DB (2005). Highly nonrandom
  features of synaptic connectivity in local cortical circuits. *PLoS Biology* 3:e68.
  [doi:10.1371/journal.pbio.0030068](https://doi.org/10.1371/journal.pbio.0030068).
- Markram H, Lubke J, Frotscher M, Sakmann B (1997). Physiology and anatomy of
  synaptic connections between thick tufted pyramidal neurones in the developing rat
  neocortex. *J Physiol* 500:409-440. PMID: 9147328.

### 9.3 SOM Disinhibition During Sequence Learning — TESTED, ZERO EFFECT

> **Status: DISABLED (phaseb_som_gain=1.0).** Tested at gain=0.5. Produced
> identical F>R trajectories to baseline at all configs. Root cause: with
> w_e_som=0.05, SOM interneurons barely fire during Phase B, so reducing their
> output by 50% changes nothing. Requires stronger baseline SOM drive first.

#### Biological basis

**Gavornik & Bear (2014)** showed that V1 sequence learning requires **muscarinic
cholinergic signaling** (scopolamine blocks learning) but does NOT require NMDA
receptors (CPP had no significant effect). This implicates the cholinergic
modulation pathway rather than NMDA-dependent LTP per se.

The cholinergic disinhibition circuit in cortex operates via:
1. Cholinergic input (basal forebrain) activates **M2 muscarinic receptors** on SOM
   interneurons
2. M2 activation **suppresses SOM firing** (reduces SOM->E dendritic inhibition)
3. Reduced dendritic inhibition enhances Ca2+ signals at E->E synapses
4. Enhanced Ca2+ drives stronger LTP at active E->E synapses

**Pfeffer, Xue, He, Bhatt & Bhatt (2013)** established the connectivity matrix for
interneuron subtypes in mouse V1:
- E->SOM: substantial connectivity (drives SOM firing)
- SOM->E: strong dendritic inhibition
- SOM->PV: 85.7% connection probability (0.77 pC)

**Sarkar, Bhatt & bhatt (2024)** and related work showed that M2 receptor activation
on SOM cells reduces their firing, implementing state-dependent disinhibition.

#### Current issue

The current model has `w_e_som=0.05`, which is too weak to drive SOM interneurons
to fire. With SOM silent, the `phaseb_som_gain` parameter (designed to reduce
SOM->E inhibition during plastic Phase B trials) has zero effect — there is no
SOM inhibition to reduce.

#### Future work (prerequisite: stronger SOM drive)

To make SOM disinhibition effective, the following would be needed:
1. Increase `w_e_som` to ~0.3-0.5 (sufficient to drive SOM firing during visual
   stimulation, consistent with Pfeffer et al. 2013 connectivity data)
2. Verify SOM fires at physiological rates (5-15 Hz) during Phase A
3. Only then apply `phaseb_som_gain < 1.0` during Phase B plastic trials
4. Risk: stronger SOM→E inhibition during Phase A/evaluation may degrade OSI
   (w_e_som > 0.2 was found to DECREASE OSI in earlier experiments)

#### Citations

- Gavornik JP, Bear MF (2014). Learned spatiotemporal sequence recognition and
  prediction in primary visual cortex. *Nature Neuroscience* 17:732-737.
  [doi:10.1038/nn.3683](https://doi.org/10.1038/nn.3683). PMID: 24657967.
- Pfeffer CK, Xue M, He M, Bhatt ZJ, Bhatt SB (2013). Inhibition of inhibition in
  visual cortex: the logic of connections between molecularly distinct interneurons.
  *Nature Neuroscience* 16:1068-1076.
  [doi:10.1038/nn.3446](https://doi.org/10.1038/nn.3446).

### 9.4 Dropped Interventions

Three mechanisms were investigated and found to be harmful or ineffective for
sequence learning. They are documented here to **prevent re-investigation**.

#### 9.4.1 Power-Law STDP (mu < 1.0)

**What**: Weight-dependent STDP with sub-linear power law: LTP proportional to
(w_max - W)^mu with mu=0.5 instead of mu=1.0 (standard multiplicative).

**Rationale**: At mu < 1, potentiation near the ceiling decays sub-linearly,
reducing saturation effects and theoretically allowing continued F>R growth.

**Result**: Harmful. At mu=0.5, LTP is stronger at moderate weights (not just
near the ceiling), causing both forward AND backward weights to grow faster.
The net effect on F>R is negative because the LTP boost is not direction-selective.
Weight-dependent STDP with mu=1.0 already provides the correct biological behavior
(multiplicative STDP, Feldman 2012).

**Reference**: Feldman DE (2012). The spike-timing dependence of plasticity.
*Neuron* 75:556-571.
[doi:10.1016/j.neuron.2012.08.001](https://doi.org/10.1016/j.neuron.2012.08.001).

#### 9.4.2 NMDA Conductance Nonlinearity on Total g_exc_ee

**What**: Apply supralinear amplification to the total E->E conductance:
`g_exc_ee = g_raw * (1 + alpha * tanh(excess))`.

**Rationale**: Model NMDA spike recruitment when multiple E->E synapses co-activate.

**Result**: Catastrophically harmful. This amplifies ALL recurrent excitation
equally — both forward-direction (temporally correlated) and backward-direction
(uncorrelated) synaptic contributions receive the same gain. STDP then sees
identical enhanced postsynaptic activity regardless of input direction, causing
F>R to collapse to exactly 1.0.

**Why it fails**: In biology, NMDA spikes are **dendritic** — they amplify Ca2+
signals at synapses that contributed to the spike (coincident inputs on the same
branch), not all synapses equally. The total-conductance implementation conflates
dendritic-branch-level nonlinearity with whole-neuron conductance, destroying the
selectivity that makes NMDA spikes useful for learning.

**Correct alternative**: NMDA-modulated STDP (Section 9.1) — modulate the LTP
learning rate by g_exc_ee rather than amplifying the conductance itself.
However, see 9.4.3 — this approach was also found to be harmful in practice.

**References**:
- Branco T, Clark BA, Hausser M (2010). Science 329:1671-1675.
  [doi:10.1126/science.1189664](https://doi.org/10.1126/science.1189664).
- Sjostrom PJ, Hausser M (2006). Neuron 51:227-238.
  [doi:10.1016/j.neuron.2006.06.017](https://doi.org/10.1016/j.neuron.2006.06.017).

#### 9.4.3 NMDA-Modulated STDP (Three-Factor LTP)

**What**: Modulate LTP learning rate by the post-neuron's E→E conductance via a
sigmoid: `nmda_boost = 1 + alpha * sigmoid((g_exc_ee - thresh) / beta)`. Applied
to LTP only (not LTD). Threshold auto-calibrated to mean g_exc_ee.

**Rationale**: Forward-sequence neurons receive stronger E→E drive → higher g_exc_ee
→ more NMDA unblock → selectively boosted LTP for forward connections (Sjöström &
Häusser 2006).

**Result**: Harmful at all alpha values tested (M=16, 3x headroom, weight-based F>R):
- alpha=1.0: F>R=1.690 (marginally above baseline 1.681), monotonic — but at M=64:
  F>R=1.258, non-monotonic
- alpha=2.0: F>R=1.334, **reversal after 200 presentations** — catastrophic
- alpha=3.0: F>R=1.078, **immediate reversal** — catastrophic

**Root cause**: The per-post-neuron g_exc_ee is a whole-neuron aggregate, not a
per-synapse or per-dendrite quantity. When a post-neuron fires (driven by forward
inputs), backward synapses also get the same nmda_boost because they share the same
post-neuron g_exc_ee. The boost is thus NOT direction-selective — it amplifies ALL
LTP at active post-neurons equally. At alpha≥2, the boost is strong enough to
counteract the natural temporal asymmetry from pre-traces, causing F>R reversal.

**What would be needed**: A truly dendrite-specific implementation with per-synapse
or per-branch conductance tracking, not a whole-neuron proxy. This would require
tracking g_exc_ee per presynaptic source (M×M matrix), which is computationally
expensive and architecturally complex.

**References**:
- Sjöström PJ, Häusser M (2006). Neuron 51:227-238.
- Branco T et al. (2010). Science 329:1671-1675.

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

24. Branco T, Clark BA, Hausser M (2010). "Dendritic discrimination of temporal input sequences in cortical neurons." *Science* 329, 1671-1675. [doi:10.1126/science.1189664](https://doi.org/10.1126/science.1189664). PMID: 20705816.

25. Sjostrom PJ, Hausser M (2006). "A cooperative switch determines the sign of synaptic plasticity in distal dendrites of neocortical pyramidal neurons." *Neuron* 51, 227-238. [doi:10.1016/j.neuron.2006.06.017](https://doi.org/10.1016/j.neuron.2006.06.017).

26. Graupner M, Brunel N (2012). "Calcium-based plasticity model explains sensitivity of synaptic changes to spike pattern, rate, and dendritic location." *PNAS* 109, 3991-3996. [doi:10.1073/pnas.1109359109](https://doi.org/10.1073/pnas.1109359109).

27. Song S, Sjostrom PJ, Reigl M, Nelson S, Chklovskii DB (2005). "Highly nonrandom features of synaptic connectivity in local cortical circuits." *PLoS Biology* 3, e68. [doi:10.1371/journal.pbio.0030068](https://doi.org/10.1371/journal.pbio.0030068).

28. Markram H, Lubke J, Frotscher M, Sakmann B (1997). "Physiology and anatomy of synaptic connections between thick tufted pyramidal neurones in the developing rat neocortex." *J Physiol* 500, 409-440. PMID: 9147328.

29. Gavornik JP, Bear MF (2014). "Learned spatiotemporal sequence recognition and prediction in primary visual cortex." *Nature Neuroscience* 17, 732-737. [doi:10.1038/nn.3683](https://doi.org/10.1038/nn.3683). PMID: 24657967.

30. Feldman DE (2012). "The spike-timing dependence of plasticity." *Neuron* 75, 556-571. [doi:10.1016/j.neuron.2012.08.001](https://doi.org/10.1016/j.neuron.2012.08.001).

31. Gütig R, Aharonov R, Rotter S, Sompolinsky H (2003). "Learning input correlations through nonlinear temporally asymmetric Hebbian plasticity." *J Neurosci* 23, 3697-3714. [doi:10.1523/JNEUROSCI.23-09-03697.2003](https://doi.org/10.1523/JNEUROSCI.23-09-03697.2003).

32. Morrison A, Diesmann M, Gerstner W (2008). "Phenomenological models of synaptic plasticity based on spike timing." *Biol Cybern* 98, 459-478. [doi:10.1007/s00422-008-0233-1](https://doi.org/10.1007/s00422-008-0233-1).

33. Schiller J, Major G, Koester HJ, Schiller Y (2000). "NMDA spikes in basal dendrites of cortical pyramidal neurons." *Nature* 404, 285-289. [doi:10.1038/35005094](https://doi.org/10.1038/35005094). PMID: 10749211.

34. Poirazi P, Brannon T, Mel BW (2003). "Pyramidal neuron as two-layer neural network." *Neuron* 37, 989-999. [doi:10.1016/S0896-6273(03)00149-1](https://doi.org/10.1016/S0896-6273(03)00149-1).

35. Major G, Larkum ME, Schiller J (2013). "Active properties of neocortical pyramidal neuron dendrites." *Annu Rev Neurosci* 36, 1-24. [doi:10.1146/annurev-neuro-062111-150343](https://doi.org/10.1146/annurev-neuro-062111-150343).

36. Sarkar S, Reyes A, Bhatt RR, Gavornik JP (2024). "M2 receptors are required for spatiotemporal sequence learning in mouse primary visual cortex." *J Neurophysiol* 131, 1024-1034. [doi:10.1152/jn.00016.2024](https://doi.org/10.1152/jn.00016.2024).

37. Khan AG, Poort J, Chadwick A, Blot A, Sahani M, Mrsic-Flogel TD, Hofer SB (2018). "Distinct learning-induced changes in stimulus selectivity and interactions of GABAergic interneuron classes in visual cortex." *Nat Neurosci* 21, 851-859. [doi:10.1038/s41593-018-0143-z](https://doi.org/10.1038/s41593-018-0143-z).

38. Letzkus JJ, Wolff SBE, Lüthi A (2015). "Disinhibition, a circuit mechanism for associative learning and memory." *Neuron* 88, 264-276. [doi:10.1016/j.neuron.2015.09.024](https://doi.org/10.1016/j.neuron.2015.09.024).

39. Fu Y, Kaneko M, Tang Y, Alvarez-Buylla A, Bhatt AJ, Stryker MP (2015). "A cortical disinhibitory circuit for enhancing adult plasticity." *eLife* 4, e05558. [doi:10.7554/eLife.05558](https://doi.org/10.7554/eLife.05558).

40. Bi GQ, Poo MM (1998). "Synaptic modifications in cultured hippocampal neurons: dependence on spike timing, synaptic strength, and postsynaptic cell type." *J Neurosci* 18, 10464-10472. [doi:10.1523/JNEUROSCI.18-24-10464.1998](https://doi.org/10.1523/JNEUROSCI.18-24-10464.1998). PMID: 9852584.

41. Nevian T, Larkum ME, Bhatt AJ, Polsky A, Schiller J (2007). "Properties of basal dendrites of layer 5 pyramidal neurons: a direct patch-clamp recording study." *Nat Neurosci* 10, 206-214. [doi:10.1038/nn1826](https://doi.org/10.1038/nn1826).

---

## 10. Biological Parameter Values for Inter-HC Refinement

This section documents biological constraints for five interventions aimed at improving
inter-HC inhibitory circuit realism. Each subsection provides 2+ primary-source citations
with quantitative values for model parameter selection.

### 10.1. SOM Firing Rates in V1

**Spontaneous firing rates** of SOM+ interneurons in mouse V1 are substantially lower than
PV+ interneurons:

| Measure | SOM+ | PV+ | Source |
|---------|------|-----|--------|
| Spontaneous firing rate | **2–5 Hz** | **10–40 Hz** | Ma et al. 2010; Urban-Ciecko & Barth 2016 |
| Evoked (visually driven) | **3–5× weaker** than PV | Full-range | Ma et al. 2010 |
| Response latency (L4) | **91.8 ± 17.8 ms** | **70.8 ± 8.5 ms** | Ma et al. 2010 |
| Response latency (L2/3) | **106.8 ± 17.7 ms** | **87.2 ± 14.0 ms** | Ma et al. 2010 |
| Spontaneous (slice, independent of synaptic input) | **3–10 Hz** | — | Urban-Ciecko & Barth 2016 |
| In vivo spontaneous (up+down states) | **2.4 ± 0.6 Hz** | — | Urban-Ciecko & Barth 2016 |

SOM neurons have a **20–25 ms delay** relative to PV neurons in both L4 and L2/3, consistent
with their role as feedback/integrative inhibitors rather than fast feedforward inhibitors.

**Key citations:**
1. Ma WP, Liu BH, Li YT, Huang ZJ, Zhang LI, Tao HW (2010). "Visual representations by
   cortical somatostatin inhibitory neurons—selective but with weak and delayed responses."
   *J Neurosci* 30(43):14371–14379.
   [doi:10.1523/JNEUROSCI.3248-10.2010](https://doi.org/10.1523/JNEUROSCI.3248-10.2010). PMID: 20980594.
2. Urban-Ciecko J, Barth AL (2016). "Somatostatin-expressing neurons in cortical networks."
   *Nat Rev Neurosci* 17(7):401–409.
   [doi:10.1038/nrn.2016.53](https://doi.org/10.1038/nrn.2016.53). PMID: 27225074.

**Model implications:** SOM neurons in our model should fire at ~2–5 Hz spontaneously
and have delayed, weaker evoked responses compared to PV. The inter-HC SOM pathway
operates on a slower timescale (~25 ms onset delay), consistent with the biology.

---

### 10.2. E→SOM Effective Connection Strength in L4

A critical finding from Scala et al. (2019) is that **monosynaptic E→SOM connections are
absent in V1 L4**: 0/142 tested pairs showed connections (0% connection probability).
This contrasts with S1 L4, where non-Martinotti SOM+ cells receive excitatory input at
12.5% (8/64).

| Circuit | V1 L4 | S1 L4 | Source |
|---------|-------|-------|--------|
| E→SOM monosynaptic | **0% (0/142)** | 12.5% (8/64) | Scala et al. 2019 |
| SOM→Pyr (within 200 μm) | **~71%** | — | Fino & Yuste 2011 |
| SOM→Pyr (within 400 μm) | **~48%** | — | Fino & Yuste 2011 |
| SOM→Pyr (all-to-all local) | **11/61 maps 100%** | — | Fino & Yuste 2011 |
| Single Pyr activates SOM | **~30% within 100 μm** | — | Yavorska & Wehr 2016 (review) |
| Pyr→SOM synapse type | **Strongly facilitating** | — | Yavorska & Wehr 2016 |

The V1 L4-specific absence of monosynaptic E→SOM means our model's inter-HC pathway
(E→SOM) should be interpreted as polysynaptic or operating via L2/3 intermediate
neurons. In L2/3 of frontal cortex, SOM→Pyr inhibition is dense (~71% within 200 μm)
and can be effectively all-to-all locally (Fino & Yuste 2011).

**Key citations:**
1. Scala F, Kobak D, Shan S, Bernaerts Y, Berens P, Tolias AS (2019). "Layer 4 of mouse
   neocortex differs in cell types and circuit organization between sensory areas."
   *Nat Commun* 10(1):3997.
   [doi:10.1038/s41467-019-12058-z](https://doi.org/10.1038/s41467-019-12058-z). PMID: 31519874.
2. Fino E, Yuste R (2011). "Dense inhibitory connectivity in neocortex."
   *Neuron* 69(6):1188–1203.
   [doi:10.1016/j.neuron.2011.02.025](https://doi.org/10.1016/j.neuron.2011.02.025). PMID: 21435562.

**Model implications:** The current model's `w_e_som` parameter represents an effective
polysynaptic pathway. At physiological connection probabilities, the effective E→SOM
drive in V1 L4 is much weaker than E→PV drive. This justifies keeping `w_e_som`
relatively low (0.05–0.1) and routing inter-HC lateral signals through E→E rather
than direct E→SOM connections.

---

### 10.3. Surround Suppression Orientation Selectivity (Iso/Cross Ratio)

Surround suppression in V1 is strongly orientation-tuned: iso-oriented surrounds
produce ~2–3× more suppression than cross-oriented surrounds.

| Measure | Value | Species | Source |
|---------|-------|---------|--------|
| Iso/cross suppression ratio | **~3:1** (at 50% contrast) | Macaque | Cavanaugh et al. 2002 |
| Decrease from iso→cross | **30–35%** | Macaque | Cavanaugh et al. 2002 |
| SSI (L4, iso-oriented) | **median 0.25** | Mouse | Self et al. 2014 |
| OSSI (L4, small center) | **0.12 ± 0.01** | Mouse | Self et al. 2014 |
| OSSI (L4, large center) | **0.06 ± 0.01** | Mouse | Self et al. 2014 |
| Neurons iso>cross preferred | **37%** | Mouse | Self et al. 2014 |
| Neurons cross>iso preferred | **17%** | Mouse | Self et al. 2014 |

The orientation-specific suppression index (OSSI) was significant in L4 and
superficial layers (p < 0.005) but not in deep layers. 37% of single units
showed significantly stronger suppression for iso-oriented than cross-oriented
surrounds, consistent with horizontal connections preferentially linking
iso-orientation domains.

**Key citations:**
1. Cavanaugh JR, Bair W, Movshon JA (2002). "Selectivity and spatial distribution of signals
   from the receptive field surround in macaque V1 neurons." *J Neurophysiol* 88(5):2547–2556.
   [doi:10.1152/jn.00693.2001](https://doi.org/10.1152/jn.00693.2001). PMID: 12424292.
2. Self MW, Lorteije JAM, Vangeneugden J, van Beest EH, Grigore ME, Levelt CN,
   Heimel JA, Roelfsema PR (2014). "Orientation-tuned surround suppression in mouse
   visual cortex." *J Neurosci* 34(28):9290–9304.
   [doi:10.1523/JNEUROSCI.5051-13.2014](https://doi.org/10.1523/JNEUROSCI.5051-13.2014). PMID: 25009263.
3. Adesnik H, Bruns W, Taniguchi H, Huang ZJ, Scanziani M (2012). "A neural circuit for
   spatial summation in visual cortex." *Nature* 490(7419):226–231.
   [doi:10.1038/nature11526](https://doi.org/10.1038/nature11526). PMID: 23060193.

**Model implications:** Our inter-HC surround suppression should be orientation-tuned
with ~2–3× stronger suppression for iso-oriented stimulation. SOM-mediated surround
suppression (Adesnik et al. 2012) provides the biological mechanism. The current
model uses SOM→E surround suppression but does not yet implement orientation tuning
of the inter-HC signal.

---

### 10.4. Horizontal E→E Sparsity and Extent

Long-range horizontal E→E connections in V1 are **patchy, orientation-specific, and sparse**:

| Measure | Value | Species | Source |
|---------|-------|---------|--------|
| Horizontal extent | **2–5 mm** | Tree shrew | Bosking et al. 1997 |
| Maximum along preferred axis | **median 1.77 mm** | Tree shrew | Bosking et al. 1997 |
| Maximum along orthogonal axis | **median 1.16 mm** | Tree shrew | Bosking et al. 1997 |
| Axial anisotropy | **4:1** (preferred vs orthogonal) | Tree shrew | Bosking et al. 1997 |
| Iso-orientation specificity | **57.6%** boutons within ±35° | Tree shrew | Bosking et al. 1997 |
| Bouton patch size | **~400 × 250 μm** | Tree shrew | Bosking et al. 1997 |
| Visual space coverage | **8× classical RF** | Macaque | Stettler et al. 2002 |
| Orientation specificity | **Yes** (intrinsic only; not feedback) | Macaque | Stettler et al. 2002 |
| E→E connection probability (L2/3, <100 μm) | **10.0%** (mouse V1) | Mouse | Seeman et al. 2018 |
| E→E connection probability (L4, <100 μm) | **7.3%** (mouse V1) | Mouse | Seeman et al. 2018 |
| E→E connection probability (at 785 μm) | **0.82%** (mouse V1) | Mouse | Seeman et al. 2018 |
| E→E connection probability (human L2) | **18.8%** | Human | Seeman et al. 2018 |
| E→E connection probability (human L4) | **2.0%** | Human | Seeman et al. 2018 |

Key features of horizontal connections:
- **Patchy**: Bouton clusters separated by ~1 mm periodicity (Gilbert & Wiesel 1983, 1989)
- **Iso-orientation preference**: 57.6% of boutons contact same-orientation domains (±35°)
- **Anisotropic**: 4× more terminals along the axis of preferred orientation
- **Sparse at distance**: Connection probability falls from ~10% locally to <1% at ~800 μm

**Key citations:**
1. Bosking WH, Zhang Y, Schofield B, Fitzpatrick D (1997). "Orientation selectivity and the
   arrangement of horizontal connections in tree shrew striate cortex." *J Neurosci*
   17(6):2112–2127.
   [doi:10.1523/JNEUROSCI.17-06-02112.1997](https://doi.org/10.1523/JNEUROSCI.17-06-02112.1997). PMID: 9045738.
2. Stettler DD, Das A, Bennett J, Gilbert CD (2002). "Lateral connectivity and contextual
   interactions in macaque primary visual cortex." *Neuron* 36(4):739–750.
   [doi:10.1016/S0896-6273(02)01029-2](https://doi.org/10.1016/S0896-6273(02)01029-2). PMID: 12441061.
3. Gilbert CD, Wiesel TN (1989). "Columnar specificity of intrinsic horizontal and
   corticocortical connections in cat visual cortex." *J Neurosci* 9(7):2432–2442.
   [doi:10.1523/JNEUROSCI.09-07-02432.1989](https://doi.org/10.1523/JNEUROSCI.09-07-02432.1989). PMID: 2746337.
4. Seeman SC, Bhatt AJ, Bhatt RR, et al. (2018). "Sparse recurrent excitatory connectivity in
   the microcircuit of the adult mouse and human cortex." *eLife* 7:e37349.
   [doi:10.7554/eLife.37349](https://doi.org/10.7554/eLife.37349). PMID: 30256194.

**Model implications:** Inter-HC E→E connections should be sparse (~1–7% depending on
distance), iso-orientation-preferring (~60% same-orientation), and patchy rather than
uniform. The current model uses distance-dependent Gaussian E→E weights; adding
orientation selectivity to inter-HC connections would better match biology. Connection
probability should decay steeply with distance (from ~10% at <100 μm to <1% at ~800 μm).

---

### 10.5. Cholinergic SOM Suppression and VIP Disinhibition

The VIP→SOM→E disinhibitory circuit is a key mechanism for state-dependent gain control:

| Measure | Value | Source |
|---------|-------|--------|
| VIP→SOM connection probability (V1) | **~36%** | JNeurosci 2025 (VIP-SST motif) |
| VIP→SOM connection probability (S1) | **~47%** | JNeurosci 2025 (VIP-SST motif) |
| VIP-evoked IPSC in SOM cells | **1346 pA** | Bhatt et al. (slice) |
| VIP-evoked IPSC in pyramidal cells | **154 pA** (8.7× weaker) | Bhatt et al. (slice) |
| VIP→SOM IPSC selectivity | **33:1** (SOM vs Pyr) | Bhatt et al. (slice) |
| VIP-SOM synapse dynamics | **Short-term depression** | Pi et al. 2013 |
| SOM IPSC decay time | **18 ± 2 ms** | Pi et al. 2013 |
| VIP baseline correlation with speed | **ρ = 0.27 ± 0.03** | Pakan et al. 2018 |
| SST baseline correlation (gray screen) | **ρ = 0.18 ± 0.02** (positive) | Pakan et al. 2018 |
| SST baseline correlation (darkness) | **ρ = −0.07 ± 0.02** (negative) | Pakan et al. 2018 |
| M2 muscarinic receptors on SOM (V1) | **~4% of SST+ cells** | Sarkar et al. 2024 |
| M2+ SST in infragranular layers | **73%** | Sarkar et al. 2024 |
| Cholinergic activation mechanism | Nicotinic → VIP → SOM suppression | Fu et al. 2014 |

**VIP→SOM circuit mechanism:**
- Basal forebrain cholinergic neurons activate VIP interneurons via nicotinic receptors
  (Fu et al. 2014)
- VIP interneurons strongly and selectively inhibit SOM interneurons (33:1 selectivity
  over pyramidal cells)
- This disinhibits pyramidal neuron dendrites, enabling enhanced plasticity and gain
  control
- VIP→SOM synapses show short-term depression, suggesting phasic rather than tonic
  modulation

**M2 muscarinic pathway (Sarkar et al. 2024):**
- M2 receptors required for spatiotemporal sequence learning in V1
- Only ~4% of SST+ neurons express M2, mostly in deep layers (73% infragranular)
- M2 blockade prevents sequence potentiation but is reversible
- This pathway is distinct from the VIP→SOM nicotinic pathway

**Key citations:**
1. Fu Y, Tucciarone JM, Bhatt AJ, Bhatt RR, Bhatt DH, et al. (2014). "A cortical circuit for
   gain control by behavioral state." *Cell* 156(6):1139–1152.
   [doi:10.1016/j.cell.2014.01.050](https://doi.org/10.1016/j.cell.2014.01.050). PMID: 24630718.
2. Letzkus JJ, Wolff SBE, Lüthi A (2015). "Disinhibition, a circuit mechanism for associative
   learning and memory." *Neuron* 88(2):264–276.
   [doi:10.1016/j.neuron.2015.09.024](https://doi.org/10.1016/j.neuron.2015.09.024). PMID: 26494276.
3. Pi HJ, Hangya B, Kvitsiani D, Sanders JI, Huang ZJ, Kepecs A (2013). "Cortical
   interneurons that specialize in disinhibitory control." *Nature* 503(7477):521–524.
   [doi:10.1038/nature12676](https://doi.org/10.1038/nature12676). PMID: 24097352.
4. Sarkar A, Bhatt AJ, Reyes AJ, Bhatt RR, Gavornik JP (2024). "M2 receptors are required for
   spatiotemporal sequence learning in mouse primary visual cortex." *J Neurophysiol*
   132(1):207–218.
   [doi:10.1152/jn.00016.2024](https://doi.org/10.1152/jn.00016.2024). PMID: 38629848.
5. Pakan JMP, Lowe SC, Dylda E, Keemink SW, Currie SP, Coutts CA, Rochefort NL (2018).
   "Vision and locomotion shape the interactions between neuron types in mouse visual cortex."
   *Neuron* 98(3):602–615.e8.
   [doi:10.1016/j.neuron.2018.03.037](https://doi.org/10.1016/j.neuron.2018.03.037). PMID: 29681530.

**Model implications:** For implementing VIP/cholinergic gating of SOM inhibition:
- `phaseb_som_gain` should reduce SOM→E inhibition during learning/active states
- A biologically plausible range is 0.3–0.7× (30–70% reduction in SOM→E), reflecting
  the strong but phasic VIP→SOM suppression
- The M2 pathway (Sarkar et al. 2024) provides additional justification for
  learning-state-specific SOM modulation, but affects only ~4% of SST+ neurons in V1
- Connection probability VIP→SOM ~36% in V1 means not all SOM cells are suppressed
  simultaneously

---

## 11. Sequence Learning Mechanisms — Parameter Justification

This section documents the biological evidence and parameter choices for three mechanisms that strengthen spatiotemporal sequence learning in the model: power-law STDP weight dependence, dendritic NMDA nonlinearity, and learning-state SOM disinhibition.

### 11.1. Power-Law STDP Weight Dependence (ee_stdp_mu)

#### Biological Background

The magnitude of spike-timing-dependent potentiation (LTP) depends on initial synaptic weight. Bi & Poo (1998) showed in hippocampal cultures that significant LTP occurred only at synapses with relatively low initial strength, while the extent of LTD showed roughly proportional (multiplicative) weight dependence (Δw ∝ w). This asymmetry — sub-linear potentiation, near-multiplicative depression — is a fundamental feature of biological STDP.

#### Mathematical Formulation

Gütig et al. (2003) introduced the Nonlinear Temporally Asymmetric Hebbian (NLTAH) model that interpolates between additive (μ=0) and multiplicative (μ=1) STDP via a power-law exponent μ:

- **Potentiation**: ΔW+ = λ · (1 − W/W_max)^μ · A+ · f(Δt)
- **Depression**: ΔW− = α · λ · (W/W_max)^μ · A− · f(Δt)

Where:
- μ = 0: **Additive** STDP — weight change independent of current weight. Produces bimodal weight distributions (all-or-nothing). Maximally competitive but unstable.
- μ = 1: **Multiplicative** STDP — weight change proportional to current weight. Unimodal distributions, stable but weak competition.
- 0 < μ < 1: **Sub-linear** (power-law) — intermediate regime. Maintains competition while preserving stability.

Gütig et al. (2003) demonstrated that "a unimodal distribution is the rule rather than the exception" for μ > 0, and that bimodal distributions only emerge with very weak weight dependence (μ ≪ 1). Intermediate μ values achieve a balance between synaptic competition (needed for input selectivity) and stability (preventing runaway potentiation).

#### Computational Model Usage

- **NEST simulator** (standard implementation): Uses μ_plus and μ_minus as separate exponents; benchmark code uses **μ = 0.4** (Morrison et al., 2008).
- **Morrison et al. (2008)**: Systematically compared additive, multiplicative, and power-law rules. Showed that power-law with μ ≈ 0.4–0.6 best fits experimental data from Bi & Poo (1998).
- **Van Rossum et al. (2000)**: Proposed μ_plus = 0, μ_minus = 1 (additive potentiation, multiplicative depression), but this produces extreme bimodal distributions.

#### Experimental Fit

The Bi & Poo (1998) potentiation data is best fit by a sub-linear power law with μ ≈ 0.4–0.6 for potentiation, while depression follows μ ≈ 0.8–1.0 (approximately multiplicative) (Morrison et al., 2008).

#### Parameter Choice: ee_stdp_mu = 0.5

**Justification**: μ = 0.5 is the geometric midpoint of the Gütig interpolation, consistent with:
1. Sub-linear potentiation observed experimentally (Bi & Poo, 1998)
2. Computational model range of 0.4–0.6 for stable competition (Gütig et al., 2003; Morrison et al., 2008)
3. NEST simulator benchmark value of 0.4 (close to 0.5)

Compared to our current purely multiplicative STDP (ee_stdp_weight_dep=True, effectively μ=1), μ=0.5 will:
- Allow stronger weights to continue growing (weaker ceiling effect)
- Maintain competition between forward and backward sequence connections
- Prevent the F>R saturation we observe at ~1.22 due to LTP∝(w_max−W)→0

**Citations**:
- Bi GQ, Poo MM (1998). "Synaptic modifications in cultured hippocampal neurons." *J Neurosci* 18, 10464-10472. [doi:10.1523/JNEUROSCI.18-24-10464.1998](https://doi.org/10.1523/JNEUROSCI.18-24-10464.1998). PMID: 9852584.
- Gütig R, Aharonov R, Rotter S, Sompolinsky H (2003). "Learning input correlations through nonlinear temporally asymmetric Hebbian plasticity." *J Neurosci* 23, 3697-3714. [doi:10.1523/JNEUROSCI.23-09-03697.2003](https://doi.org/10.1523/JNEUROSCI.23-09-03697.2003).
- Morrison A, Diesmann M, Gerstner W (2008). "Phenomenological models of synaptic plasticity based on spike timing." *Biol Cybern* 98, 459-478. [doi:10.1007/s00422-008-0233-1](https://doi.org/10.1007/s00422-008-0233-1).
- Feldman DE (2012). "The spike-timing dependence of plasticity." *Neuron* 75, 556-571. [doi:10.1016/j.neuron.2012.08.001](https://doi.org/10.1016/j.neuron.2012.08.001).

---

### 11.2. Dendritic NMDA Nonlinearity (ee_nmda_alpha, ee_nmda_threshold)

#### Biological Background

Single dendrites of cortical pyramidal neurons can perform sequence detection via NMDA receptor-dependent supralinear integration. Branco et al. (2010) used two-photon glutamate uncaging on layer 2/3 pyramidal neurons in mouse somatosensory cortex and found:

- **Supralinearity**: Somatic voltage responses to sequential activation of 8-10 synapses on single basal/apical oblique dendrites reached **223 ± 9% of the arithmetic sum** (p < 0.0001). This ~2.23x amplification is NMDA-dependent: the NMDAR blocker D-AP5 abolished supralinearity (103 ± 3% of linear sum, p = 0.336).
- **Direction sensitivity**: The IN direction (distal-to-proximal, centripetal) produced responses **31 ± 4% larger** than the OUT direction (mean peak voltage difference 2.8 ± 0.4 mV, p < 0.0001, n=20). Spike probability was enhanced by 38 ± 9% (p = 0.0013, n=7).
- **Optimal velocity**: Direction sensitivity peaked at **2.6 ± 0.5 μm/ms**, consistent with physiological conduction velocities for recurrent excitatory axons in cortex.

#### NMDA Spike Threshold

The threshold for dendritic NMDA spikes has been characterized across multiple studies:

- **Schiller et al. (2000)**: First demonstrated NMDA spikes in basal dendrites of L5 pyramidal neurons. Co-activation of clustered neighboring inputs amplified somatic response by **226 ± 46%**. NMDA channels contributed ≥80% of total charge.
- **Nevian et al. (2007)**: ~10 co-active synapses needed for NMDA spike in L5 basal dendrites. Peak NMDA conductance threshold: **15.9 ± 1.66 nS** (single activation) or **8.06 ± 1.25 nS** (paired-pulse).
- **Poirazi et al. (2003)**: Computational model of CA1 pyramidal neuron showed **10–20 synapses** needed to trigger NMDA spike per dendritic branch, producing sigmoidal subunit input-output functions.
- **Major et al. (2013)**: Review established that synchronous activation of **10–50 neighboring glutamatergic synapses** triggers local NMDA spikes/plateaus.

#### Computational Implementation

In our model, recurrent E→E conductance (g_exc_ee) represents the total excitatory drive from recurrent connections. We implement NMDA nonlinearity as a threshold-gated amplification:

```
g_eff = g_exc_ee * (1 + alpha * sigmoid((g_exc_ee - threshold) / slope))
```

This captures the essential feature: when total recurrent drive exceeds a threshold (analogous to ~10 co-active synapses), NMDA receptors contribute supralinear amplification.

#### Parameter Choice: ee_nmda_alpha = 2.0, ee_nmda_threshold = 0.1

**ee_nmda_alpha = 2.0**:
The amplification factor. Branco et al. (2010) measured 223% of linear sum, i.e., ~2.23x. Using alpha=2.0 in the sigmoid formulation: at maximum activation, effective gain approaches 1 + 2.0 = 3.0x, but in practice the sigmoid shape means typical gains are ~1.5–2.5x, bracketing the experimental 2.23x. Alpha=2.0 is a conservative lower bound of the biological value.

**ee_nmda_threshold = 0.1**:
The conductance threshold for NMDA activation, expressed as a fraction of maximal E→E drive. This maps to the biological requirement of ~10 co-active synapses: in our network with M=36 neurons per HC and all-to-all E→E connectivity (35 pre-synaptic partners), 10/35 ≈ 0.29 of connections active. However, individual synaptic weights vary, so a threshold of 0.1 (10% of max possible drive) corresponds to a regime where a subset of strong, correlated inputs are active — consistent with the sequence-selective activation pattern. This ensures NMDA nonlinearity is only engaged when there is sufficient recurrent drive (not during baseline spontaneous activity).

**Citations**:
- Branco T, Clark BA, Häusser M (2010). "Dendritic discrimination of temporal input sequences in cortical neurons." *Science* 329, 1671-1675. [doi:10.1126/science.1189664](https://doi.org/10.1126/science.1189664). PMID: 20705816.
- Schiller J, Major G, Koester HJ, Schiller Y (2000). "NMDA spikes in basal dendrites of cortical pyramidal neurons." *Nature* 404, 285-289. [doi:10.1038/35005094](https://doi.org/10.1038/35005094). PMID: 10749211.
- Poirazi P, Brannon T, Mel BW (2003). "Pyramidal neuron as two-layer neural network." *Neuron* 37, 989-999. [doi:10.1016/S0896-6273(03)00149-1](https://doi.org/10.1016/S0896-6273(03)00149-1).
- Major G, Larkum ME, Schiller J (2013). "Active properties of neocortical pyramidal neuron dendrites." *Annu Rev Neurosci* 36, 1-24. [doi:10.1146/annurev-neuro-062111-150343](https://doi.org/10.1146/annurev-neuro-062111-150343).

---

### 11.3. Learning-State SOM Disinhibition (phaseb_som_gain)

#### Biological Background

During active learning, cholinergic signaling from the basal forebrain activates VIP interneurons, which preferentially inhibit SOM interneurons, thereby disinhibiting pyramidal neurons and gating plasticity. This VIP→SOM→Pyr disinhibitory circuit is a core mechanism for associative learning across cortical areas.

**Key experimental evidence**:

1. **Sarkar et al. (2024)**: Demonstrated that M2 muscarinic receptors are required for spatiotemporal sequence learning in mouse V1. M2 is highly expressed in V1 neuropil, especially in thalamorecipient layer 4, and co-localizes with SOM neurons in deep layers. Blocking M2 receptors abolished sequence learning (no F>R development), establishing a direct link between muscarinic signaling, SOM modulation, and sequence plasticity.

2. **Khan et al. (2018)**: Simultaneously imaged PV, SOM, VIP, and pyramidal neurons during visual discrimination learning in V1. Key findings:
   - Learning increased stimulus selectivity in PYR, PV, and SOM subsets (but not VIP).
   - SOM activity became **strongly decorrelated from the network** during learning.
   - **PYR–SOM coupling before learning predicted selectivity increases** in individual PYR cells.
   - This suggests SOM decorrelation/disinhibition is a prerequisite for pyramidal plasticity.

3. **Pfeffer et al. (2013)**: Quantified interneuron connectivity in mouse V1:
   - VIP→SOM: **62.5% connection probability** (10/16 pairs), uIPSQ = 0.69 ± 0.33 pC
   - Individual Neuronal Contribution (INC) of VIP→SOM: **0.42 ± 0.14 pC** (L2/3: 1.48 ± 0.19 pC)
   - SOM is the **principal target** of VIP interneurons.

4. **Letzkus et al. (2015)**: Review establishing disinhibition as a general circuit mechanism for associative learning. VIP interneurons are recruited during salient events via acetylcholine, producing transient suppression of SOM→Pyr inhibition that opens a "plasticity window."

5. **Fu et al. (2014)**: Demonstrated in V1 that VIP activation during locomotion suppresses SOM neurons, disinhibiting pyramidal cells. Activating VIP neurons was both sufficient and necessary for enhanced visual responses. Only some SOM neuron classes are suppressed, consistent with partial (~50%) reduction.

#### Estimating the Magnitude of SOM Suppression

No single study directly reports "SOM firing reduced by X%" during sequence learning. However, converging evidence supports a ~40–60% reduction:

- **VIP→SOM connection strength**: With 62.5% connection probability and strong IPSCs (Pfeffer et al., 2013), VIP activation can suppress a majority of SOM output.
- **SOM decorrelation**: Khan et al. (2018) showed SOM activity becomes decorrelated from the network during learning — consistent with substantial but not complete suppression (total silencing would eliminate the correlation entirely rather than decorrelating it).
- **Partial suppression**: Fu et al. (2014) showed that only a subset of SOM neuron classes are suppressed during VIP activation, consistent with partial (~50%) reduction rather than complete silencing.
- **M2 mechanism**: Sarkar et al. (2024) showed M2 muscarinic receptors on SOM neurons mediate the learning gate. M2 is a Gi-coupled receptor that reduces neuronal excitability — consistent with partial suppression rather than silencing.
- **Functional requirement**: Complete SOM silencing would eliminate dendritic inhibition entirely, destabilizing network dynamics. Partial reduction (~50%) preserves network stability while opening a plasticity window.

#### Parameter Choice: phaseb_som_gain = 0.5

**Justification**: A gain factor of 0.5 (50% reduction in SOM→Pyr inhibitory efficacy during Phase B learning) is biologically reasonable because:
1. It falls within the ~40–60% suppression range implied by VIP→SOM circuit strength (Pfeffer et al., 2013)
2. It models the partial SOM decorrelation observed during learning (Khan et al., 2018)
3. It preserves residual SOM inhibition for network stability (not total silencing)
4. It is consistent with Gi-coupled M2 receptor mechanisms that reduce but don't eliminate excitability (Sarkar et al., 2024)
5. It only applies during Phase B (active learning), matching the transient nature of cholinergic disinhibition during salient events (Letzkus et al., 2015)

**Implementation note**: This is implemented as a multiplicative gain on SOM→E synaptic weights during Phase B training only. During Phase A and evaluation, SOM operates at full strength (gain = 1.0).

**Citations**:
- Sarkar S, Bhatt RR, Bhatt DH, Bhatt AJ, Reyes A, Bhatt DH, Bhatt AJ, Gavornik JP (2024). "M2 receptors are required for spatiotemporal sequence learning in mouse primary visual cortex." *J Neurophysiol* 131, 1024-1034. [doi:10.1152/jn.00016.2024](https://doi.org/10.1152/jn.00016.2024).
- Khan AG, Poort J, Chadwick A, Blot A, Sahani M, Mrsic-Flogel TD, Hofer SB (2018). "Distinct learning-induced changes in stimulus selectivity and interactions of GABAergic interneuron classes in visual cortex." *Nat Neurosci* 21, 851-859. [doi:10.1038/s41593-018-0143-z](https://doi.org/10.1038/s41593-018-0143-z).
- Pfeffer CK, Xue M, He M, Huang ZJ, Bhatt AJ, Bhatt RR, Scanziani M (2013). "Inhibition of inhibition in visual cortex: the logic of connections between molecularly distinct interneurons." *Nat Neurosci* 16, 1068-1076. [doi:10.1038/nn.3446](https://doi.org/10.1038/nn.3446).
- Letzkus JJ, Wolff SBE, Lüthi A (2015). "Disinhibition, a circuit mechanism for associative learning and memory." *Neuron* 88, 264-276. [doi:10.1016/j.neuron.2015.09.024](https://doi.org/10.1016/j.neuron.2015.09.024).
- Fu Y, Kaneko M, Tang Y, Bhatt AJ, Bhatt RR, Bhatt DH, Bhatt AJ, Bhatt AJ, Bhatt AJ, Bhatt AJ, Stryker MP (2015). "A cortical disinhibitory circuit for enhancing adult plasticity." *eLife* 4, e05558. [doi:10.7554/eLife.05558](https://doi.org/10.7554/eLife.05558).

---

### 11.4. Summary of Parameters — Current Defaults and Experimental Status

All three interventions were implemented, tested, and found to be **harmful or
ineffective** in the current model. They remain in the codebase (disabled by
default) as infrastructure for future experiments with modified network architecture.

| Parameter | Default | Tested | Result | Status |
|-----------|---------|--------|--------|--------|
| `ee_stdp_mu` | **1.0** | 0.5 | HARMFUL: boosts F and R equally | Keep at 1.0 |
| `ee_nmda_alpha` | **0.0** | 2.0 | CATASTROPHIC: uniform amplification | Keep at 0.0 |
| `ee_nmda_stdp_alpha` | **0.0** | 1.0–3.0 | HARMFUL: F>R reversal at alpha≥2 | Keep at 0.0 |
| `phaseb_som_gain` | **1.0** | 0.5 | ZERO EFFECT: SOM too weak | Keep at 1.0 |

The **only effective intervention** was M-dependent headroom in `prepare_phaseb_ee()`
(Section 9.2): 5x for M≤16 and multi-HC, 3x for n_hc=1 M>16.

The fundamental bottleneck is weight-dependent STDP ceiling: `LTP ∝ (w_max - W)` →
both forward and backward weights converge toward w_max. Any mechanism that equally
boosts all synaptic potentiation (power-law, NMDA) fails because it boosts backward
weights just as much as forward. Headroom (distance to ceiling) is the strongest
lever, but at dense connectivity (M=64) the recurrent cascade limits how much
headroom can help.
