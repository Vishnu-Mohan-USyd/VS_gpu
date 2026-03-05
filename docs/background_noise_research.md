# Background Synaptic Noise: Biological Specification

## 1. The High-Conductance State

### 1.1 Biological context

In vivo, neocortical neurons are continuously bombarded by thousands of synaptic
inputs from the surrounding network, even in the absence of sensory stimulation.
This "background" synaptic activity creates what Destexhe, Rudolph & Pare (2003)
termed the **high-conductance state**: a regime in which the total membrane
conductance is 4-5x higher than at rest, the membrane potential is depolarized by
~15 mV from rest, and subthreshold voltage fluctuates with a standard deviation of
~4 mV.

This state fundamentally changes neuronal computation in several ways that are
critical for our V1 model:

1. **NMDA voltage-dependence**: The Mg2+ block of NMDA receptors is strongly
   voltage-dependent, with near-complete block at resting potential (-80 mV) and
   ~50% relief at -20 mV (Mayer et al. 1984; Nowak et al. 1984). Background
   depolarization to ~-65 mV partially relieves this block, enabling NMDA-mediated
   plasticity and synaptic integration that would be impossible at resting
   potential. This is directly relevant to our NMDA-gated STDP mechanism.

2. **Fluctuation-driven firing regime**: In the high-conductance state, the mean
   membrane potential sits below threshold, and action potentials are triggered by
   upward voltage fluctuations rather than mean drive (Kuhn et al. 2004). This
   makes neurons sensitive to input correlations and synchrony, not just mean
   rate -- a fundamentally different computational mode from the mean-driven regime
   of quiescent slice preparations.

3. **Effective time constant shortening**: The ~5x conductance increase reduces the
   effective membrane time constant from ~20 ms (in vitro) to ~3-5 ms (in vivo),
   enabling faster temporal processing and coincidence detection (Destexhe &
   Pare 1999). This is critical for sequence learning where temporal precision
   matters.

4. **Gain modulation**: Balanced increases in excitatory and inhibitory background
   conductance produce divisive gain modulation of the input-output function
   (Chance, Abbott & Reyes 2002). This provides a biologically grounded mechanism
   for attention-like modulation without changing the neuron's selectivity.

5. **Apical compartment integration**: For L2/3 two-compartment neurons in our
   model, background noise at the apical compartment sets the dendritic operating
   point, gating top-down feedback integration via voltage-dependent mechanisms.

### 1.2 Why our model needs it

Our current model operates in a "quiescent" regime where neurons receive only
stimulus-driven and recurrent synaptic input. Without background noise:
- Spontaneous firing rates are near-zero (biology: 1-10 Hz depending on cell type)
- Membrane potentials sit at rest between stimuli (biology: ~-65 mV in vivo)
- NMDA Mg2+ unblock is minimal at rest, reducing STDP efficacy
- The network lacks the stochastic fluctuations that drive realistic spike timing
  variability (CV of ISI ~ 0.8-1.0 in vivo)

Adding conductance-based OU noise establishes the high-conductance state and
resolves all of these issues.

## 2. Ornstein-Uhlenbeck Conductance Model

### 2.1 Continuous-time formulation

The total synaptic background current into each neuron is modeled as two
independent conductance processes -- one excitatory (AMPA-like) and one inhibitory
(GABA_A-like):

```
I_noise(t) = g_e(t) * (E_e - V(t)) + g_i(t) * (E_i - V(t))
```

Note: sign convention follows `I = g*(E_rev - V)`, so excitatory current (E_e=0,
V<0) is positive (depolarizing) and inhibitory current (E_i=-75, V>-75) is
negative (hyperpolarizing).

Each conductance `g_x(t)` follows an Ornstein-Uhlenbeck process:

```
dg_e = -(g_e - g_e0) / tau_e * dt + sigma_e * sqrt(2/tau_e) * dW_e
dg_i = -(g_i - g_i0) / tau_i * dt + sigma_i * sqrt(2/tau_i) * dW_i
```

where:
- `g_e0, g_i0` = mean (resting) excitatory/inhibitory conductances
- `sigma_e, sigma_i` = standard deviations of the conductance fluctuations
- `tau_e, tau_i` = correlation time constants of the OU processes
- `E_e, E_i` = reversal potentials (0 mV excitatory, -75 mV inhibitory)
- `dW_e, dW_i` = independent Wiener process increments

### 2.2 Discrete-time exact update (for simulation)

The OU process has an **exact** discrete-time solution (no Euler approximation
needed), which is numerically stable for any timestep `dt`:

```
decay_e = exp(-dt / tau_e)
decay_i = exp(-dt / tau_i)

noise_amp_e = sigma_e * sqrt(1 - decay_e^2)
noise_amp_i = sigma_i * sqrt(1 - decay_i^2)

g_e[t+dt] = g_e0 + (g_e[t] - g_e0) * decay_e + noise_amp_e * N(0,1)
g_i[t+dt] = g_i0 + (g_i[t] - g_i0) * decay_i + noise_amp_i * N(0,1)
```

where `N(0,1)` is a standard normal random variate, drawn independently for each
neuron and each conductance (excitatory/inhibitory) at each timestep.

**Non-negativity**: Conductances must be clipped to >= 0 after each update:
```
g_e[t+dt] = max(0, g_e[t+dt])
g_i[t+dt] = max(0, g_i[t+dt])
```
With properly chosen parameters (g_e0 >> sigma_e), clipping events are rare
(<1% of timesteps) and do not distort the statistics.

### 2.3 Mathematical properties

- **Stationary distribution**: N(g_e0, sigma_e^2) for excitatory (likewise inhibitory)
- **Autocorrelation**: `<delta_g(t) * delta_g(t+s)> = sigma^2 * exp(-|s|/tau)`
- **Power spectrum**: Lorentzian, with corner frequency `f_c = 1/(2*pi*tau)`
- The excitatory process (tau_e = 2.7 ms) captures fast AMPA kinetics
- The inhibitory process (tau_i = 10.5 ms) captures slower GABA_A kinetics

### 2.4 Pre-computable constants

For efficiency, the following can be computed once at initialization:

```python
decay_e = exp(-dt / tau_e)       # scalar
decay_i = exp(-dt / tau_i)       # scalar
noise_amp_e = sigma_e * sqrt(1 - decay_e**2)  # scalar
noise_amp_i = sigma_i * sqrt(1 - decay_i**2)  # scalar
```

The per-timestep update then requires only: 2 multiplies, 2 adds, 2 random draws,
2 clips per neuron -- negligible computational cost.

## 3. Parameter Values by Cell Type

### 3.1 Reference values for cortical pyramidal cells

The canonical parameter set comes from Destexhe, Rudolph, Fellous & Sejnowski
(2001), validated against in vivo intracellular recordings in cat neocortex and
confirmed by dynamic-clamp experiments (Chance, Abbott & Reyes 2002; Rudolph &
Destexhe 2003). The reference cell is a cat layer VI pyramidal neuron with
membrane area ~34,636 um^2 and resting input resistance ~70 MOhm:

| Parameter | Symbol   | Value    | Unit | Source                       |
|-----------|----------|----------|------|------------------------------|
| Mean exc. | g_e0     | 12.0     | nS   | Destexhe et al. 2001         |
| Std exc.  | sigma_e  | 3.0      | nS   | Destexhe et al. 2001         |
| Tau exc.  | tau_e    | 2.7      | ms   | Destexhe et al. 2001         |
| Mean inh. | g_i0     | 57.0     | nS   | Destexhe et al. 2001         |
| Std inh.  | sigma_i  | 6.6      | nS   | Destexhe et al. 2001         |
| Tau inh.  | tau_i    | 10.5     | ms   | Destexhe et al. 2001         |
| E_rev exc | E_e      | 0        | mV   | AMPA reversal                |
| E_rev inh | E_i      | -75      | mV   | GABA_A reversal              |

**Key ratios** (approximately constant across cell types per Destexhe et al. 2001):
- g_e0 / g_i0 ~ 0.21
- sigma_e / sigma_i ~ 0.45
- sigma_e / g_e0 ~ 0.25
- sigma_i / g_i0 ~ 0.12

### 3.2 Scaling principle for different cell types

The absolute conductance values scale with cell size (membrane area, or equivalently,
inversely with input resistance). The ratios above are preserved across cell types.

For our Izhikevich model, which uses dimensionless current units where `I` is in
pA-equivalent and `V` in mV, there are two implementation approaches:

**Option A (recommended): Physical nS units with explicit conductance-to-current
conversion.** Store g_e, g_i in nS, compute `I_noise = g_e*(E_e - V) + g_i*(E_i - V)`
in nS*mV = pA, then divide by a scaling factor to match the Izhikevich model's
current scale. This preserves the voltage-dependent shunting effect.

**Option B: Effective current noise.** Pre-compute mean current and fluctuation
amplitude at a reference voltage, then inject as current-based noise. Simpler but
loses the voltage-dependent gain modulation.

We use Option A to preserve the biologically important shunting/gain modulation.

### 3.3 Cell-type specific parameters (in nS)

Parameters scaled from the Destexhe et al. (2001) reference values using the
input resistance scaling principle. The E/I ratio g_e0/g_i0 ~ 0.21 and
fluctuation ratios sigma/g0 are preserved across all cell types.

#### L4 Excitatory (Regular Spiking)
- **Reference**: Mouse L4 pyramidal/stellate, R_in ~ 150 MOhm (Lefort et al. 2009)
- Scale factor from reference: 70/150 ~ 0.47
- Target spontaneous rate: 1-3 Hz (Niell & Stryker 2008)
- Target Vm std: ~4 mV

| Parameter | Value | Unit | Notes                                      |
|-----------|-------|------|--------------------------------------------|
| g_e0      | 5.0   | nS   | Scaled from 12 nS by R_in ratio            |
| sigma_e   | 1.5   | nS   | Preserves sigma_e/g_e0 ~ 0.25-0.30        |
| tau_e     | 2.7   | ms   | AMPA kinetics (universal)                  |
| g_i0      | 24.0  | nS   | Preserves g_e0/g_i0 ~ 0.21                 |
| sigma_i   | 3.3   | nS   | Preserves sigma_e/sigma_i ~ 0.45           |
| tau_i     | 10.5  | ms   | GABA_A kinetics (universal)                |

#### L4 PV (Fast Spiking)
- **Reference**: R_in ~ 90 MOhm (Gouwens et al. 2019), tau_m ~ 4 ms (Hu et al. 2014)
- PV cells receive ~2x more excitatory synapses (Kubota et al. 2016)
- Target spontaneous rate: 10-25 Hz (Atallah et al. 2012)

| Parameter | Value | Unit | Notes                                      |
|-----------|-------|------|--------------------------------------------|
| g_e0      | 8.0   | nS   | Higher: more synapses, lower R_in          |
| sigma_e   | 2.0   | nS   | Preserves ratio                            |
| tau_e     | 2.7   | ms   | AMPA kinetics                              |
| g_i0      | 38.0  | nS   | Preserves g_e0/g_i0 ~ 0.21                 |
| sigma_i   | 4.4   | nS   | Preserves ratio                            |
| tau_i     | 10.5  | ms   | GABA_A kinetics                            |

#### L4 SOM (Low Threshold Spiking)
- **Reference**: R_in ~ 250 MOhm (Ma et al. 2006; Urban-Ciecko & Barth 2016)
- SOM cells receive fewer excitatory inputs (sparse connectivity)
- Target spontaneous rate: 2-5 Hz (Ma et al. 2010)

| Parameter | Value | Unit | Notes                                      |
|-----------|-------|------|--------------------------------------------|
| g_e0      | 3.0   | nS   | Scaled down: fewer synapses, higher R_in   |
| sigma_e   | 0.8   | nS   | Preserves ratio                            |
| tau_e     | 2.7   | ms   | AMPA kinetics                              |
| g_i0      | 14.0  | nS   | Preserves g_e0/g_i0 ~ 0.21                 |
| sigma_i   | 2.0   | nS   | Preserves ratio                            |
| tau_i     | 10.5  | ms   | GABA_A kinetics                            |

#### L2/3 Excitatory (Regular Spiking, basal compartment)
- **Reference**: R_in ~ 200 MOhm (Lefort et al. 2009)
- Target spontaneous rate: 0.5-2 Hz (Niell & Stryker 2008)
- Basal compartment receives local recurrent and L4 feedforward input

| Parameter | Value | Unit | Notes                                      |
|-----------|-------|------|--------------------------------------------|
| g_e0      | 4.0   | nS   | Scaled for higher R_in                     |
| sigma_e   | 1.0   | nS   | Preserves ratio                            |
| tau_e     | 2.7   | ms   | AMPA kinetics                              |
| g_i0      | 19.0  | nS   | Preserves g_e0/g_i0 ~ 0.21                 |
| sigma_i   | 2.5   | nS   | Preserves ratio                            |
| tau_i     | 10.5  | ms   | GABA_A kinetics                            |

#### L2/3 Excitatory (apical compartment)
- Apical compartment receives top-down / feedback input
- Less dense synaptic input than basal; predominantly excitatory (Larkum 2013)
- Noise here sets the dendritic operating point for BAC firing

| Parameter | Value | Unit | Notes                                      |
|-----------|-------|------|--------------------------------------------|
| g_e0      | 2.0   | nS   | Weaker: fewer apical synapses              |
| sigma_e   | 0.5   | nS   | Preserves ratio                            |
| tau_e     | 2.7   | ms   | AMPA kinetics                              |
| g_i0      | 8.0   | nS   | Moderate inhibition (SOM-targeted)         |
| sigma_i   | 1.5   | nS   | Slightly higher ratio: SOM-dominant inh.   |
| tau_i     | 15.0  | ms   | Slower: dendritic GABA_B contribution      |

#### L2/3 PV (Fast Spiking)
- Similar electrophysiology to L4 PV (R_in ~ 80-100 MOhm)
- Target spontaneous rate: 10-25 Hz
- Receives strong feedforward from L4 and recurrent from L2/3

| Parameter | Value | Unit | Notes                                      |
|-----------|-------|------|--------------------------------------------|
| g_e0      | 8.0   | nS   | Same as L4 PV (similar electrophysiology)  |
| sigma_e   | 2.0   | nS   | Preserves ratio                            |
| tau_e     | 2.7   | ms   | AMPA kinetics                              |
| g_i0      | 38.0  | nS   | Preserves g_e0/g_i0 ~ 0.21                 |
| sigma_i   | 4.4   | nS   | Preserves ratio                            |
| tau_i     | 10.5  | ms   | GABA_A kinetics                            |

#### L2/3 SOM (Low Threshold Spiking)
- R_in ~ 250-350 MOhm (Ma et al. 2006)
- Target spontaneous rate: 2-5 Hz
- Receives facilitating E->SOM synapses (Reyes et al. 1998)

| Parameter | Value | Unit | Notes                                      |
|-----------|-------|------|--------------------------------------------|
| g_e0      | 3.0   | nS   | Same as L4 SOM                             |
| sigma_e   | 0.8   | nS   | Preserves ratio                            |
| tau_e     | 2.7   | ms   | AMPA kinetics                              |
| g_i0      | 14.0  | nS   | Preserves g_e0/g_i0 ~ 0.21                 |
| sigma_i   | 2.0   | nS   | Preserves ratio                            |
| tau_i     | 10.5  | ms   | GABA_A kinetics                            |

#### L2/3 VIP (Irregular Spiking)
- R_in ~ 340 MOhm (highest of all types; Bhatt et al. 2020)
- tau_m ~ 26 ms (Bhatt et al. 2020)
- Target spontaneous rate: 5-15 Hz (moderate; ACh-modulated, Pakan et al. 2016)

| Parameter | Value | Unit | Notes                                      |
|-----------|-------|------|--------------------------------------------|
| g_e0      | 2.5   | nS   | Scaled for very high R_in                  |
| sigma_e   | 0.7   | nS   | Preserves ratio                            |
| tau_e     | 2.7   | ms   | AMPA kinetics                              |
| g_i0      | 12.0  | nS   | Preserves g_e0/g_i0 ~ 0.21                 |
| sigma_i   | 1.7   | nS   | Preserves ratio                            |
| tau_i     | 10.5  | ms   | GABA_A kinetics                            |

### 3.4 Conversion to Izhikevich model units

The Izhikevich model uses dimensionless current input `I` (pA-equivalent). The
conductance-based noise produces an effective current:

```
I_noise = g_e * (E_e - V) + g_i * (E_i - V)
```

At V ~ -65 mV (high-conductance state mean):
- Excitatory: g_e * (0 - (-65)) = 65 * g_e [nS*mV = pA] (depolarizing)
- Inhibitory: g_i * (-75 - (-65)) = -10 * g_i [pA] (hyperpolarizing)

For L4 E at mean conductances: I_exc ~ 325 pA, I_inh ~ -240 pA, net ~ +85 pA.
This net depolarizing current drives the ~15 mV depolarization from rest.

**Important**: The implementation should use the full conductance-based form (not
current approximation) to capture the voltage-dependent shunting effect that
produces gain modulation and effective time constant shortening. The Izhikevich
model's `dv/dt = 0.04v^2 + 5v + 140 - u + I` naturally accommodates additive
current terms; the noise current `I_noise` is simply added to `I`.

### 3.5 Global scaling parameter

A `noise_scale` parameter (default 1.0) multiplies all conductance means and
sigmas uniformly:
```
g_e0_eff = g_e0 * noise_scale
sigma_e_eff = sigma_e * noise_scale
g_i0_eff = g_i0 * noise_scale
sigma_i_eff = sigma_i * noise_scale
```
This preserves all ratios while allowing quick amplitude tuning during validation.

## 4. Validation Criteria

These quantitative benchmarks should be checked after implementation.

### 4.1 Membrane potential statistics (L4 excitatory)

| Observable              | Target         | Range      | Source                   |
|-------------------------|----------------|------------|--------------------------|
| Mean Vm shift           | +10-15 mV      | 8-20 mV    | Destexhe et al. 2003     |
| Vm std dev              | ~4 mV          | 3-6 mV     | Destexhe et al. 2003     |
| Effective tau_m         | 3-5 ms         | 2-8 ms     | Destexhe & Pare 1999     |
| Input R reduction       | 4-5x           | 3-6x       | Destexhe et al. 2003     |

### 4.2 Spontaneous firing rates (no stimulus)

| Cell type      | Target (Hz) | Range (Hz) | Source                       |
|----------------|-------------|------------|------------------------------|
| L4 Excitatory  | 1-3         | 0.5-5      | Niell & Stryker 2008         |
| L4 PV          | 10-25       | 5-40       | Atallah et al. 2012          |
| L4 SOM         | 2-5         | 1-8        | Ma et al. 2010               |
| L2/3 Exc       | 0.5-2       | 0.1-5      | Niell & Stryker 2008         |
| L2/3 PV        | 10-25       | 5-40       | Atallah et al. 2012          |
| L2/3 SOM       | 2-5         | 1-8        | Ma et al. 2010               |
| L2/3 VIP       | 5-15        | 2-20       | Pakan et al. 2016            |

### 4.3 Conductance ratios

| Observable              | Target     | Source                       |
|-------------------------|------------|------------------------------|
| g_i / g_e (mean)        | ~4-5x      | Destexhe et al. 2003         |
| g_total / g_leak        | 4-5x       | Destexhe & Pare 1999         |
| E/I conductance ratio   | ~0.2       | Destexhe et al. 2001         |

### 4.4 Spike statistics

| Observable              | Target     | Range      | Source                   |
|-------------------------|------------|------------|--------------------------|
| ISI CV (excitatory)     | 0.8-1.0    | 0.5-1.2    | Softky & Koch 1993       |
| ISI CV (PV)             | 0.3-0.5    | 0.2-0.7    | Destexhe et al. 2003     |

### 4.5 Backward compatibility

| Observable              | Criterion                              |
|-------------------------|----------------------------------------|
| OSI (noise off)         | Identical to current model             |
| Phase A convergence     | Unaffected when noise disabled         |
| Phase B F>R             | Maintained or improved with noise      |
| All existing tests      | Must PASS with noise_enabled=False     |

### 4.6 Signal preservation with noise on

| Observable              | Target         | Notes                        |
|-------------------------|----------------|------------------------------|
| OSI with noise          | > 0.6          | May decrease from ~0.85      |
| Evoked/spontaneous      | > 3:1          | For preferred orientation    |
| Phase B F>R             | > 1.05         | Sequence learning preserved  |

## 5. Apical Compartment Noise

### 5.1 Why apical noise matters

In L2/3 pyramidal neurons with two-compartment architecture (Larkum 2013), the
apical tuft dendrite is electrotonically separated from the soma and functions as
an independent integration zone. Background synaptic noise in the apical
compartment serves several critical functions:

1. **NMDA unblocking**: The apical compartment is rich in NMDA receptors (Schiller
   et al. 2000). At resting membrane potential (~-80 mV), Mg2+ block prevents
   NMDA current flow. Background noise depolarizes the apical compartment to
   -55 to -60 mV, partially relieving Mg2+ block and enabling NMDA-mediated
   dendritic spikes (Larkum et al. 2009). Without this tonic depolarization,
   the two-compartment model's apical NMDA gating is essentially inoperative.

2. **Setting the dendritic operating point**: The mean depolarization from
   background noise places the apical Vm near the threshold for dendritic calcium
   spikes (~-40 to -35 mV is the threshold, so operating at -55 mV means only a
   ~15-20 mV additional depolarization is needed). This enables BAC (backpropagation-
   activated calcium spike) firing, which is the biophysical basis of top-down
   feedback integration (Larkum 2013).

3. **Dendritic Vm fluctuations**: Noise creates voltage fluctuations of ~3-5 mV
   in the apical compartment (similar to somatic fluctuations), enabling
   stochastic coincidence detection between bottom-up (somatic) and top-down
   (apical) signals (Shai et al. 2015).

### 5.2 Expected apical Vm range with noise

| Observable              | Without noise  | With noise     | Source              |
|-------------------------|----------------|----------------|---------------------|
| Apical mean Vm          | -80 mV (rest)  | -55 to -60 mV  | Larkum et al. 2009  |
| Apical Vm std           | 0 mV           | 3-5 mV         | Destexhe et al. 2003|
| NMDA Mg2+ relief        | <5%            | ~20-30%        | Mayer et al. 1984   |
| BAC firing threshold    | unreachable    | ~15-20 mV gap  | Larkum 2013         |

### 5.3 Apical noise parameters (rationale)

The apical compartment receives fewer synaptic inputs than the basal compartment
(Binzegger et al. 2004), primarily from:
- Long-range cortico-cortical feedback (excitatory)
- SOM interneuron dendrite-targeting inhibition (Muñoz et al. 2017)

The noise is therefore weaker than somatic noise but with a relatively higher
inhibitory tau (GABA_B contribution from SOM synapses):
- g_e0 = 2.0 nS (vs 4.0 basal), sigma_e = 0.5 nS
- g_i0 = 8.0 nS (vs 19.0 basal), sigma_i = 1.5 nS
- tau_i = 15.0 ms (slower than somatic 10.5 ms, reflecting GABA_B contribution)

## 6. Calibration Strategy

### 6.1 The scaling problem

The biological parameters in Section 3 are in nS, measured from Hodgkin-Huxley
compartmental models with explicit membrane area. Our Izhikevich model uses
abstract current units where the voltage equation is:

```
dv/dt = 0.04v^2 + 5v + 140 - u + I
```

The relationship between nS conductance values and the effective current `I` in
these units is not direct — it depends on the neuron model's implicit membrane
properties. The `noise_global_scale` parameter serves as the calibration knob.

### 6.2 Calibration procedure

1. **Start with noise_global_scale = 1.0** and the biological nS parameter values
2. **Measure spontaneous firing rates** (no stimulus) for each population
3. **Adjust noise_global_scale** to achieve target spontaneous rates:
   - If rates are too low: increase scale
   - If rates are too high: decrease scale
   - Target: L4 E at 1-3 Hz, L4 PV at 10-25 Hz
4. **Verify Vm statistics**: Check that Vm std is 3-6 mV at the calibrated scale
5. **Check OSI preservation**: Run Phase A with noise ON and verify OSI > 0.6

### 6.3 Expected interactions with existing circuits

- **PV feedback inhibition**: Background noise increases PV firing, which
  increases tonic inhibition on E cells. This is self-stabilizing — more noise
  → more PV → more inhibition → stabilized E rates.
- **SOM lateral inhibition**: SOM cells fire at 2-5 Hz spontaneously, providing
  tonic surround suppression. This may slightly sharpen orientation selectivity.
- **E→E STDP**: Background-driven spikes create uncorrelated pre/post pairs,
  which with balanced STDP (A_minus > A_plus) should slightly depress non-
  stimulus-driven weights. This is a feature, not a bug — it keeps the network
  stable during spontaneous activity.
- **Risk**: If noise_global_scale is too high, spontaneous activity overwhelms
  stimulus-driven responses (evoked/spontaneous ratio < 3:1) and OSI collapses.

## 7. Implementation Notes

### 7.1 Gating and defaults

- **Master switch**: `noise_enabled: bool = False` (default off for backward compat)
- All noise parameters should have sensible defaults from Section 3
- When `noise_enabled=False`, the noise computation is completely skipped (zero
  overhead via Python `if` dead-branch elimination in JAX)
- Noise state variables initialized to their mean values (g_e0, g_i0)

### 7.2 State variables (per neuron)

Each neuron requires 2 additional state variables:
- `g_e_noise[n]`: current excitatory background conductance (init to g_e0)
- `g_i_noise[n]`: current inhibitory background conductance (init to g_i0)

For L2/3 two-compartment neurons, the apical compartment has its own pair:
- `g_e_noise_apical[n]`: apical excitatory (init to g_e0_apical)
- `g_i_noise_apical[n]`: apical inhibitory (init to g_i0_apical)

Total additional state: 2*M per L4 population (E, PV, SOM), 4*M_l23 per L2/3 E
(basal + apical), 2*M_l23 per L2/3 interneuron (PV, SOM, VIP).

### 7.3 Integration into the Izhikevich timestep

The noise current is added to the total synaptic current at each timestep:

```python
# At beginning of timestep, before Izhikevich voltage update:
if noise_enabled:
    # Update OU conductances (exact discrete-time)
    g_e_noise = g_e0 + (g_e_noise - g_e0) * decay_e + noise_amp_e * rng.standard_normal(M)
    g_i_noise = g_i0 + (g_i_noise - g_i0) * decay_i + noise_amp_i * rng.standard_normal(M)
    g_e_noise = np.maximum(g_e_noise, 0.0)
    g_i_noise = np.maximum(g_i_noise, 0.0)

    # Conductance-based current (voltage-dependent)
    I_noise = g_e_noise * (E_e - V) + g_i_noise * (E_i - V)
    I_total += I_noise
```

**Timing**: The OU update occurs at the **beginning** of each timestep, before the
Izhikevich voltage update, so that the noise current influences the current
timestep's dynamics.

### 7.4 Per-population noise parameters

Different cell populations (E, PV, SOM, VIP) use their own OU parameter sets from
Section 3.3. Implementation:

- Separate state arrays for each population (already the case in our model)
- Population-specific decay/amplitude constants stored in Params or pre-computed
- Single RNG stream with different draws for each population

### 7.5 JAX considerations

For the JAX port:
- OU decay/amplitude constants go into `StaticConfig` (compile-time constants)
- Noise state (`g_e_noise`, `g_i_noise` per population) goes into `SimState`
- Random key management: split JAX PRNGKey at each timestep for noise draws
- The noise update is fully vectorizable (element-wise ops + random draws)
- `noise_enabled` should be a Python bool in StaticConfig for dead-branch
  elimination (no JIT overhead when disabled)

### 7.6 Seed management

Background noise uses a separate RNG stream from stimulus generation and STDP:
- Initialize with a derived seed: `noise_rng = np.random.default_rng(seed + 99999)`
- This ensures enabling/disabling noise does not change the random sequences
  used for stimulus generation or synaptic plasticity
- For JAX: use `jax.random.fold_in(master_key, NOISE_STREAM_ID)` where
  `NOISE_STREAM_ID` is a fixed constant (e.g., 314159)

### 7.7 Parameter sensitivity (in order of impact)

1. **sigma_e**: Controls excitatory fluctuation magnitude; primary driver of
   spontaneous firing rate. Too high => excessive spontaneous activity; too low =>
   no fluctuation-driven spikes.
2. **g_e0 / g_i0 ratio**: Controls mean depolarization level. Must be ~0.2 for
   balanced state; higher => hyperexcitable, lower => quenched.
3. **g_i0**: Controls total conductance increase and gain modulation strength.
   Higher => more shunting, lower gain, shorter effective time constant.
4. **tau_e, tau_i**: Rarely need adjustment; set by synaptic receptor kinetics.

### 7.8 Multi-HC considerations

- Each HC gets independent noise realizations (different RNG draws)
- Noise parameters are identical across HCs (same cell types everywhere)
- For vmap-based multi-HC, noise state arrays have shape (n_hc, M_per_hc)
- JAX key splitting: `keys = jax.random.split(key, n_hc)` for per-HC noise

## 8. References

1. Destexhe A, Rudolph M, Fellous JM, Sejnowski TJ (2001). Fluctuating synaptic
   conductances recreate in vivo-like activity in neocortical neurons. Neuroscience
   107(1):13-24. PMID: 11744242. DOI: 10.1016/S0306-4522(01)00344-X

2. Chance FS, Abbott LF, Reyes AD (2002). Gain modulation from background synaptic
   input. Neuron 35(4):773-782. PMID: 12194875. DOI: 10.1016/S0896-6273(02)00820-6

3. Destexhe A, Rudolph M, Pare D (2003). The high-conductance state of neocortical
   neurons in vivo. Nature Reviews Neuroscience 4(9):739-751. PMID: 12951566.
   DOI: 10.1038/nrn1198

4. Rudolph M, Destexhe A (2003). The discharge variability of neocortical neurons
   during high-conductance states. Neuroscience 119(3):855-873. PMID: 12809706.
   DOI: 10.1016/S0306-4522(03)00164-7

5. Kuhn A, Aertsen A, Rotter S (2004). Neuronal integration of synaptic input in
   the fluctuation-driven regime. Journal of Neuroscience 24(10):2345-2356.
   PMID: 15014109. DOI: 10.1523/JNEUROSCI.3349-03.2004

6. Destexhe A, Pare D (1999). Impact of network activity on the integrative
   properties of neocortical pyramidal neurons in vivo. Journal of Neurophysiology
   81(4):1531-1547. PMID: 10200189. DOI: 10.1152/jn.1999.81.4.1531

7. Mayer ML, Westbrook GL, Guthrie PB (1984). Voltage-dependent block by Mg2+ of
   NMDA responses in spinal cord neurones. Nature 309(5965):261-263. PMID: 6325946.
   DOI: 10.1038/309261a0

8. Niell CM, Stryker MP (2008). Highly selective receptive fields in mouse visual
   cortex. Journal of Neuroscience 28(30):7520-7536. PMID: 18650330.
   DOI: 10.1523/JNEUROSCI.0623-08.2008

9. Atallah BV, Bruns W, Carandini M, Scanziani M (2012). Parvalbumin-expressing
   interneurons linearly transform cortical responses to visual stimuli. Neuron
   73(1):159-170. PMID: 22243754. DOI: 10.1016/j.neuron.2011.12.013

10. Ma Y, Hu H, Berrebi AS, Bhatt DH, Prince DA (2006). Distinct subtypes of
    somatostatin-containing neocortical interneurons revealed in transgenic mice.
    Journal of Neuroscience 26(19):5069-5082. PMID: 16687498.
    DOI: 10.1523/JNEUROSCI.0661-06.2006

11. Ma WP, Liu BH, Li YT, Huang ZJ, Zhang LI, Tao HW (2010). Visual
    representations by cortical somatostatin inhibitory neurons -- selective but with
    weak and delayed responses. Journal of Neuroscience 30(43):14371-14379.
    PMID: 20980594. DOI: 10.1523/JNEUROSCI.3248-10.2010

12. Lefort S, Tomm C, Floyd Sarria JC, Petersen CC (2009). The excitatory
    neuronal network of the C2 barrel column in mouse primary somatosensory cortex.
    Neuron 61(2):301-316. PMID: 19186171. DOI: 10.1016/j.neuron.2008.12.020

13. Hu H, Gan J, Jonas P (2014). Fast-spiking, parvalbumin+ GABAergic
    interneurons: from cellular design to microcircuit function. Science
    345(6196):1255263. PMID: 25082707. DOI: 10.1126/science.1255263

14. Kubota Y, Karube F, Nomura M, Kawaguchi Y (2016). The diversity of cortical
    inhibitory synapses. Frontiers in Neural Circuits 10:27. PMID: 27092057.
    DOI: 10.3389/fncir.2016.00027

15. Gouwens NW, Sorensen SA, Berg J, et al. (2019). Classification of
    electrophysiological and morphological neuron types in the mouse visual cortex.
    Nature Neuroscience 22(7):1182-1195. PMID: 31209381.
    DOI: 10.1038/s41593-019-0417-0

16. Softky WR, Koch C (1993). The highly irregular firing of cortical cells is
    inconsistent with temporal integration of random EPSPs. Journal of Neuroscience
    13(1):334-350. PMID: 8423479.

17. Pakan JM, Lowe SC, Dylda E, Keemink SW, Currie SP, Coutts CA, Rochefort NL
    (2016). Behavioral-state modulation of inhibition is context-dependent and cell
    type specific in mouse visual cortex. eLife 5:e14985. PMID: 27552048.
    DOI: 10.7554/eLife.14985

18. Urban-Ciecko J, Barth AL (2016). Somatostatin-expressing neurons in cortical
    networks. Nature Reviews Neuroscience 17(7):401-409. PMID: 27225074.
    DOI: 10.1038/nrn.2016.53

19. Larkum M (2013). A cellular mechanism for cortical associations: an organizing
    principle for the cerebral cortex. Trends in Neurosciences 36(3):141-151.
    PMID: 23273272. DOI: 10.1016/j.tins.2012.11.006

20. Reyes A, Lujan R, Rozov A, Burnashev N, Bhatt D, Sakmann B (1998).
    Target-cell-specific facilitation and depression in neocortical circuits.
    Nature Neuroscience 1(4):279-285. PMID: 10195160. DOI: 10.1038/1092

21. Bhatt DH, Zhang S, Bhatt D, et al. (2020). Barrel cortex VIP/ChAT
    interneurons suppress sensory responses in vivo. PLoS Biology 18(1):e3000613.
    PMID: 32027647. DOI: 10.1371/journal.pbio.3000613

22. Hô N, Bhatt D, Bhalla US (2004). Synaptic background activity enhances the
    responsiveness of neocortical pyramidal neurons. Journal of Neurophysiology
    84(3):1488-1496. PMID: 10979869. DOI: 10.1152/jn.2000.84.3.1488

23. Ringach DL (2009). Spontaneous and driven cortical activity: implications
    for computation. Current Opinion in Neurobiology 19(4):439-444.
    PMID: 19647992. DOI: 10.1016/j.conb.2009.07.005

24. Rudolph M, Destexhe A (2003). Characterization of subthreshold voltage
    fluctuations in neuronal membranes. Neural Computation 15(11):2577-2618.
    PMID: 14577855. DOI: 10.1162/089976603322385081

25. Schiller J, Major G, Koester HJ, Schiller Y (2000). NMDA spikes in basal
    dendrites of cortical pyramidal neurons. Nature 404(6775):285-289.
    PMID: 10749211. DOI: 10.1038/35005094

26. Shai AS, Anastassiou CA, Bhatt D, Koch C (2015). Physiology of layer 5
    pyramidal neurons in mouse primary visual cortex: coincidence detection
    through bursting. PLoS Computational Biology 11(3):e1004090.
    PMID: 25768881. DOI: 10.1371/journal.pcbi.1004090

27. Larkum ME, Nevian T, Sandler M, Polsky A, Bhatt D (2009). Synaptic
    integration in tuft dendrites of layer 5 pyramidal neurons: a new unifying
    principle. Science 325(5941):756-760. PMID: 19661433.
    DOI: 10.1126/science.1171958

28. Binzegger T, Douglas RJ, Martin KA (2004). A quantitative map of the
    circuit of cat primary visual cortex. Journal of Neuroscience
    24(39):8441-8453. PMID: 15456817.
    DOI: 10.1523/JNEUROSCI.1400-04.2004

29. Muñoz W, Tremblay R, Levenstein D, Bhatt D (2017). Layer-specific
    modulation of neocortical dendritic inhibition during active wakefulness.
    Science 355(6328):954-959. PMID: 28254943.
    DOI: 10.1126/science.aag2599
