# Flappy Fly: design notes

Goal: a single fruit fly, simulated from the connectome, flies Flappy Bird, learns
from its failures, and shows its brain working while it plays. Build approach agreed
with Peter on 2026-09-25: hybrid learning (option C), built from scratch, one fly.

## 1. The brain (pipeline/build_brain.py)

Source: Janelia/Google **male CNS** connectome, Traced bodies only (165k). Files:
`body-annotations`, `body-neurotransmitters`, `connectome-weights` (body-to-body
synapse counts).

* Dropped: nerve cord intrinsic, sensory and motor neurons, other sensory modalities.
  The model is brain-only; descending neurons are its outputs.
* Edges with ≥ 3 synapses. Path finding uses "strong" edges (≥ 1% of the target's input).
* Kept (hops = 3):
  1. neurons within 3 strong hops of the eye (L1/L2/L3) **and** within 3 hops of a
     descending neuron or the mushroom body;
  2. neurons on ≤ 3-hop paths from MBONs to descending neurons;
  3. the mushroom body core (MBONs, PAM/PPL1 DANs, APL, DPM) and Kenyon cells within
     4 hops of the eye (the visual γd / αβp KCs);
  4. reachable descending neurons.
  Dead ends are pruned iteratively. Result: **79,992 neurons, 4,270,367 edges**.
* Signs: consensus / cell-type / per-body predicted transmitter. ACh +, GABA/Glu/histamine −,
  modulators 0. Dopamine acts only through plasticity.
* Removed axo-axonal contact classes KC→KC, KC→DAN, DAN→KC (they create runaway loops
  in point-neuron models; lesson from the FruitFlyBrain project).
* DAN→MBON compartment map: `C[d,m] = Σ_k w(d→k)·w(k→m)` over Kenyon cells, normalized
  per DAN. This decides which dopamine neuron teaches which MBON's synapses.

### Eye geometry
Only 15 columnar types carry hex coordinates. A linear fit of lamina soma positions to
(hex1, hex2) gives R² 0.99 for the dorsoventral axis: **hex1+hex2 runs ventral→dorsal,
hex2−hex1 runs anterior→posterior**, the same on both eyes. These map linearly (1st to 99th
percentile) to elevation −70…+70° and azimuth −15…160°.
Receptive-field centres of the other visual neurons are propagated through excitatory
synapses (synapse-weighted mean of the inputs' directions, 6 passes), with the spread
kept as a measure of field size.

## 2. Dynamics (web/js/engine.js)

Leaky integrate-and-fire with an exponential synaptic current, Shiu et al. 2024 values
(V_rest −52, V_th −45 mV, τ_m 20 ms, τ_syn 5 ms, refractory 2.2 ms, 0.275 mV per synapse),
dt 1 ms, exact integration, Gaussian membrane noise 0.5 mV/√ms. Event-driven spike
propagation over a CSR table. 80k neurons run at about 2.5× real time in Node (single core).

### Development (web/js/homeostasis.js, sim/develop.mjs)
With Shiu weights the visual signal dies within two synapses: optic lobe neurons are
graded, and the ≥3-synapse subnetwork loses weak inputs. Instead of fitting gains by hand,
the fly "grows up" watching flight scenes. Each neuron multiplicatively scales all of its
inputs toward a class set point (eye 12, optic 10, VPN 8, central 5, DN 6, KC 1.5, MBON 8,
DAN 3 Hz). Then the gains are frozen. Two lessons:
* With high membrane noise (1.4 mV/√ms), homeostasis *lowers* the gains of central
  neurons and lets noise drive them. Dropping the noise to 0.5 forced synaptic drive.
* A fast rate (η 0.25) oscillates; 40 s fast + 150 s slow (η 0.08, warm start) is stable.

## 3. The eye (web/js/eye.js)
The world is ray-cast in 3D from the fly's head for each column (5 rays, 2.5° acceptance).
Pipes are **pillars** (52 deep, 200 wide) with a horizontal slot, the ground is a striped
plane, and the sky is bright.
* First version: infinitely wide walls. The frontal field was always dark and the gap
  hardly visible; even the raw eye encoded the gap offset at R² ≈ 0.2. Pillars fixed it.
* Photoreceptors: low-pass 8 ms, adaptation 300 ms. L1/L2 respond to negative contrast
  (transient) plus a little darkness. L3 responds to sustained darkness. All three
  depolarize to dark, as real lamina cells do. The inhibitory L1→Mi1 synapse then makes
  the ON pathway.

## 4. Where the task information lives (diagnostics, pipeline/decode.py)
Ridge decoding with blocked 5-fold CV, recorded under a scripted/random pilot:

| Representation | gap offset R² | altitude R² | pipe distance R² |
|---|---|---|---|
| Raw rendered image (1,768 directions, no neurons) | **0.90** | 0.98 | 0.96 |
| Lamina neurons, 128 PCs | 0.73 | 0.82 | 0.91 |
| All columnar visual neurons, 160 PCs | 0.66 | 0.70 | 0.90 |
| Coarse pools (type × 8 elevation × 3 azimuth bins) | 0.45 | 0.45 | 0.83 |
| Visual projection neurons (pools) | ≈ 0 | ≈ 0 | 0.72 |
| Descending neurons, Kenyon cells, MBONs, DNg02 | ≈ 0 | ≈ 0 | ≤ 0.4 |

**The connectome's central brain knows that a pipe is coming (looming), not where the
gap is.** Retinotopy is pooled away in the lobula output. The real fly's Kenyon cells
and descending neurons don't carry the variable Flappy Bird needs. Vertical speed is
weakly encoded everywhere (R² ≈ 0.1), so the readout also gets an efference copy of
the last wing beat, which real flies have.

## 5. Learning (web/js/agent.js, web/js/plasticity.js)
* Every 40 ms: flap or glide. Two readouts, Q(flap) and Q(glide), are linear in
  conjunctions of (64 lamina PCs + bias) × (6 Gaussian bins of time since the last flap)
  = 390 features. They are trained by SARSA(λ = 0.8, γ = 0.94, normalized step 0.1).
* **Dopamine = TD error.** δ is injected into PAM (δ > 0) or PPL1 (δ < 0) as current for
  120 ms. Their spikes gate plasticity at the 8,286 KC→MBON synapses:
  `Δf = η · DA_m · E_e`, where E_e is a KC-spike-triggered eligibility trace of the MBON
  rate deviation (node perturbation), DA_m is the compartment-weighted phasic DAN rate,
  and f recovers to 1 with τ 2 min.
* **Replay** between lives: 3,000 one-step Q-learning updates on a 30k-transition memory.
  This made the difference between no learning and learning within ~60 lives.
* Rewards: +1 per pipe, −1 per crash, plus potential-based shaping −|gap offset|/100
  ("innate attraction to the opening"). It leaves the optimal policy unchanged
  (Ng et al. 1999). Without it, learning fails at this budget.
* Biology-only mode: flap when the DNg02 population's z-score exceeds −0.2 (calibrated
  to a hovering flap rate). Only KC→MBON synapses learn.

### Dead ends worth remembering
* Actor-critic policy gradient: unstable and slow even on ground-truth features.
* Linear readout over 4k to 7k raw pools: no learning in 300 lives.
* A random KC-like expansion of PCs decoded worse than the PCs themselves.
* Temporal smoothing of PCs: no gain (the noise is representational, not temporal).
* zsh does not word-split `$var`: parameter sweeps silently passed NaN. Use `${=var}`.

## 6. Browser app
`web/js/worker.js` runs the fly in a module Web Worker and posts frames (~30 Hz) with
per-neuron spike counts. `brainview.js` draws all 80k somata as additive WebGL2 points
(frontal view, drag to rotate). The panels show the fly's-eye view (right eye), the
decision (Q values, DNg02 rate), the dopamine trace, per-MBON memory strength and the
learning curve.
