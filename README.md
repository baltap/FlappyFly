# Flappy Fly

A fruit fly brain plays Flappy Bird and learns from its crashes.

The brain is a spiking model built from the **Janelia/Google male CNS connectome**
(166k neurons, 125M synapses). It keeps the 79,992 neurons that sit on short paths
from the eyes to the descending neurons, plus the mushroom body learning circuit.
The game is rendered in 3D onto the fly's compound eye. Crashes and passed pipes
become dopamine, and dopamine changes synapses. A live view shows every neuron firing.

```
eye (4,363 lamina columns) → optic lobe → visual projection neurons → central brain
      → mushroom body (KC → MBON, dopamine-gated) → descending neurons (DNg02 = wing power)
```

## Run it

```bash
npm run serve        # http://localhost:8350
```

**Results:** a newborn fly passes almost no pipes. After ~300 lives of learning from its
crashes, the shipped fly averages **36.6 pipes per life** with learning frozen (median 33,
6 of 30 lives hit the 2-minute cap), and 34.8 on pipe layouts it has never seen. The same
fly with its eyes cut scores 0. Biology-only control: 0. It does not yet cope with the
original game's 100 px gaps. Details in [docs/RESULTS.md](docs/RESULTS.md).

Everything runs in the browser (a Web Worker simulates the brain at ~1 ms steps).
Pick **Trained fly** to watch a fly that has learned, or **Newborn fly** to watch
one learn from scratch.

## Rebuild from the connectome

Needs the male CNS feather files (`body-annotations`, `body-neurotransmitters`,
`connectome-weights`). They're cached in the FruitFlyBrain project by default; pass
`--data DIR` otherwise.

```bash
uv run python pipeline/build_brain.py          # web/data/brain.bin + brain.json
node sim/develop.mjs 40 0.25                   # development: homeostatic scaling …
node sim/develop.mjs 150 0.08 warm             # … then slow refinement → gains.f32
node sim/record_neurons.mjs 300                # record retinotopic activity in flight
uv run python pipeline/develop_npca.py lam 128 L1,L2,L3    # visual components
node sim/train.mjs --mode hybrid --episodes 300 --pcaK 64 --seed 3 --out runs/fly.json
node sim/evaluate.mjs runs/fly.json --lives 30          # add --blind for the ablation
cp runs/fly.json web/data/pretrained.json
npm test
```

## What is biology and what is not

| Part | Source |
|---|---|
| Which neurons exist and who connects to whom | Connectome (Traced bodies, ≥3 synapses) |
| Excitatory or inhibitory | Predicted neurotransmitter (ACh +, GABA/Glu −) |
| Neuron dynamics | Leaky integrate-and-fire, Shiu et al. 2024 parameters |
| Viewing direction of each eye column | Hex column coordinates, fitted to lamina somata |
| Input gain of each neuron | Development: homeostatic synaptic scaling, then frozen |
| Learning in the brain | Dopamine (PAM reward / PPL1 punishment) gates KC → MBON plasticity |
| Flap decision, biology-only mode | DNg02 population burst (wing stroke amplitude, Namiki et al. 2022) |
| Flap decision, hybrid mode | **Not connectome:** learned readout standing in for the missing nerve-cord premotor circuits |
| Readout input | Principal components of lamina activity (developmental PCA) + efference copy of the last wing beat |
| Extra reward for approaching the gap | **Teacher signal** (potential-based shaping); the fly itself only sees |

See [docs/DESIGN.md](docs/DESIGN.md) for the reasoning and [docs/RESULTS.md](docs/RESULTS.md)
for the measured numbers, including what did not work.
