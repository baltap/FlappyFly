# Results (2026-09-25)

All numbers are from `sim/train.mjs` (learning curves) and `sim/evaluate.mjs` (frozen
flies: learning, plasticity and exploration off). Pipes use 150 px gaps with up to 60 px
jumps between pipes ("training" difficulty) unless stated otherwise. Lives are capped at
120 s (79 pipes).

## Can the fly learn from its failures?

Yes, in hybrid mode. Mean pipes passed per life while learning:

| Run | lives 1–50 | 51–100 | 101–200 | 201–300 | 301–400 | best |
|---|---|---|---|---|---|---|
| **64 PCs + replay, seed 3 (shipped fly)** | 2.1 | 3.1 | 3.5 | 6.9 | 6.7 | 35 |
| 64 PCs + replay, seed 2 | 0.6 | 1.5 | 2.1 | 4.2 | 5.6 | 28 |
| 64 PCs + replay, seed 1 | 2.0 | 3.0 | 3.5 | 3.6 | 4.8 | 20 |
| 96 PCs + replay | 0.9 | 2.0 | 2.6 | 4.1 | 5.0 | 21 |
| 128 PCs + replay | 1.4 | 2.1 | 2.5 | 5.7 | 5.6 | 30 |
| 64 PCs + replay, **no shaping** | 0.6 | 0.7 | 1.8 | 2.7 | 3.3 | 25 |
| 64 PCs, no replay | 0.3 | 0.8 | 1.2 | 1.6 | – | 17 |
| **Biology only** (DNg02 decides, KC→MBON learns) | 0.02 | 0.00 | – | – | – | 1 |
| Biology only, no plasticity | 0.00 | 0.00 | – | – | – | 0 |

A newborn fly passes almost nothing in its first lives. Every hybrid run improves
steadily over a few hundred lives, across seeds and readout sizes.

## The trained fly, frozen (runs/eval, 30 lives each)

| Condition | mean | median | best | lives with 0 |
|---|---|---|---|---|
| Shipped fly (checkpoint at life ~300) | **36.6** | 33 | 79 (cap, 6×) | 0% |
| Same fly, unseen pipe layouts (seed 99) | **34.8** | 27 | 79 (cap, 7×) | 0% |
| Same fly, **blind** (lamina input cut) | 0 | 0 | 0 | 100% |
| Same fly, original difficulty (100 px gaps, 140 px jumps) | 0.5 | 0 | 3 | 67% |
| Other checkpoints: 64 PCs s1 / s2, 96, 128, no shaping | 8.8 / 6.3 / 6.0 / 5.3 / 2.0 | 7 / 4 / 4 / 3 / 2 | 23 / 34 / 24 / 27 / 6 | |

* The blind ablation shows the decisions come from what the connectome's eye sees.
* Frozen checkpoints can beat the same fly's continued online learning (seed 3 was
  averaging ~5 while still learning at life 450). Continued learning with ε = 0.01 and
  replay keeps shifting the readout. That's why the app's trained fly starts with learning off.
* Seed-to-seed spread is large (frozen means 6–37). Seed 3 is the best of five
  comparable runs, not a typical one.
* The fly does **not** handle the original game's tighter pipes. The lamina components
  locate the gap to roughly ±30 px, about the whole margin a 100 px gap leaves.

## What the brain itself knows (decoding, docs/DESIGN.md §4)

* Eye and optic lobe: gap position, altitude and pipe distance are all decodable.
* Visual projection neurons: pipe distance (looming) yes, gap position no.
* Central brain, Kenyon cells, MBONs, descending neurons, DNg02: essentially nothing
  about gap position. DNg02 fires at a median 0.2 Hz in flight.

So the honest answer to "can the connectome learn Flappy Bird by itself?" is **no**: the
information the task needs doesn't reach the fly's learning and motor centres in this
model. The connectome's eye and optic lobe provide it. A learned premotor readout (the
part the brain-only connectome lacks) turns it into skill, trained by the same dopamine
signal that drives the mushroom body.
