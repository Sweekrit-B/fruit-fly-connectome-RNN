# Fruit Fly Motion Detection

A biologically-constrained recurrent neural network that learns to detect the direction of motion (left vs. right), built from the real synaptic connectome of the *Drosophila* (fruit fly) visual system rather than an arbitrary architecture.

The model doesn't process real video. It's a reduced, 1D model of a single strip of the fly's compound eye — 16 photoreceptor positions in a row, evolving over time — and the question the project asks is: **if you constrain a small recurrent network's connectivity to match what's actually wired up in the fly brain, can it learn to be a direction-selective motion detector, the way real T4/T5 neurons are?**

## Biological background

The fly's visual motion-detection circuit is one of the best-characterized neural circuits in any animal, and this project models a simplified version of its core feed-forward + lateral pathway:

```
R1-R6 (photoreceptors)
   -> L1, L2 (lamina neurons)
       -> Mi1, Tm3 (medulla, "ON" pathway - tuned to brightness increases)
       -> Tm1, Tm2 (medulla, "OFF" pathway - tuned to brightness decreases)
           -> T4 (lobula plate - detects moving brightness increases)
           -> T5 (lobula plate - detects moving brightness decreases)
```

- **R1-R6**: photoreceptors. Due to neural superposition, R1-R6 from neighboring facets converge onto the same L1/L2 neurons, so their input is pooled. Brighter light *suppresses* firing (histamine, inhibitory).
- **L1 / L2**: excited by darkness, inhibited by light. They carry the same signal down two separate pathways — L1 preferentially feeds the neurons that track brightness *increases* (ON pathway), L2 feeds the ones that track brightness *decreases* (OFF pathway).
- **Mi1 / Tm3**: continue the ON pathway from L1; the two don't behave identically in time (one carries a more delayed signal), which matters for motion detection.
- **Tm1 / Tm2**: continue the OFF pathway from L2.
- **T4 / T5**: the actual direction-selective output neurons. T4 compares its ON-pathway inputs *across neighboring columns* to detect moving brightness increases; T5 does the same for the OFF pathway and brightness decreases. In the real fly, T4 and T5 each have four subtypes tuned to the four cardinal directions (front-to-back, back-to-front, up, down) — this project currently uses the merged T4/T5 population (see [Neuron types](#neuron-types-and-connectome-data) below), since the stimulus here is 1D (horizontal motion only).

### Sign / neurotransmitter assignments

Each neuron type's excitatory vs. inhibitory sign (used to enforce [Dale's law](https://en.wikipedia.org/wiki/Dale%27s_principle) — all of a neuron's outgoing synapses share one sign) was assigned from known neurotransmitter identity:

| Type | Neurotransmitter | Sign | Basis |
|---|---|---|---|
| R1-R6 | Histamine | inhibitory | Depolarizes in light, releases *more* histamine, which inhibits downstream cells — brighter light suppresses the pathway. |
| L1 | Glutamate | inhibitory | Molina-Obando et al. 2019, *eLife* — L1's glutamate is inhibitory onto Mi1/Tm3, providing the sign inversion that makes the ON pathway end up excited by light increases (double negative: R inhibits L1, L1 inhibits Mi1/Tm3). |
| L2 | Acetylcholine | excitatory | Standard ACh convention. Notably a *different* sign mechanism from L1 despite both being suppressed by R identically — this is exactly what splits one shared photoreceptor input into the ON and OFF pathways. |
| Mi1, Tm3 | Acetylcholine | excitatory | Standard ACh convention; continue the ON pathway. |
| Tm1, Tm2 | Acetylcholine | excitatory | Standard ACh convention; continue the OFF pathway. |
| T4 | Acetylcholine | excitatory | Dominant type by far in the queried data (3438 ACh vs. 1 GABA) — the single GABA neuron was treated as noise, not a real minority population. |
| T5 | Acetylcholine | excitatory | Same reasoning as T4 (3356 ACh vs. 1 "unclear"). |

## Data source

Connectivity data comes from **[neuPrint](https://neuprint.janelia.org)**, Janelia FlyEM's connectome database, using the `male-cns:v1.0` dataset. `query_neuprint.ipynb` queries for the neuron types above within the lamina, medulla, and lobula ROIs, pulls their synaptic adjacency, and pivots it into an average connection-weight matrix per type pair.

Two files are derived from this and checked into the repo:

- **`type_matrix.csv`** — average synapse weight from each neuron type (rows) to each neuron type (columns), pooled across all instances of that type. This is real, queried connectome data.
- **`sign_matrix.csv`** — the excitatory (+1) / inhibitory (-1) sign per presynaptic type, from the table above.

### Neuron types and connectome data

The model supports either configuration, controlled entirely by what's in `type_matrix.csv`/`sign_matrix.csv` — the model code infers the neuron-type count and which columns are T4/T5 automatically:

- **Merged (9 types, current default)**: `R1-R6, L1, L2, Mi1, Tm3, Tm1, Tm2, T4, T5` — all T4 subtypes pooled into one `T4` type, all T5 subtypes into one `T5` type.
- **Split (15 types, explored but reverted)**: `..., T4a, T4b, T4c, T4d, T5a, T5b, T5c, T5d` — the real anatomical subtypes kept separate. This is more biologically faithful, but two of the four subtypes per pathway are tuned to *vertical* motion, which this 1D horizontal stimulus can never provide a training signal for — so it was set aside in favor of the simpler merged version for now.

**Important — no real lateral connectivity data exists.** neuPrint gives synapse counts between neuron *types*, not between specific *spatial columns*. There's no queried data for "does column *i*'s Mi1 project to column *i+1*'s T4" — that spatial, cross-column wiring is inherent to how motion detection works, but it isn't something the current query captures. The model's lateral (left/right neighbor) connections are therefore **free-learned parameters**, not connectome data — see [Model architecture](#model-architecture).

## Repository structure

| File | Purpose |
|---|---|
| `query_neuprint.ipynb` | Queries neuPrint for the circuit's neuron types and their connectivity; builds and saves `type_matrix.csv` / `sign_matrix.csv`. Requires a `NEUPRINT_TOKEN` in `.env`. |
| `type_matrix.csv`, `sign_matrix.csv` | The connectome-derived weight and sign matrices (see above). |
| `stimulus_generator.py` | Procedurally generates synthetic training stimuli (moving bars, drifting gratings, static patterns) and a `SyntheticMotionDataset` for streaming them into training. |
| `generate_training_stimuli.ipynb` | Sanity-check notebook: visualizes example stimuli from the generator before trusting them for training. |
| `motion_detection.ipynb` | Defines the `MotionDetection` model, loads the connectome matrices, trains the model, and plots loss/accuracy. |
| `query_neuprint.ipynb` output cells | Contains the biological reasoning/citations reproduced in this README. |

## Model architecture

`MotionDetection` (in `motion_detection.ipynb`) is a small recurrent network unrolled over time, defined at every one of the 16 spatial positions and `num_types` neuron types (9 or 15, inferred from the loaded connectome matrix):

- **State**: `potential`, shape `(batch, 16, num_types)` — a membrane-potential-like value per position per neuron type, initialized to zero.
- **Per-timestep update** (a discrete leaky integrator):
  1. **Self (same-position) input** — from the real connectome: `softplus(weights) * sign`, where `weights` is `type_matrix.csv` (normalized so no neuron type can receive more than a total input of 1 across all its inputs, which keeps the 24-step recurrence numerically stable) and `sign` is `sign_matrix.csv`.
  2. **Lateral input** — from the left and right neighboring positions, via separately-learned `weights_left` / `weights_right` matrices (no real data for these, as noted above), signed the same way as the self-connections.
  3. **Leak/decay** — a per-type, learned, `sigmoid`-constrained decay rate applied to the current potential.
  4. **Stimulus injection** — the raw pixel value at that timestep is added directly into the first neuron-type "slot" (the photoreceptor input channel) at every position.
  5. `output = tanh(potential)` becomes the next timestep's input.
- **Readout**: after all timesteps, sum the T4 and T5 neuron-type activity across all 16 positions (and across all T4/T5 subtype columns, if using the 15-type version), and pass `T4 + T5` through a sigmoid. The output is interpreted as **P(rightward motion)**.

### Why the lateral weights need a specific initialization (this was the hardest bug to find)

Initializing `weights_left` and `weights_right` to the *same* value creates an exact symmetric fixed point: with identical weights, the network's response to a stimulus and to its mirror image is mathematically identical, so it starts at exactly chance accuracy — and because the (left/right-balanced) training data is itself mirror-symmetric, the *expected* gradient update to both matrices stays equal forever. This isn't a slow-convergence problem, it's a stable saddle point that no amount of training time escapes on its own.

The fix: initialize `weights_left`/`weights_right` from a shared, connectome-shaped structural prior (the same-position connectivity pattern, scaled down), **plus independent random noise per side**. The shared structure gives training a more informed starting point than an arbitrary constant; the independent noise is what actually breaks the symmetry so gradient descent can tell left from right.

A second, related trap: the lateral weights go through `softplus(x)`, whose gradient is `sigmoid(x)`. Initializing `x` too negative (to keep the lateral pathway weak and the recurrent dynamics stable) also drives that gradient toward zero — at `x = -10`, gradient flow to those parameters is roughly 20,000x weaker than near `x = 0`, effectively freezing the one pathway that needs to learn. `x = -3` turned out to be the sweet spot: stable dynamics, but with enough gradient (`sigmoid(-3) ≈ 0.047`) to actually train.

## Training setup

- **Synthetic stimuli** (`stimulus_generator.py`, fully vectorized, no fixed dataset): each training sample is a moving light bar, a drifting sinusoidal grating, or a static (no-motion) pattern, with randomized direction, speed/frequency, bar width, and contrast. Because it's generated procedurally rather than loaded from disk, every batch the model ever sees is effectively fresh, unseen data — there's no train/test split needed and no memorization risk.
- **Labels**: `1.0` = rightward, `0.0` = leftward, `0.5` = static/no motion. The `0.5` choice is deliberate — since the model outputs a sigmoid (a probability), `0.5` is a valid target meaning "no directional preference," letting one binary-cross-entropy loss represent all three cases without a separate multi-class setup.
- **Loss**: `BCELoss` against those labels.
- **Optimizer**: Adam, with gradient-norm clipping (max norm 1.0) to keep the 24-step recurrent backprop stable.
- **Data loading**: `SyntheticMotionDataset` is an `IterableDataset` that yields whole pre-generated batches (not one sample at a time), so both the vectorized generation and `DataLoader(num_workers=...)` parallelism actually do something — generation happens on separate CPU worker processes while the main process trains.
- **Metrics**: per-epoch loss and direction accuracy. Accuracy is computed only over samples with a definite direction label (`label != 0.5`), since static samples have no "correct" class to grade against.

## Results and open questions

With the fixes above (numerical stability, symmetry-breaking, connectome-shaped lateral init), the model reliably moves off chance-level (~50%) direction accuracy over several hundred epochs, with the best observed run reaching roughly 70% — though results are still seed-sensitive, since the amount of symmetry-breaking noise a given random seed happens to produce affects how decisively the model learns. Two natural next steps that were identified but not yet fully explored:

- **Fix a random seed** (`torch.manual_seed(...)`) so a good run can be exactly reproduced and studied, rather than being a one-off.
- **Run a multi-seed sweep** to see whether the variance is mostly noise around a consistent ~55-70% capability, or whether most seeds actually get stuck near chance and only occasionally break out — which would suggest the symmetry-breaking approach needs to be made more deliberate rather than relying on random luck.

## Debugging notes / lessons learned

A number of real bugs were found and fixed along the way, worth recording since they're easy to reintroduce:

- `if not weights:` on a multi-element tensor raises `RuntimeError` (ambiguous truth value) — must be `weights is None`.
- A bare `sigmoid(...)` call with no corresponding import is a `NameError` — needs `torch.sigmoid`.
- Applying `sigmoid` inside the model *and* training with `BCEWithLogitsLoss` (which applies its own internal sigmoid) double-squashes the output — pick one: raw logits + `BCEWithLogitsLoss`, or `sigmoid` output + plain `BCELoss`.
- `pandas.read_csv(...).values` produces `float64` by default, while the rest of the model's tensors are `float32` — needs an explicit `.float()` cast or matmuls fail on dtype mismatch.
- Raw connectome weights (synapse counts) are unnormalized and, compounded over a 24-step recurrent unroll, saturate `tanh`/`sigmoid` almost immediately — fixed by normalizing so the total input any single neuron type can receive (summed across all its inputs) is bounded by 1, and by giving the leak/decay term a strong initial bias instead of starting at zero.

## Sources

- neuPrint, Janelia FlyEM — [https://neuprint.janelia.org](https://neuprint.janelia.org), `male-cns:v1.0` dataset. Primary source for all connectivity (`type_matrix.csv`) and neuron-type data used in this project.
- Molina-Obando, S. et al. (2019). *Julia sets the tone: dynamic gain control in the *Drosophila* visual system* [L1's glutamatergic inhibition onto Mi1/Tm3]. *eLife*. (Cited in `query_neuprint.ipynb` for the L1 sign assignment — see that notebook for the exact reasoning; no DOI/link was recorded alongside the original citation, so verify before citing elsewhere.)

## Running this project

1. **Regenerate the connectome data** (optional — `type_matrix.csv`/`sign_matrix.csv` are already checked in): set `NEUPRINT_TOKEN` in a `.env` file, then run `query_neuprint.ipynb` top to bottom. This makes a live call to the neuPrint server.
2. **Sanity-check the stimulus generator** (optional): run `generate_training_stimuli.ipynb` to visualize example bars/gratings/static patterns.
3. **Train the model**: run `motion_detection.ipynb` top to bottom. It loads `type_matrix.csv`/`sign_matrix.csv`, builds the `MotionDetection` model, and trains it, printing per-epoch loss/accuracy and plotting both at the end.

Requires: `torch`, `pandas`, `matplotlib`, and (only for step 1) `neuprint-python`, `python-dotenv`.
