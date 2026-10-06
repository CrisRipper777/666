# FlowMAG Full-Model NC Stability & Efficiency Study V1

## Study goal

Audit the V0.2.1 implementation fixes and describe full-model NC validation stability, internal routing, gradient health, representation probes, runtime, and GPU memory. This is a descriptive robustness study, not an ablation or SOTA comparison.

## Code fixes and protocol

- Removed the Contrast base bias so a zero residual (`hbar - mu`) maps to zero.
- Conditional routing statistics are computed only over nodes with a valid same-modal or cross-modal source; source availability is reported separately.
- Cross-attention entropy includes real sources and Null, is computed per target and head then averaged across heads, and normalized by `log(D_i + 1)` for targets with `D_i > 0`. Entropy is interpreted as attention diffuseness.
- Added graph-flow versus FFN update magnitudes and pre-clipping component gradient norms. Diagnostic-off forwards avoid summary reductions and scalar CPU copies.
- RMS diagnostics use chunked reductions to cap temporary memory on larger full graphs without changing model outputs or training gradients.
- Five datasets use the existing full-graph NC protocol: 300 epoch cap, AdamW, learning rate 1e-3, weight decay 1e-4, patience 30, validation every epoch, early-stop minimum epoch 30, gradient clip 1.0, no scheduler, default FlowMAG architecture and schedules.
- FlowMAG dimensions are fixed across datasets: modality priors 64 + 64, graph block 128, hidden size 256, two layers, relation/router dimensions 64, adapter rank 16, two cross-attention heads, FFN multiplier 2, dropout 0.2, and attention dropout 0.0.
- Routing defaults are smooth probability 0.75, same/cross admission 0.80/0.20, temperatures 2.0 to 1.0 annealed by 0.60 progress, warmup/ramp 0.05/0.10, update scales 0.1/0.1, and auxiliary weights 0.05/0.10/0.001 (prior/graph-start/router-Z).
- Seeds 42, 43, and 44 are optimization/model seeds on one dataset-provided split loaded once per dataset; MAGB split files use seed 42, while ele-fashion uses its supplied fixed split. They are not independent data splits.
- Best checkpoints are selected by validation accuracy. Representation probes freeze each best checkpoint, fit a fresh linear classifier on train nodes, and select probe stopping by validation accuracy only.
- Probe seed is fixed at 2026; probes run for at most 100 epochs with patience 15 and Adam learning rate 0.01.
- All metric analysis uses train and validation indices only. Test metrics were not evaluated, read for model selection, or reported. No architecture ablation or hyperparameter search was performed.

## Training and validation results

| Dataset | Status | Completed seeds | Validation accuracy (mean ± SD) | Validation macro-F1 (mean ± SD) | Best epochs |
|---|---|---:|---:|---:|---|
| Movies | healthy | 3/3 | 52.69 ± 0.03% | 41.24 ± 1.67% | 45, 48, 53 |
| Toys | healthy | 3/3 | 77.09 ± 0.39% | 74.27 ± 0.41% | 54, 52, 48 |
| Grocery | healthy | 3/3 | 80.10 ± 0.11% | 71.97 ± 0.31% | 47, 45, 48 |
| Reddit-S | healthy | 3/3 | 95.43 ± 0.13% | 90.97 ± 0.55% | 56, 167, 121 |
| ele-fashion | failed / incomplete | 0/3 | NA ± NA% | NA ± NA% | NA |

ele-fashion: Seed 42 reached epoch 2, then the backward pass OOMed requesting 192 MiB with 193.69 MiB free; the initial 96 MiB diagnostic scratch OOM was reduced by chunked RMS, but training still exceeded the GPU limit. Seeds 43 and 44 were not attempted.

## Runtime and memory

FlowMAG medians use checkpoint-eligible epochs after excluding the first three timed eligible epochs. The optional SAGE reference uses ten full-graph train epochs, excludes the first three, and reports medians over the final seven. GPU memory is measured through PyTorch peak allocated/reserved counters. `edges_per_train_second` is descriptive system throughput, not evidence of algorithmic complexity.

| Dataset | Model | Seed | Train median (s) | Validation median (s) | Peak train allocated (MB) | Peak train reserved (MB) | Nodes | Directed edges | Edges / train-second |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | FlowMAG | 42 | 0.192 | 0.105 | 6611.8 | 7158.0 | 16672 | 160802 | 837708.0 |
| Movies | FlowMAG | 43 | 0.228 | 0.108 | 6613.6 | 7150.0 | 16672 | 160802 | 704247.7 |
| Movies | FlowMAG | 44 | 0.189 | 0.107 | 6613.6 | 7150.0 | 16672 | 160802 | 850616.0 |
| Toys | FlowMAG | 42 | 0.176 | 0.119 | 5934.7 | 6488.0 | 20695 | 113402 | 645206.9 |
| Toys | FlowMAG | 43 | 0.177 | 0.122 | 5935.4 | 6498.0 | 20695 | 113402 | 639044.5 |
| Toys | FlowMAG | 44 | 0.176 | 0.109 | 5935.4 | 6498.0 | 20695 | 113402 | 643614.7 |
| Grocery | FlowMAG | 42 | 0.176 | 0.085 | 6165.0 | 6662.0 | 17074 | 142262 | 810143.7 |
| Grocery | FlowMAG | 43 | 0.223 | 0.108 | 6164.0 | 6662.0 | 17074 | 142262 | 638869.0 |
| Grocery | FlowMAG | 44 | 0.192 | 0.136 | 6164.3 | 6662.0 | 17074 | 142262 | 742631.8 |
| Reddit-S | FlowMAG | 42 | 0.240 | 0.122 | 9620.0 | 10402.0 | 15894 | 283080 | 1178946.2 |
| Reddit-S | FlowMAG | 43 | 0.236 | 0.108 | 9623.9 | 10412.0 | 15894 | 283080 | 1197446.2 |
| Reddit-S | FlowMAG | 44 | 0.244 | 0.133 | 9623.9 | 10412.0 | 15894 | 283080 | 1160599.3 |
| ele-fashion | FlowMAG | 42 | NA | NA | NA | NA | 97766 | 399172 | NA |
| ele-fashion | FlowMAG | 43 | NA | NA | NA | NA | 97766 | 399172 | NA |
| ele-fashion | FlowMAG | 44 | NA | NA | NA | NA | 97766 | 399172 | NA |
| Movies | SAGE_reference | reference | 0.019 | NA | 1267.6 | 2210.0 | 16672 | 160802 | 8664904.3 |
| ele-fashion | SAGE_reference | reference | 0.050 | NA | 2738.8 | 2924.0 | 97766 | 399172 | 7963336.1 |

## Internal routing and update behavior

Last-five-eligible-epoch medians are aggregated per seed, then summarized across the three optimization/model seeds with population SD. Availability and conditional routing remain distinct in the CSV.

| Dataset | Metric | Mean across layer × modality summaries | Across-seed SD |
|---|---|---:|---:|
| Movies | same_valid_rate | 1.000 | 0.000 |
| Movies | cross_valid_rate | 1.000 | 0.000 |
| Movies | pi_smooth_given_same_valid | 0.434 | 0.027 |
| Movies | pi_contrast_given_same_valid | 0.566 | 0.027 |
| Movies | same_admission_given_valid | 0.985 | 0.000 |
| Movies | cross_admission_given_valid | 0.493 | 0.095 |
| Movies | null_mass_given_cross_valid | 0.650 | 0.008 |
| Movies | normalized_cross_attention_entropy | 0.214 | 0.019 |
| Movies | graph_update_ratio | 0.097 | 0.004 |
| Movies | ffn_update_ratio | 0.257 | 0.006 |
| Movies | smooth_contrast_cosine | 0.103 | 0.009 |
| Toys | same_valid_rate | 1.000 | 0.000 |
| Toys | cross_valid_rate | 1.000 | 0.000 |
| Toys | pi_smooth_given_same_valid | 0.419 | 0.029 |
| Toys | pi_contrast_given_same_valid | 0.581 | 0.029 |
| Toys | same_admission_given_valid | 0.988 | 0.002 |
| Toys | cross_admission_given_valid | 0.756 | 0.063 |
| Toys | null_mass_given_cross_valid | 0.394 | 0.022 |
| Toys | normalized_cross_attention_entropy | 0.381 | 0.027 |
| Toys | graph_update_ratio | 0.138 | 0.006 |
| Toys | ffn_update_ratio | 0.269 | 0.005 |
| Toys | smooth_contrast_cosine | 0.126 | 0.005 |
| Grocery | same_valid_rate | 1.000 | 0.000 |
| Grocery | cross_valid_rate | 1.000 | 0.000 |
| Grocery | pi_smooth_given_same_valid | 0.413 | 0.026 |
| Grocery | pi_contrast_given_same_valid | 0.587 | 0.026 |
| Grocery | same_admission_given_valid | 0.989 | 0.003 |
| Grocery | cross_admission_given_valid | 0.857 | 0.057 |
| Grocery | null_mass_given_cross_valid | 0.383 | 0.040 |
| Grocery | normalized_cross_attention_entropy | 0.398 | 0.015 |
| Grocery | graph_update_ratio | 0.142 | 0.004 |
| Grocery | ffn_update_ratio | 0.273 | 0.003 |
| Grocery | smooth_contrast_cosine | 0.141 | 0.014 |
| Reddit-S | same_valid_rate | 1.000 | 0.000 |
| Reddit-S | cross_valid_rate | 1.000 | 0.000 |
| Reddit-S | pi_smooth_given_same_valid | 0.689 | 0.014 |
| Reddit-S | pi_contrast_given_same_valid | 0.311 | 0.014 |
| Reddit-S | same_admission_given_valid | 0.999 | 0.002 |
| Reddit-S | cross_admission_given_valid | 0.987 | 0.008 |
| Reddit-S | null_mass_given_cross_valid | 0.381 | 0.038 |
| Reddit-S | normalized_cross_attention_entropy | 0.354 | 0.016 |
| Reddit-S | graph_update_ratio | 0.220 | 0.025 |
| Reddit-S | ffn_update_ratio | 0.297 | 0.003 |
| Reddit-S | smooth_contrast_cosine | 0.117 | 0.002 |

## Gradient health

Gradient norms below are raw pre-clipping L2 norms. The study summarizes Contrast, cross-attention, controller, same-relation, and FFN groups per layer.

| Dataset | Component | Mean layer × seed norm (last-five median) |
|---|---|---:|
| Movies | contrast | 0.002485 |
| Movies | controller | 0.012835 |
| Movies | cross_attention | 0.007918 |
| Movies | ffn | 0.048466 |
| Movies | same_relation | 0.004605 |
| Toys | contrast | 0.001472 |
| Toys | controller | 0.017026 |
| Toys | cross_attention | 0.011204 |
| Toys | ffn | 0.044460 |
| Toys | same_relation | 0.002383 |
| Grocery | contrast | 0.001787 |
| Grocery | controller | 0.003110 |
| Grocery | cross_attention | 0.009610 |
| Grocery | ffn | 0.033631 |
| Grocery | same_relation | 0.002901 |
| Reddit-S | contrast | 0.000044 |
| Reddit-S | controller | 0.000329 |
| Reddit-S | cross_attention | 0.000513 |
| Reddit-S | ffn | 0.001459 |
| Reddit-S | same_relation | 0.000025 |

## Frozen-checkpoint representation probes

The main NC classifier metrics above and linear-probe metrics below come from separate training procedures. Prior-only is `concat(r_text, r_image)`, graph-only is the graph block, and Prior+Graph is their concatenation. Probe rows use train-fit / validation-evaluate only; probe validation scores do not select FlowMAG checkpoints.

| Dataset | Seed | Prior-only val acc / F1 | Graph-only val acc / F1 | Prior+Graph val acc / F1 | Majority val acc / F1 |
|---|---:|---:|---:|---:|---:|
| Movies | 42 | 50.63 / 40.95 | 50.93 / 42.25 | 52.28 / 43.70 | 32.93 / 2.48 |
| Movies | 43 | 50.87 / 41.13 | 50.03 / 40.42 | 52.28 / 45.60 | 32.93 / 2.48 |
| Movies | 44 | 50.45 / 36.57 | 49.73 / 30.96 | 51.71 / 44.17 | 32.93 / 2.48 |
| Toys | 42 | 73.23 / 70.89 | 75.50 / 72.95 | 76.88 / 74.55 | 10.34 / 1.04 |
| Toys | 43 | 73.42 / 70.96 | 75.94 / 73.06 | 77.27 / 74.87 | 10.34 / 1.04 |
| Toys | 44 | 72.99 / 70.85 | 75.65 / 72.56 | 77.34 / 75.10 | 10.34 / 1.04 |
| Grocery | 42 | 75.14 / 66.95 | 79.09 / 71.47 | 79.62 / 66.87 | 14.14 / 1.24 |
| Grocery | 43 | 75.96 / 67.77 | 79.18 / 70.48 | 79.77 / 69.14 | 14.14 / 1.24 |
| Grocery | 44 | 75.43 / 67.77 | 78.13 / 68.22 | 79.18 / 68.80 | 14.14 / 1.24 |
| Reddit-S | 42 | 92.07 / 87.19 | 95.63 / 91.38 | 95.12 / 90.61 | 8.56 / 0.79 |
| Reddit-S | 43 | 92.58 / 87.60 | 95.56 / 91.29 | 95.47 / 91.21 | 8.56 / 0.79 |
| Reddit-S | 44 | 92.61 / 87.39 | 95.25 / 89.90 | 95.19 / 90.14 | 8.56 / 0.79 |
| ele-fashion | 42 | NA | NA | NA | 28.65 / 4.05 |
| ele-fashion | 43 | NA | NA | NA | 28.65 / 4.05 |
| ele-fashion | 44 | NA | NA | NA | 28.65 / 4.05 |

## Cross-dataset consistency

Consistency is described from per-seed values and mean/SD summaries. Three seeds are treated as a robustness sample only; no seed-level p-values or t-tests are used.

| Dataset | Status | Evidence note |
|---|---|---|
| Movies | healthy | three runs and required summaries completed; no joint graph-bypass trigger |
| Toys | healthy | three runs and required summaries completed; no joint graph-bypass trigger |
| Grocery | healthy | three runs and required summaries completed; no joint graph-bypass trigger |
| Reddit-S | healthy | three runs and required summaries completed; no joint graph-bypass trigger |
| ele-fashion | failed / incomplete | Seed 42 reached epoch 2, then the backward pass OOMed requesting 192 MiB with 193.69 MiB free; the initial 96 MiB diagnostic scratch OOM was reduced by chunked RMS, but training still exceeded the GPU limit. Seeds 43 and 44 were not attempted. |

## Failures, threats to validity, and scope

- Formal test metrics were not evaluated, read for checkpoint selection, or reported. Only train and validation labels were copied into the runner's working label tensor.
- No architecture ablation, B0–B4 study, hyperparameter search, dataset-specific tuning, LP campaign, or books-nc run was performed.
- The three optimization/model seeds share a single loaded split, so this does not measure split-to-split uncertainty.
- The validation split is used for both FlowMAG checkpoint selection and probe early stopping; probe outcomes are descriptive and do not feed back into FlowMAG selection.
- Routing and update summaries are layer/modality averages; they do not establish causal component necessity.
- GPU timing depends on the shared device and runtime environment. PyTorch allocator reserved memory may include cached blocks.
- SAGE rows are fixed-budget system references only; no SAGE validation or test accuracy is calculated or compared.
- Low flow usage alone is not interpreted as evidence to remove a component. Health labels are descriptive operational/optimization flags only.

- OOM evidence (ele-fashion): Seed 42 was unavailable after CUDA OOM: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 192.00 MiB. GPU 0 has a total capacity of 23.68 GiB of which 193.69 MiB is free. Including non-PyTorch memory, this process has 23.40 GiB memory in use. Of the allocated memory 22.76 GiB is allocated by PyTorch, and 348.50 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables) An earlier attempt also OOMed in diagnostic RMS; chunked RMS reduced scratch memory, after which epochs 1–2 completed before the backward-pass OOM. Earlier detail: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 96.00 MiB. GPU 0 has a total capacity of 23.68 GiB of which 63.69 MiB is free. Including non-PyTorch memory, this process has 23.53 GiB memory in use. Of the allocated memory 23.13 GiB is allocated by PyTorch, and 91.15 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

## Factual assessment

- **Movies: healthy** — three runs and required summaries completed; no joint graph-bypass trigger.
- **Toys: healthy** — three runs and required summaries completed; no joint graph-bypass trigger.
- **Grocery: healthy** — three runs and required summaries completed; no joint graph-bypass trigger.
- **Reddit-S: healthy** — three runs and required summaries completed; no joint graph-bypass trigger.
- **ele-fashion: failed / incomplete** — Seed 42 reached epoch 2, then the backward pass OOMed requesting 192 MiB with 193.69 MiB free; the initial 96 MiB diagnostic scratch OOM was reduced by chunked RMS, but training still exceeded the GPU limit. Seeds 43 and 44 were not attempted.

No claim is made that FlowMAG is better than state of the art, that the proposed factorization has been causally validated, or that any flow is necessary.

## Artifacts

- `summary.csv` — per-seed and across-seed validation results.
- `routing_summary.csv` — per-seed and mean/SD conditional routing and update summaries at best epoch and over the final five eligible epochs.
- `gradient_summary.csv` — pre-clipping gradient norms at the same windows.
- `efficiency.csv` — runtime, PyTorch memory peaks, parameters, graph size, and descriptive throughput.
- `probe_summary.csv` — frozen checkpoint linear probes and majority baselines.
- `run_manifest.json` — exact protocol, seeds, completion state, and warnings.

Raw logs, epoch diagnostics, and checkpoints are under `/hdd1/DataInHere/YHF/666/outputs/flowmag_nc_stability_v1` and are ignored by Git.
