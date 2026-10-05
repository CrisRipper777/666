# FlowMAG V0.2.1 Implementation Report

## Status and scope

- Branch: `research/flowmag-v0.2.1`
- Base: fetched `origin/main` at `579d8dcde6d9bf6d39efb4f9d88bf70a78acc38e`
- Commit: the implementation commit is recorded in the final completion message and branch history.
- Scope: encoder implementation, optional task hooks, automated tests, documentation, and technical smoke validation.
- No formal experiment, B0–B4, ablation, five-dataset training, hyperparameter search, link-prediction campaign, test-metric analysis, or scientific conclusion was run.

Compact test and smoke results are in [SMOKE_SUMMARY.json](SMOKE_SUMMARY.json). Raw logs and checkpoints remain under ignored `outputs/` and are not committed.

## Files changed

- `src/models/flowmag.py`
- `src/models/flowmag_components/__init__.py`
- `src/models/flowmag_components/adapters.py`
- `src/models/flowmag_components/attention.py`
- `src/models/flowmag_components/controller.py`
- `src/models/flowmag_components/layer.py`
- `src/models/flowmag_components/auxiliary.py`
- `configs/model/flowmag.yaml`
- `src/tasks/common.py`
- `src/tasks/nc.py`
- `src/tasks/lp.py`
- `tests/test_flowmag_components.py`
- `tests/test_flowmag_model.py`
- `tests/test_flowmag_task_hooks.py`
- `docs/flowmag_v0_2_1.md`
- `README.md`
- `CLAUDE.md`
- `reports/flowmag_v0_2_1/IMPLEMENTATION_REPORT.md`
- `reports/flowmag_v0_2_1/SMOKE_SUMMARY.json`

## Architecture

FlowMAG keeps text and image as separate streams. For each modality it checks that all raw dimensions are finite and the cleaned feature has nonzero absolute sum. NaN and infinities are replaced with zero before the independent modality stem; invalid rows select a learned missing embedding. Valid source masks are applied independently to same-modal and cross-modal source edges. A target with missing modality can still receive messages from valid neighbors.

Each layer applies a shared PreNorm to both streams. The same-modal relation kernel uses shared query/key bases with separate text/image low-rank adapters, normalizes valid incoming edges, and computes the weighted aggregation once. Smooth and contrast transforms share their respective bases across modalities and use modality adapters. They produce aggregation/smoothing-like and diversification/contrast-like flows. Empty valid neighborhoods force both same-modal flows and effective same admission to zero.

Cross attention is directional (`t2i` and `i2t`) with shared query, key, value, target-reference, and output bases plus direction-specific low-rank adapters. Each direction has an independent learned Null key and bias. For targets with valid source edges, the Null logit receives `log(D)`; targets without a valid real source attend only to Null. Cross corrections aggregate weighted source-value minus target-reference differences. The output projection and its adapters have no bias, and the correction is explicitly zero for an empty source set.

A mask-aware shared scalar scorer pools the two modality tokens into a controller context. Both-missing nodes use a learned all-missing context. The context only affects the controller and is excluded from final `z`. Smooth/contrast candidate descriptors concatenate affine-free layer-normalized direction and log RMS magnitude, computed entirely from detached candidates. The same and cross admissions are separate from frequency selection; cross admission also takes detached Null mass. Actual graph-update flows remain differentiable.

Each layer then applies the configured fixed same/cross update scales and residual dropout, followed by a modality-shared FFN and a second residual dropout. The final output concatenates independent text prior, image prior, and graph-enriched blocks. There is no normalization or projection over the concatenated output.

## Parameter sharing

| Components | Shared parameters | Separate parameters |
|---|---|---|
| Modality stems | None | Text and image stem weights and missing embeddings |
| PreNorm and FFN, within each layer | One PreNorm and one FFN are reused for text/image | Different layers have independent parameters |
| Same-modal relation Q/K | Base maps shared by text/image | Low-rank adapters keyed by `text` and `image` |
| Smooth and contrast transforms | One shared base per transform | Modality-specific low-rank adapters; smooth and contrast are independent |
| Cross Q/K/V/reference/output | Base maps shared across directions | `t2i` and `i2t` low-rank adapters |
| Cross Null source | None | Null key and bias per direction |
| Controller per layer | Context scorer, candidate scorer, and admission MLPs shared between target modalities | Modality and flow-type embeddings; smooth/contrast flow biases |
| Final projections | Graph fuse shared across nodes | Text/image prior projections and norms; graph output norm |
| Supervised heads | N/A | Task-specific NC or LP heads; these are not used for inference |

Adapter computation is `base(x) + B_key(A_key(x))`. `A` uses Kaiming initialization and `B` is zero-initialized, so the initial adapter delta is zero while its zero factor can receive gradient.

## Shapes and parameter counts

For Movies, raw input is `[16672, 1536]`, with text and image widths 768 each. Default FlowMAG maps each stream to `[N, 256]`, runs two graph layers at width 256, and emits `[N, 64]` text prior, `[N, 64]` image prior, and `[N, 128]` graph block. Thus `z` is `[N, 256]`. All dimensions are configurable; `out_dim = 2 * prior_dim_per_modality + graph_block_dim`.

| Task | Total encoder parameters | Train-only auxiliary heads | Inference parameters |
|---|---:|---:|---:|
| NC, 20 classes | 2,530,252 | 5,180 | 2,525,072 |
| LP | 2,525,331 | 259 | 2,525,072 |

The NC task classifier and LP projection/LinkPredictor are runner-side baseline modules and are not counted in the encoder rows.

## Training schedules and checkpoint eligibility

`set_epoch(epoch)` computes normalized progress as `epoch / total_epochs`. Phase A fixes frequency to `[0.75, 0.25]`, same admission to `0.8`, cross admission to `0.2`, and effective Router-Z weight to zero. Phase B linearly interpolates from those priors to learned values over `controller_ramp_ratio`; invalid neighborhoods still set their effective admission to zero. Phase C uses learned soft values. The smooth/contrast prior bias is `router_temp_start * log(3)`, giving the requested 0.75 initial probability at temperature 2. Temperature anneals linearly to its configured endpoint, without hard routing.

The NC and LP runners call the optional checkpoint eligibility helper after validation. A model without the hook remains eligible every epoch. FlowMAG is ineligible until warmup plus controller ramp is complete; such validations neither replace best state nor consume patience. After best weights are restored, FlowMAG resets its non-parameter schedule state to the selected epoch before final inference. The graph auxiliary coefficient decreases linearly to zero at `graph_aux_decay_end_ratio`; prior auxiliary weight is fixed.

`forward()` returns only label-free Router-Z in `aux_loss`, computed from frequency logits on nodes with a valid same-modal neighborhood and multiplied by the phase-effective coefficient. `compute_supervised_aux_loss()` is separate. NC uses only labels passed for training indices. LP uses only the current sampled training batch's `edge_label_index` and `edge_label`. The runners multiply the sum of encoder and supervised auxiliary losses by `task.loss.aux_weight`. Models without the optional hook receive scalar zero. Existing models also remain checkpoint-eligible by default.

For LP, FlowMAG declares `requires_full_lp_sampler_depth=True`. The existing resolver clips or extends configured fanouts to `model.num_layers`; other models keep their existing fixed three-fanout path. The baseline `LinkPredictor` and positive message-edge removal remain unchanged.

## Inference and diagnostics

Inference computes modality masks/stems in batches and stores the initial modality states on CPU. It then creates a one-hop full-neighbor `NeighborLoader` per layer with all nodes as input seeds and `shuffle=False`. For every sampled batch, it gathers the current global states via `n_id`, runs the layer on the sampled edges, and stores only the first `batch_size` seed outputs. Final prior and graph projections also run in batches; the method returns CPU embeddings. The two-layer synthetic equivalence check had maximum absolute difference `4.77e-7` in eval mode.

`aux_info` contains detached scalar diagnostics by layer and modality: mean frequency weights, same/cross admission, smooth/contrast/cross RMS, update/state RMS, and smooth/contrast cosine. Each cross direction reports mean `null_source_mass` and attention entropy. Top-level diagnostics report phase, `rho`, temperature, current graph and Router-Z weights, and prior/graph block norms. The training-only `_supervised` entry carries per-node tensors for the auxiliary hook and is omitted in eval mode.

## Tests and technical smoke

`python -m pytest -q` passed all 21 tests. Coverage includes adapter initialization and gradient, empty same neighborhoods, zero cross correction without real sources, Null cardinality correction, detached descriptors, exact warmup and linear ramp, checkpoint timing, modality source masks and missing targets, Router-Z phase behavior, NC/LP auxiliary label boundaries, finite end-to-end gradients, config validation, output contract and prior-block preservation, sampler depth, and full/layerwise inference equivalence.

`python -m compileall src tests` and `git diff --check` passed. Real Movies runs all set `task.evaluate_test=false`:

- FlowMAG NC: 2 epochs, full graph, CUDA; validation ran each epoch. Best epoch 1. The run completed with validation accuracy 0.3293 and macro-F1 0.0248.
- FlowMAG LP: 1 epoch, two training batches, zero workers, CUDA; resolved fanouts `[5, 5]`. Validation MRR was 0.1621.
- MLP NC regression: 1 epoch, two training batches, CUDA. Completed with validation accuracy 0.3686.
- SAGE LP regression: 1 epoch, one training batch, zero workers, CUDA; preserved fanouts `[5, 5, 5]`. Validation MRR was 0.0728.

These short validation values only show that training, checkpoint selection, and validation execute. They are not model comparisons or research evidence. No test metric was computed.

## Warnings and limitations

- PyTorch Geometric emitted a dependency deprecation warning for `torch_geometric.distributed`; no FlowMAG warning appeared in the smoke runs.
- PyTorch CUDA was available and the four smoke runs completed on `cuda:0`. The container's `nvidia-smi` command could not query the driver, so peak process memory was not captured. Between the FlowMAG NC and LP smokes, PyTorch reported 23.35 GiB free on `cuda:0`.
- Layerwise inference uses full one-hop neighborhoods and therefore has work proportional to the supplied graph's edge count across all node batches. It reduces activation residency, not total edge work.
- No formal experiment, B0–B4, full benchmark, hyperparameter search, LP campaign, test-set performance analysis, or final scientific conclusion was performed.
- No architecture, runner-compatibility, or smoke-command requirement was intentionally deviated from.
