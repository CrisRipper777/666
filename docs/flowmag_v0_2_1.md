# FlowMAG V0.2.1

FlowMAG is a MAG-specific encoder for graphs whose nodes have text and image features. Its controller separates three choices: aggregation-like versus diversification/contrast-like processing, same-modal admission, and directional cross-modal admission. The design adds a MAG-specific inductive bias; it does not make a claim that text and image must induce different effective edge weights.

## 1. Problem abstraction

The baseline input is `[text, visual]`. FlowMAG treats the two feature streams separately, builds same-modal and directional cross-modal messages on the graph, and returns three explicit blocks. The controller uses node-local multimodal context and candidate descriptors. It does not use global graph pooling.

## 2. Prior-retaining modality streams

Text and image have independent `LayerNorm → Linear → GELU → LayerNorm` stems. An all-finite, nonzero modality is marked valid. NaN and infinite values are replaced before projection; invalid modalities use a learnable missing embedding. The initial graph states are the stem outputs. Independent prior projections are computed from these initial states and remain explicit in the final representation.

## 3. Same relation kernel

Each layer shares its query and key base maps between modalities and has separate low-rank text/image adapters. Incoming attention normalizes over valid source edges. The relation is feature-conditioned; consequently, it permits modality-conditioned weights without requiring that the resulting text and image adjacency weights differ.

## 4. Aggregation and diversification flows

The valid incoming same-modal sources are aggregated once. An aggregation/smoothing-like transform maps the weighted mean, while an independent diversification/contrast-like transform maps `pre_norm_target - weighted_mean`. Both flows use shared base transforms with modality adapters. If no valid source exists, both flows and the effective same-modal admission are zero. These terms are not described as strict spectral low/high-pass filters because the attention kernel is feature-conditioned and is not a fixed symmetric spectral operator.

## 5. Directional cross flow and Null

Text-to-image and image-to-text use separate low-rank adapters around shared query, key, value, reference, and output bases. Each direction has a learned Null key and bias. When a target has valid source neighbors, `log(source_count)` is added to the Null score; when it has none, Null is the only source. Attention dropout is fixed to zero in V0.2.1.

Cross messages are weighted differences between each source value and the target reference value. The output projection has no bias, and the cross correction is explicitly zeroed when no valid source exists. The reported Null quantity is `null_source_mass` (source-level rejection mass), not calibrated confidence. Target modality validity does not mask graph updates, so valid neighbors can provide a graph update to a target with a missing modality.

## 6. Factorized controller

A shared scalar scorer attends over two mask-aware modality tokens. If both are missing, a learned all-missing context is used. This context only enters the controller. Candidate descriptors concatenate an affine-free normalized direction and log RMS magnitude from detached flow tensors. Detached Null mass is included in cross admission; the actual flow tensors used in graph updates stay attached.

One candidate scorer handles smooth and contrast candidates with flow-type embeddings and flow biases. Separate MLPs estimate same-modal and cross-modal admissions. V0.2.1 does not use load balancing, orthogonality, contrastive alignment, or oracle routing losses.

## 7. Training schedule

`set_epoch(epoch)` updates a normalized training progress. Phase A fixes frequency to `[0.75, 0.25]`, same admission to `0.8`, cross admission to `0.2`, and the effective Router-Z coefficient to zero. Phase B linearly blends those priors with learned values over `controller_ramp_ratio`. Phase C uses the learned values. Invalid same/cross neighborhoods always set their corresponding effective admission to zero.

The learned smooth prior is initialized by a frequency-logit difference of `router_temp_start * log(3)`, so at temperature 2 it begins at 0.75. Temperature anneals from the configured start to end by `router_temp_anneal_end_ratio`; it remains soft. The graph auxiliary coefficient decreases linearly to zero by `graph_aux_decay_end_ratio`.

## 8. Auxiliary supervision

The encoder's `aux_loss` contains only label-free Router-Z. Prior and graph deep supervision is computed by `compute_supervised_aux_loss()` outside `forward()` and multiplied by `task.loss.aux_weight` in the runners. NC heads see only the passed training-node indices and labels. LP heads see only the current training batch's edge labels and pairs. Prior modality losses are averaged over modalities that have valid supervised examples; graph loss uses all passed training examples.

The NC runner and LP runner ask optional model hooks whether a validation checkpoint is eligible. Models without a hook remain eligible at every validation epoch. For FlowMAG, best-state updates and patience changes begin only after the controller ramp ends. LP retains the baseline `LinkPredictor`; FlowMAG's `requires_full_lp_sampler_depth` flag resolves sampler fanouts to its encoder depth. Positive LP label edges continue to be removed from the sampled message graph by the existing runner.

After loading best weights, the runners call a separate optional schedule-restore hook. FlowMAG restores `rho`, temperature, and auxiliary coefficients to the selected epoch before final inference; older models have no hook and retain their previous behavior.

## 9. NC/LP compatibility and inference

The public model contract is `Model(cfg, data_info)`, `forward(x, edge_index) → (z, None, None, aux_loss, aux_info)`, `out_dim`, and `inference(x, edge_index, device, batch_size)`. `out_dim` is computed from configured prior and graph block dimensions. No joint normalization or final fusion projection is applied after concatenating the blocks.

Layerwise inference batches the stem computation, then runs each FlowMAG layer with `NeighborLoader(num_neighbors=[-1], shuffle=False)` and saves only each batch's seed-node states. Final prior and graph projections run in batches. This retains the full incoming neighborhood for each target at each layer. On an evaluation-mode graph, this path should match full forward up to floating-point aggregation order.

## Evidence boundaries

Technical smoke runs check execution and validation wiring only. They are not formal experiments or scientific conclusions. This document does not claim downstream tasks will use every output block. No test metrics are used to judge or tune the model.
