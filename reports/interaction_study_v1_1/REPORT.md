# MAG Interaction Study V1.1 — Corrective and Robustness Re-analysis

## 1. Positioning

This is a **post-hoc corrective and robustness audit of V1**. It reuses V1's frozen validation split, hidden states, alignments, host state dictionaries, and response tensors. It does not retrain diagnostic hosts. No test labels or test metrics were accessed. The findings remain specific to the graph-free frozen diagnostic host and the stated action banks.

## 2. Reused V1 artifacts

All 15 formal dataset × host-seed raw artifacts passed existence, required-key, shape-schema, and SHA256 checks before analysis. Their shape schemas are in `protocol.json`; the complete hashes are listed here.

| Dataset | Seed | Artifact | SHA256 |
|---|---:|---|---|
| Movies | 0 | `outputs/interaction_study_v1/Movies/Movies_seed_0_raw.pt` | `efffe97d1234d937ac13be1609c18e7898f212b1af3dcf886d57731d141b1bf0` |
| Movies | 1 | `outputs/interaction_study_v1/Movies/Movies_seed_1_raw.pt` | `4c6b844d82f3346ae4923cbf7d788204abc790a6599cb7ae4d1683cc0ac5bf43` |
| Movies | 2 | `outputs/interaction_study_v1/Movies/Movies_seed_2_raw.pt` | `998660704650b9589c537c10b8fb05d2b1cb9ac32d2f00b9b59a6bf9c53c58c2` |
| Toys | 0 | `outputs/interaction_study_v1/Toys/Toys_seed_0_raw.pt` | `e7c3a46744818d271939193b3c187dc90166aa56bd9e524c7900c1f7c7e02cf0` |
| Toys | 1 | `outputs/interaction_study_v1/Toys/Toys_seed_1_raw.pt` | `1b66b60afd8ef04a41d361c512686d033e03a7f59f4939d3c2bb262387f364ed` |
| Toys | 2 | `outputs/interaction_study_v1/Toys/Toys_seed_2_raw.pt` | `e04635b44f276aaf4567a287cea6ba13067b0739986ea85551510040ebabeca6` |
| Grocery | 0 | `outputs/interaction_study_v1/Grocery/Grocery_seed_0_raw.pt` | `36b27a7b86a8eddb9dd3919c2da7f709958f23372119a524e5011c175f51f3f4` |
| Grocery | 1 | `outputs/interaction_study_v1/Grocery/Grocery_seed_1_raw.pt` | `010a7cbf15273fb23399c0593819b62fa7248291730708ab73afba1c4655e3e8` |
| Grocery | 2 | `outputs/interaction_study_v1/Grocery/Grocery_seed_2_raw.pt` | `d8dbc8a7643d9f999306e4941b12e51311cca39539c1081a213056a40a71b3c2` |
| Reddit-S | 0 | `outputs/interaction_study_v1/Reddit-S/Reddit-S_seed_0_raw.pt` | `729e19c382ce011ab3c3f8b9aa097505f1914c159f67cc5754f97dfd00870d52` |
| Reddit-S | 1 | `outputs/interaction_study_v1/Reddit-S/Reddit-S_seed_1_raw.pt` | `36ef9254f6c3d754d366a2fa4eea5e2e402fe02e843cf7402089404ad1e50d05` |
| Reddit-S | 2 | `outputs/interaction_study_v1/Reddit-S/Reddit-S_seed_2_raw.pt` | `ddda9d447dd83d1adbfc58136292e3bb13ad139c2fa8f76be180a0201a7d1577` |
| ele-fashion | 0 | `outputs/interaction_study_v1/ele-fashion/ele-fashion_seed_0_raw.pt` | `9e2812cfe29a4b28ce0bf984386d628135da6a932def110d022b48c1de776742` |
| ele-fashion | 1 | `outputs/interaction_study_v1/ele-fashion/ele-fashion_seed_1_raw.pt` | `b42a6a484125ada36065154d84ac4d4d900cec74fe19ab1eed00971c4d443dd6` |
| ele-fashion | 2 | `outputs/interaction_study_v1/ele-fashion/ele-fashion_seed_2_raw.pt` | `6d2e1a9c22387f6f918a1df16ada0a297d996345c266a9752aa1a6573b62b83b` |

## 3. P1 — Correct receiver orientation and reference noise

For each canonical edge, a validation endpoint is now the receiver; an edge with two validation endpoints yields both directed interactions. V1 stored one source-averaged percentile per physical edge, so this orientation correction preserves the edge-weighted point estimate while changing receiver-block assignment and its uncertainty.

The reference audit uses five deterministic draws `[4103, 5103, 6103, 7103, 8103]`, 32 nonedges per source, and the same graph/physical edge set. Within each draw, text and image use the same sampled source/nonedge pattern. The descriptive ratio is not a formal signal-to-noise statistic.

| Dataset | V1 mean gap | Corrected mean gap (95% receiver CI) | Median | Spearman | Cross signal mean (CI) | Text instability mean (CI) | Image instability mean (CI) | Adjusted gap mean (CI) | Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | 0.2899 | 0.2899 [0.2818, 0.2984] | 0.2031 | 0.2253 | 0.2963 [0.2859, 0.3068] | 0.0371 [0.0365, 0.0379] | 0.0198 [0.0190, 0.0205] | 0.2394 [0.2295, 0.2489] | 5.2089 |
| Toys | 0.2832 | 0.2832 [0.2760, 0.2900] | 0.1875 | 0.1637 | 0.2904 [0.2820, 0.2980] | 0.0343 [0.0337, 0.0349] | 0.0200 [0.0193, 0.0208] | 0.2361 [0.2278, 0.2445] | 5.3486 |
| Grocery | 0.3035 | 0.3035 [0.2978, 0.3091] | 0.2500 | 0.1740 | 0.3219 [0.3134, 0.3301] | 0.0398 [0.0391, 0.0404] | 0.0331 [0.0322, 0.0342] | 0.2490 [0.2409, 0.2575] | 4.4151 |
| Reddit-S | 0.2231 | 0.2231 [0.2121, 0.2340] | 0.1250 | 0.1014 | 0.2266 [0.2133, 0.2403] | 0.0335 [0.0324, 0.0347] | 0.0157 [0.0147, 0.0167] | 0.1774 [0.1637, 0.1912] | 4.6024 |
| ele-fashion | 0.1329 | 0.1329 [0.1264, 0.1399] | 0.0312 | 0.3578 | 0.1367 [0.1267, 0.1472] | 0.0117 [0.0112, 0.0123] | 0.0241 [0.0233, 0.0250] | 0.1009 [0.0922, 0.1104] | 3.8159 |

A positive adjusted-gap receiver-bootstrap lower bound indicates that the mean cross-modal percentile gap exceeds the summed within-modality reference instability under this conservative definition. Raw gap magnitude is retained regardless of that interval.

## 4. P2 — Directional Cross-modal Neighborhood Utility

The formal name is **Directional Cross-modal Neighborhood Utility**. For each direction and alpha, same-modal, true cross-modal, and matched random cross-modal perturbations are separately scaled to equal relative norm. The cross-only baseline is `LR − LC`; conditional graph gain is `LSR − LSC`; conditional graph cross delta is `LSC − LS`; conditional random cross delta is `LSR − LS`. The interaction quantity is reported only as a same/cross perturbation interaction term.

Each seed row reports its own receiver-bootstrap interval and sign fractions. The seed-averaged row first averages each receiver's metric over the three host seeds, then bootstraps those receiver values. No seed-interval endpoints are averaged. The evidence grade follows the V1.1 Strong / Consistent-Weak / Mixed / Unavailable rubric; no minimum effect cutoff is added.

| Dataset | Direction | Metric | Seed-averaged mean (receiver CI) | Seed mean signs (0/1/2) | Grade |
|---|---|---|---:|---|---|
| Movies | T2I | `cross_only_graph_gain` | 0.0131 [0.0109, 0.0153] | 1, 1, 1 | Strong |
| Movies | T2I | `conditional_graph_gain` | 0.0115 [0.0094, 0.0138] | 1, 1, 1 | Strong |
| Movies | T2I | `conditional_cross_delta_graph` | -0.0103 [-0.0139, -0.0067] | -1, -1, -1 | Strong |
| Movies | T2I | `conditional_cross_delta_random` | 0.0012 [-0.0025, 0.0048] | 1, 1, 1 | Mixed |
| Movies | I2T | `cross_only_graph_gain` | 0.0069 [0.0046, 0.0092] | 1, 1, 1 | Strong |
| Movies | I2T | `conditional_graph_gain` | 0.0054 [0.0032, 0.0074] | 1, 1, 1 | Strong |
| Movies | I2T | `conditional_cross_delta_graph` | 0.0153 [0.0115, 0.0195] | 1, 1, 1 | Strong |
| Movies | I2T | `conditional_cross_delta_random` | 0.0207 [0.0165, 0.0251] | 1, 1, 1 | Strong |
| Movies | T2I_minus_I2T | `cross_only_graph_gain` | 0.0062 [0.0041, 0.0082] | 1, 1, 1 | Strong |
| Movies | T2I_minus_I2T | `conditional_graph_gain` | 0.0061 [0.0040, 0.0082] | 1, 1, 1 | Strong |
| Movies | T2I_minus_I2T | `conditional_cross_delta_graph` | -0.0256 [-0.0312, -0.0196] | -1, -1, -1 | Strong |
| Movies | T2I_minus_I2T | `conditional_cross_delta_random` | -0.0195 [-0.0247, -0.0146] | -1, -1, -1 | Strong |
| Toys | T2I | `cross_only_graph_gain` | 0.0228 [0.0204, 0.0251] | 1, 1, 1 | Strong |
| Toys | T2I | `conditional_graph_gain` | 0.0184 [0.0161, 0.0205] | 1, 1, 1 | Strong |
| Toys | T2I | `conditional_cross_delta_graph` | -0.0222 [-0.0255, -0.0191] | -1, -1, -1 | Strong |
| Toys | T2I | `conditional_cross_delta_random` | -0.0039 [-0.0062, -0.0014] | -1, -1, -1 | Strong |
| Toys | I2T | `cross_only_graph_gain` | 0.0212 [0.0189, 0.0234] | 1, 1, 1 | Strong |
| Toys | I2T | `conditional_graph_gain` | 0.0179 [0.0156, 0.0203] | 1, 1, 1 | Strong |
| Toys | I2T | `conditional_cross_delta_graph` | -0.0035 [-0.0064, -0.0007] | -1, -1, -1 | Strong |
| Toys | I2T | `conditional_cross_delta_random` | 0.0144 [0.0119, 0.0171] | 1, 1, 1 | Strong |
| Toys | T2I_minus_I2T | `cross_only_graph_gain` | 0.0016 [-0.0006, 0.0037] | 1, 1, 1 | Mixed |
| Toys | T2I_minus_I2T | `conditional_graph_gain` | 0.0005 [-0.0015, 0.0026] | -1, 1, 1 | Mixed |
| Toys | T2I_minus_I2T | `conditional_cross_delta_graph` | -0.0188 [-0.0231, -0.0137] | -1, -1, -1 | Strong |
| Toys | T2I_minus_I2T | `conditional_cross_delta_random` | -0.0183 [-0.0225, -0.0142] | -1, -1, -1 | Strong |
| Grocery | T2I | `cross_only_graph_gain` | 0.0259 [0.0230, 0.0288] | 1, 1, 1 | Strong |
| Grocery | T2I | `conditional_graph_gain` | 0.0195 [0.0165, 0.0223] | 1, 1, 1 | Strong |
| Grocery | T2I | `conditional_cross_delta_graph` | -0.0247 [-0.0287, -0.0207] | -1, -1, -1 | Strong |
| Grocery | T2I | `conditional_cross_delta_random` | -0.0052 [-0.0089, -0.0020] | -1, -1, -1 | Strong |
| Grocery | I2T | `cross_only_graph_gain` | 0.0293 [0.0257, 0.0329] | 1, 1, 1 | Strong |
| Grocery | I2T | `conditional_graph_gain` | 0.0233 [0.0196, 0.0271] | 1, 1, 1 | Strong |
| Grocery | I2T | `conditional_cross_delta_graph` | 0.0007 [-0.0035, 0.0052] | -1, -1, 1 | Mixed |
| Grocery | I2T | `conditional_cross_delta_random` | 0.0240 [0.0204, 0.0281] | 1, 1, 1 | Strong |
| Grocery | T2I_minus_I2T | `cross_only_graph_gain` | -0.0034 [-0.0063, -0.0007] | -1, -1, -1 | Strong |
| Grocery | T2I_minus_I2T | `conditional_graph_gain` | -0.0038 [-0.0066, -0.0010] | -1, -1, -1 | Strong |
| Grocery | T2I_minus_I2T | `conditional_cross_delta_graph` | -0.0254 [-0.0323, -0.0187] | -1, -1, -1 | Strong |
| Grocery | T2I_minus_I2T | `conditional_cross_delta_random` | -0.0292 [-0.0354, -0.0228] | -1, -1, -1 | Strong |
| Reddit-S | T2I | `cross_only_graph_gain` | 0.0197 [0.0168, 0.0229] | 1, 1, 1 | Strong |
| Reddit-S | T2I | `conditional_graph_gain` | 0.0160 [0.0134, 0.0187] | 1, 1, 1 | Strong |
| Reddit-S | T2I | `conditional_cross_delta_graph` | -0.0180 [-0.0212, -0.0152] | -1, -1, -1 | Strong |
| Reddit-S | T2I | `conditional_cross_delta_random` | -0.0020 [-0.0036, -0.0005] | -1, -1, -1 | Strong |
| Reddit-S | I2T | `cross_only_graph_gain` | 0.0143 [0.0119, 0.0168] | 1, 1, 1 | Strong |
| Reddit-S | I2T | `conditional_graph_gain` | 0.0125 [0.0103, 0.0149] | 1, 1, 1 | Strong |
| Reddit-S | I2T | `conditional_cross_delta_graph` | -0.0101 [-0.0130, -0.0075] | -1, -1, -1 | Strong |
| Reddit-S | I2T | `conditional_cross_delta_random` | 0.0024 [-0.0001, 0.0050] | 1, 1, 1 | Mixed |
| Reddit-S | T2I_minus_I2T | `cross_only_graph_gain` | 0.0053 [0.0028, 0.0080] | 1, 1, 1 | Strong |
| Reddit-S | T2I_minus_I2T | `conditional_graph_gain` | 0.0035 [0.0011, 0.0059] | 1, 1, 1 | Strong |
| Reddit-S | T2I_minus_I2T | `conditional_cross_delta_graph` | -0.0079 [-0.0116, -0.0045] | -1, -1, -1 | Strong |
| Reddit-S | T2I_minus_I2T | `conditional_cross_delta_random` | -0.0044 [-0.0078, -0.0012] | -1, -1, -1 | Strong |
| ele-fashion | T2I | `cross_only_graph_gain` | 0.0024 [0.0015, 0.0033] | 1, 1, 1 | Strong |
| ele-fashion | T2I | `conditional_graph_gain` | 0.0012 [0.0003, 0.0020] | 1, 1, 1 | Strong |
| ele-fashion | T2I | `conditional_cross_delta_graph` | -0.0005 [-0.0018, 0.0006] | -1, -1, -1 | Mixed |
| ele-fashion | T2I | `conditional_cross_delta_random` | 0.0007 [-0.0003, 0.0017] | 1, 1, 1 | Mixed |
| ele-fashion | I2T | `cross_only_graph_gain` | 0.0036 [0.0021, 0.0052] | 1, 1, 1 | Strong |
| ele-fashion | I2T | `conditional_graph_gain` | 0.0010 [-0.0004, 0.0025] | 1, 1, 1 | Mixed |
| ele-fashion | I2T | `conditional_cross_delta_graph` | 0.0007 [-0.0007, 0.0020] | 1, 1, 1 | Mixed |
| ele-fashion | I2T | `conditional_cross_delta_random` | 0.0018 [0.0006, 0.0030] | 1, 1, 1 | Strong |
| ele-fashion | T2I_minus_I2T | `cross_only_graph_gain` | -0.0012 [-0.0024, -0.0001] | -1, -1, -1 | Consistent-Weak |
| ele-fashion | T2I_minus_I2T | `conditional_graph_gain` | 0.0002 [-0.0011, 0.0014] | 1, -1, 1 | Mixed |
| ele-fashion | T2I_minus_I2T | `conditional_cross_delta_graph` | -0.0012 [-0.0030, 0.0005] | -1, -1, -1 | Mixed |
| ele-fashion | T2I_minus_I2T | `conditional_cross_delta_random` | -0.0011 [-0.0025, 0.0005] | -1, -1, -1 | Mixed |

Directional asymmetry rows use T2I minus I2T receiver-level contrasts. Alpha sensitivity at 0.05 and 0.2 is included in `p2_conditional_utility.csv` alongside the primary 0.1 rows.

## 5. P3 — Receiver-block matched structural response

The primary deterministic estimate matches `semantic_bin × source_degree_bin × receiver_degree_bin` strata with both `structure_bin=0` and `structure_bin=3`, then weights each stratum by `min(n_high,n_low)`. The 95% percentile interval resamples validation receiver blocks, retains every sampled receiver's rows with multiplicity, and re-forms the strata on each draw. Draws without any valid matched stratum are omitted and counted. Standardization divides the matched estimate by the population standard deviation of all report-interaction `D_same` values for that seed and modality.

| Dataset | Modality | Seed | Matched effect at α=.1 (95% receiver CI) | Standardized | Matched strata | Interactions/arm | Valid bootstrap draws | α sign stable | Evidence grade |
|---|---|---:|---:|---:|---:|---:|---:|:---:|---|
| Movies | text | 0 | -0.0281 [-0.1039, 0.0495] | -0.0252 | 32 | 6790 | 1000 | yes | Weak |
| Movies | image | 0 | -0.0389 [-0.1103, 0.0309] | -0.0365 | 32 | 6790 | 1000 | yes | Weak |
| Movies | text | 1 | -0.0213 [-0.0937, 0.0526] | -0.0197 | 32 | 6790 | 1000 | yes | Weak |
| Movies | image | 1 | -0.0548 [-0.1299, 0.0199] | -0.0504 | 32 | 6790 | 1000 | yes | Weak |
| Movies | text | 2 | -0.0603 [-0.1414, 0.0133] | -0.0514 | 32 | 6790 | 1000 | yes | Weak |
| Movies | image | 2 | -0.0351 [-0.1080, 0.0379] | -0.0314 | 32 | 6790 | 1000 | yes | Weak |
| Toys | text | 0 | -0.0990 [-0.1714, -0.0352] | -0.0953 | 18 | 4187 | 1000 | yes | Strong |
| Toys | image | 0 | -0.0871 [-0.1701, -0.0099] | -0.0745 | 9 | 4220 | 1000 | yes | Strong |
| Toys | text | 1 | -0.0886 [-0.1555, -0.0234] | -0.0861 | 18 | 4187 | 1000 | yes | Strong |
| Toys | image | 1 | -0.1074 [-0.1807, -0.0359] | -0.1004 | 9 | 4220 | 1000 | yes | Strong |
| Toys | text | 2 | -0.1015 [-0.1684, -0.0314] | -0.0977 | 18 | 4187 | 1000 | yes | Strong |
| Toys | image | 2 | -0.1070 [-0.1774, -0.0322] | -0.0953 | 9 | 4220 | 1000 | yes | Strong |
| Grocery | text | 0 | -0.0693 [-0.1652, 0.0318] | -0.0482 | 32 | 5114 | 1000 | yes | Weak |
| Grocery | image | 0 | -0.0190 [-0.1506, 0.1401] | -0.0124 | 32 | 5114 | 1000 | yes | Mixed |
| Grocery | text | 1 | -0.0336 [-0.1253, 0.0551] | -0.0250 | 32 | 5114 | 1000 | yes | Weak |
| Grocery | image | 1 | 0.0409 [-0.0702, 0.1761] | 0.0287 | 32 | 5114 | 1000 | yes | Mixed |
| Grocery | text | 2 | -0.0550 [-0.1530, 0.0408] | -0.0390 | 32 | 5114 | 1000 | yes | Weak |
| Grocery | image | 2 | -0.0013 [-0.1300, 0.1271] | -0.0009 | 32 | 5114 | 1000 | no | Mixed |
| Reddit-S | text | 0 | NA [NA, NA] | NA | 0 | 0 | 0 | NA | Unavailable |
| Reddit-S | image | 0 | NA [NA, NA] | NA | 0 | 0 | 0 | NA | Unavailable |
| Reddit-S | text | 1 | NA [NA, NA] | NA | 0 | 0 | 0 | NA | Unavailable |
| Reddit-S | image | 1 | NA [NA, NA] | NA | 0 | 0 | 0 | NA | Unavailable |
| Reddit-S | text | 2 | NA [NA, NA] | NA | 0 | 0 | 0 | NA | Unavailable |
| Reddit-S | image | 2 | NA [NA, NA] | NA | 0 | 0 | 0 | NA | Unavailable |
| ele-fashion | text | 0 | 0.0024 [-0.0757, 0.0933] | 0.0033 | 9 | 7459 | 1000 | yes | Weak |
| ele-fashion | image | 0 | -0.0123 [-0.0641, 0.0435] | -0.0241 | 9 | 7459 | 1000 | yes | Mixed |
| ele-fashion | text | 1 | 0.0294 [-0.0668, 0.1652] | 0.0382 | 9 | 7459 | 1000 | yes | Weak |
| ele-fashion | image | 1 | 0.0004 [-0.0639, 0.0671] | 0.0008 | 9 | 7459 | 1000 | no | Mixed |
| ele-fashion | text | 2 | 0.0151 [-0.0403, 0.0779] | 0.0222 | 9 | 7459 | 1000 | yes | Weak |
| ele-fashion | image | 2 | -0.0035 [-0.0510, 0.0484] | -0.0075 | 9 | 7459 | 1000 | yes | Mixed |

Per-dataset modality descriptions:

- **Movies:** text Weak; image Weak; mixed modalities.
- **Toys:** text Strong; image Strong; both modalities consistent.
- **Grocery:** text Weak; image Mixed; mixed modalities.
- **Reddit-S:** text Unavailable; image Unavailable; unavailable.
- **ele-fashion:** text Weak; image Mixed; mixed modalities.

Reddit-S remains unavailable because no report interactions form legal high/low embeddedness pairs within the specified strata. No bin threshold or jitter was changed.

Alpha 0.05/0.2 point estimates are reported as sensitivity checks. `sign_consistent_across_alpha` records whether a seed's deterministic matched-effect sign agrees at 0.05, 0.1, and 0.2; it does not replace the alpha=0.1 receiver-bootstrap grade. V1 repeated-subsample intervals appear only as historical diagnostics in the CSV and are not used for V1.1 evidence grades.

| Dataset | Modality | Seeds with same matched-effect sign / available seeds at α=.05/.1/.2 |
|---|---|---:|
| Movies | text | 3/3 |
| Movies | image | 3/3 |
| Toys | text | 3/3 |
| Toys | image | 3/3 |
| Grocery | text | 3/3 |
| Grocery | image | 2/3 |
| Reddit-S | text | 0/0 |
| Reddit-S | image | 0/0 |
| ele-fashion | text | 3/3 |
| ele-fashion | image | 2/3 |

## 6. Operator preference and matched random-action null

Functional validation profiles are reused from V1 in the same canonical validation-node order. The functional fixed action is the V1 calibration-selected action. Null calibration/report profiles are computed from the frozen host using the same graph source counts, alpha, equal-relative-norm rule, head, and four actions. Stability is reported for both banks. Margin-filtered subsets affect only the stability summaries.

Top-action agreement is shown as the mean over the three seed pairs; per-pair values and Cohen kappa, per-action Pearson/Spearman profile correlations, and centered four-action profile cosine are in `operator_stability.csv`.

| Dataset | Modality | Bank | Pair agreement | Pair kappa | Unanimous top action | At least 2/3 agree | Profile cosine mean | Agreement if both margins ≥1e−3 | Agreement in both top-half margin sets |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Movies | text | functional | 0.7184 | 0.5838 | 0.5996 | 0.9559 | 0.7483 | 0.7466 | 0.9270 |
| Movies | text | matched_random_null | 0.7237 | 0.5737 | 0.6125 | 0.9460 | 0.7297 | 0.7563 | 0.9296 |
| Movies | image | functional | 0.7623 | 0.6378 | 0.6575 | 0.9721 | 0.8091 | 0.7835 | 0.9416 |
| Movies | image | matched_random_null | 0.7662 | 0.6517 | 0.6638 | 0.9712 | 0.8310 | 0.8002 | 0.9528 |
| Toys | text | functional | 0.7450 | 0.6081 | 0.6357 | 0.9638 | 0.8042 | 0.7963 | 0.8714 |
| Toys | text | matched_random_null | 0.6907 | 0.5378 | 0.5668 | 0.9386 | 0.6990 | 0.7748 | 0.8477 |
| Toys | image | functional | 0.7639 | 0.6082 | 0.6632 | 0.9652 | 0.8094 | 0.8169 | 0.8931 |
| Toys | image | matched_random_null | 0.6822 | 0.5274 | 0.5547 | 0.9372 | 0.6838 | 0.7596 | 0.8283 |
| Grocery | text | functional | 0.7330 | 0.5782 | 0.6199 | 0.9593 | 0.8002 | 0.7991 | 0.8530 |
| Grocery | text | matched_random_null | 0.6576 | 0.4911 | 0.5268 | 0.9192 | 0.6658 | 0.7520 | 0.8236 |
| Grocery | image | functional | 0.7691 | 0.6026 | 0.6706 | 0.9660 | 0.8141 | 0.8248 | 0.8728 |
| Grocery | image | matched_random_null | 0.6561 | 0.5043 | 0.5151 | 0.9382 | 0.6673 | 0.7309 | 0.7820 |
| Reddit-S | text | functional | 0.7511 | 0.5532 | 0.6360 | 0.9811 | 0.9076 | 0.8604 | 0.8553 |
| Reddit-S | text | matched_random_null | 0.6595 | 0.4836 | 0.5206 | 0.9374 | 0.7742 | 0.7882 | 0.7750 |
| Reddit-S | image | functional | 0.8481 | 0.6180 | 0.7823 | 0.9796 | 0.8744 | 0.9523 | 0.9525 |
| Reddit-S | image | matched_random_null | 0.6374 | 0.4383 | 0.4998 | 0.9126 | 0.6475 | 0.7872 | 0.7886 |
| ele-fashion | text | functional | 0.7040 | 0.5609 | 0.5778 | 0.9563 | 0.7324 | 0.7828 | 0.8213 |
| ele-fashion | text | matched_random_null | 0.7203 | 0.5202 | 0.6089 | 0.9431 | 0.7112 | 0.8083 | 0.8443 |
| ele-fashion | image | functional | 0.6865 | 0.5267 | 0.5506 | 0.9584 | 0.7360 | 0.7642 | 0.7952 |
| ele-fashion | image | matched_random_null | 0.6361 | 0.4731 | 0.4877 | 0.9328 | 0.6524 | 0.7474 | 0.7696 |

Paired validation quantities compare the functional and matched random-null banks node by node. Positive excess headroom means functional adaptive response spread is larger; positive best-action advantage means the best functional response has lower CE; positive global-fixed advantage means the calibration-selected functional action has lower CE. These are frozen-host, label-aware diagnostic quantities, not deployable performance.

| Dataset | Modality | Seed | Functional headroom | Null headroom | Excess headroom | Best-action advantage | Global-fixed advantage |
|---|---|---:|---:|---:|---:|---:|---:|
| Movies | text | 0 | 0.0845 [0.0797, 0.0894] | 0.0831 [0.0804, 0.0858] | 0.0014 [-0.0029, 0.0057] | 0.0038 [0.0014, 0.0063] | 0.0024 [-0.0018, 0.0062] |
| Movies | text | 1 | 0.0816 [0.0771, 0.0864] | 0.0797 [0.0772, 0.0822] | 0.0019 [-0.0026, 0.0066] | 0.0045 [0.0021, 0.0067] | 0.0026 [-0.0012, 0.0070] |
| Movies | text | 2 | 0.0867 [0.0819, 0.0915] | 0.0875 [0.0847, 0.0904] | -0.0008 [-0.0054, 0.0038] | 0.0020 [-0.0004, 0.0045] | 0.0028 [-0.0014, 0.0067] |
| Movies | image | 0 | 0.0530 [0.0497, 0.0565] | 0.0767 [0.0744, 0.0790] | -0.0237 [-0.0272, -0.0200] | 0.0108 [0.0084, 0.0132] | 0.0345 [0.0309, 0.0380] |
| Movies | image | 1 | 0.0544 [0.0508, 0.0582] | 0.0779 [0.0741, 0.0822] | -0.0235 [-0.0278, -0.0193] | 0.0103 [0.0078, 0.0130] | 0.0338 [0.0296, 0.0378] |
| Movies | image | 2 | 0.0543 [0.0509, 0.0578] | 0.0815 [0.0790, 0.0839] | -0.0272 [-0.0311, -0.0232] | 0.0111 [0.0085, 0.0139] | 0.0383 [0.0341, 0.0419] |
| Toys | text | 0 | 0.0424 [0.0390, 0.0458] | 0.0430 [0.0399, 0.0461] | -0.0006 [-0.0048, 0.0037] | 0.0203 [0.0181, 0.0226] | 0.0209 [0.0164, 0.0255] |
| Toys | text | 1 | 0.0408 [0.0377, 0.0443] | 0.0407 [0.0379, 0.0435] | 0.0001 [-0.0041, 0.0044] | 0.0179 [0.0158, 0.0200] | 0.0178 [0.0131, 0.0225] |
| Toys | text | 2 | 0.0428 [0.0396, 0.0459] | 0.0423 [0.0393, 0.0454] | 0.0005 [-0.0039, 0.0049] | 0.0201 [0.0178, 0.0224] | 0.0196 [0.0151, 0.0242] |
| Toys | image | 0 | 0.0351 [0.0323, 0.0381] | 0.0437 [0.0409, 0.0466] | -0.0085 [-0.0114, -0.0057] | 0.0312 [0.0286, 0.0339] | 0.0397 [0.0361, 0.0433] |
| Toys | image | 1 | 0.0325 [0.0299, 0.0350] | 0.0397 [0.0372, 0.0420] | -0.0072 [-0.0098, -0.0045] | 0.0297 [0.0273, 0.0322] | 0.0369 [0.0336, 0.0403] |
| Toys | image | 2 | 0.0350 [0.0323, 0.0378] | 0.0418 [0.0391, 0.0442] | -0.0068 [-0.0098, -0.0037] | 0.0314 [0.0289, 0.0340] | 0.0382 [0.0347, 0.0419] |
| Grocery | text | 0 | 0.0493 [0.0446, 0.0541] | 0.0560 [0.0509, 0.0613] | -0.0066 [-0.0133, -0.0005] | 0.0265 [0.0228, 0.0302] | 0.0331 [0.0256, 0.0399] |
| Grocery | text | 1 | 0.0437 [0.0395, 0.0477] | 0.0480 [0.0441, 0.0523] | -0.0043 [-0.0098, 0.0014] | 0.0224 [0.0188, 0.0260] | 0.0267 [0.0202, 0.0329] |
| Grocery | text | 2 | 0.0510 [0.0460, 0.0559] | 0.0550 [0.0501, 0.0605] | -0.0040 [-0.0108, 0.0027] | 0.0228 [0.0191, 0.0262] | 0.0268 [0.0200, 0.0338] |
| Grocery | image | 0 | 0.0390 [0.0348, 0.0438] | 0.0526 [0.0489, 0.0566] | -0.0136 [-0.0180, -0.0094] | 0.0360 [0.0326, 0.0395] | 0.0496 [0.0449, 0.0547] |
| Grocery | image | 1 | 0.0343 [0.0307, 0.0383] | 0.0453 [0.0418, 0.0490] | -0.0110 [-0.0147, -0.0072] | 0.0335 [0.0302, 0.0369] | 0.0444 [0.0397, 0.0491] |
| Grocery | image | 2 | 0.0369 [0.0333, 0.0409] | 0.0485 [0.0449, 0.0524] | -0.0116 [-0.0158, -0.0071] | 0.0374 [0.0337, 0.0413] | 0.0490 [0.0439, 0.0539] |
| Reddit-S | text | 0 | 0.0134 [0.0110, 0.0158] | 0.0205 [0.0184, 0.0226] | -0.0071 [-0.0096, -0.0045] | 0.0149 [0.0120, 0.0181] | 0.0220 [0.0182, 0.0261] |
| Reddit-S | text | 1 | 0.0120 [0.0099, 0.0142] | 0.0203 [0.0182, 0.0224] | -0.0083 [-0.0107, -0.0062] | 0.0145 [0.0116, 0.0178] | 0.0228 [0.0192, 0.0269] |
| Reddit-S | text | 2 | 0.0119 [0.0098, 0.0142] | 0.0193 [0.0173, 0.0212] | -0.0074 [-0.0095, -0.0051] | 0.0153 [0.0123, 0.0183] | 0.0227 [0.0190, 0.0268] |
| Reddit-S | image | 0 | 0.0131 [0.0106, 0.0158] | 0.0182 [0.0152, 0.0214] | -0.0051 [-0.0087, -0.0016] | 0.0336 [0.0295, 0.0381] | 0.0387 [0.0331, 0.0447] |
| Reddit-S | image | 1 | 0.0112 [0.0088, 0.0136] | 0.0165 [0.0147, 0.0185] | -0.0053 [-0.0076, -0.0030] | 0.0310 [0.0271, 0.0351] | 0.0363 [0.0316, 0.0409] |
| Reddit-S | image | 2 | 0.0124 [0.0097, 0.0150] | 0.0169 [0.0145, 0.0192] | -0.0045 [-0.0076, -0.0015] | 0.0336 [0.0297, 0.0378] | 0.0381 [0.0330, 0.0429] |
| ele-fashion | text | 0 | 0.0283 [0.0267, 0.0299] | 0.0283 [0.0272, 0.0295] | 0.0000 [-0.0015, 0.0015] | 0.0058 [0.0047, 0.0068] | 0.0058 [0.0044, 0.0072] |
| ele-fashion | text | 1 | 0.0298 [0.0281, 0.0315] | 0.0287 [0.0276, 0.0299] | 0.0010 [-0.0006, 0.0027] | 0.0072 [0.0061, 0.0082] | 0.0061 [0.0047, 0.0076] |
| ele-fashion | text | 2 | 0.0292 [0.0275, 0.0309] | 0.0280 [0.0268, 0.0291] | 0.0012 [-0.0005, 0.0028] | 0.0069 [0.0059, 0.0078] | 0.0057 [0.0042, 0.0071] |
| ele-fashion | image | 0 | 0.0244 [0.0230, 0.0258] | 0.0223 [0.0215, 0.0233] | 0.0021 [0.0008, 0.0034] | 0.0058 [0.0051, 0.0066] | 0.0037 [0.0027, 0.0048] |
| ele-fashion | image | 1 | 0.0260 [0.0245, 0.0275] | 0.0248 [0.0239, 0.0258] | 0.0012 [-0.0002, 0.0026] | 0.0053 [0.0045, 0.0061] | 0.0041 [0.0030, 0.0052] |
| ele-fashion | image | 2 | 0.0233 [0.0220, 0.0247] | 0.0220 [0.0211, 0.0228] | 0.0014 [-0.0000, 0.0027] | 0.0052 [0.0045, 0.0060] | 0.0039 [0.0028, 0.0049] |

## 7. Cross-dataset consistency

| Question | Dataset-level observations |
|---|---|
| P1 adjusted gap CI lower bound > 0 | 5/5 datasets |
| P2 `cross_only_graph_gain` grades across dataset × direction | {'Strong': 10} |
| P2 `conditional_graph_gain` grades across dataset × direction | {'Strong': 9, 'Mixed': 1} |
| P2 `conditional_cross_delta_graph` grades across dataset × direction | {'Strong': 7, 'Mixed': 3} |
| P3 text modality grades | {'Weak': 3, 'Strong': 1, 'Unavailable': 1} |
| P3 image modality grades | {'Weak': 1, 'Strong': 1, 'Mixed': 2, 'Unavailable': 1} |

## 8. Corrections relative to V1

- **P1:** the corrected point estimate is compared directly with V1; because V1 stores edge-level source-averaged percentiles, reassigning receiver orientation leaves that point estimate unchanged. The receiver-block interval is recomputed. Whether the gap clears reference-sampling instability is shown by each dataset's adjusted-gap interval, without a magnitude cutoff.
- **P2:** the graph-versus-random cross-only baseline is retained, while the new conditional contrasts ask what remains after the same-modal graph perturbation. Conditional results that differ from V1's cross-only pattern are explicitly weakened or unresolved at that dataset/direction; no universal direction is imposed.
- **P3:** V1's repeated-subsample intervals are historical only. The primary conclusions now depend on deterministic weighted matching and receiver-block intervals, with text and image graded separately. Results outside the V1 direction or with unavailable strata are weakened or unavailable under the corrected uncertainty.
- **Operator:** V1's response profiles remain descriptive. V1.1 adds seed agreement and a matched random-action comparison; stability and paired headroom are reported as measurements, without a GO/NO-GO threshold.

### Observed disposition from V1.1
- **P1:** the original point magnitude is retained by construction for all analyzed datasets. The adjusted-gap interval is above zero for ['Movies', 'Toys', 'Grocery', 'Reddit-S', 'ele-fashion']; it is not above zero for none. The V1 gap magnitude remains a descriptive fact; evidence beyond reference sampling noise is dataset-specific.
- **P2:** conditional graph utility is retained in ['Movies T2I', 'Toys T2I', 'Toys I2T', 'Grocery T2I', 'Reddit-S T2I', 'Reddit-S I2T'] under positive conditional graph gain and negative graph cross delta. Opposite-signed receiver intervals occur in no cells; the remaining cells are weakened or unresolved: ['Movies I2T', 'Grocery I2T', 'ele-fashion T2I', 'ele-fashion I2T']. This is a per-cell description, not a cross-dataset rule.
- **P3:** Movies/text: weakened or unresolved under receiver-block uncertainty (Weak); Movies/image: weakened or unresolved under receiver-block uncertainty (Weak); Toys/text: remains directionally consistent (Strong); Toys/image: remains directionally consistent (Strong); Grocery/text: weakened or unresolved under receiver-block uncertainty (Weak); Grocery/image: weakened or unresolved under receiver-block uncertainty (Mixed); Reddit-S/text: unavailable (Unavailable); Reddit-S/image: unavailable (Unavailable); ele-fashion/text: weakened or unresolved under receiver-block uncertainty (Weak); ele-fashion/image: weakened or unresolved under receiver-block uncertainty (Mixed).
- **Operator vs matched null:** functional excess-headroom intervals are wholly positive in ['ele-fashion/image/seed0'], wholly negative in ['Movies/image/seed0', 'Movies/image/seed1', 'Movies/image/seed2', 'Toys/image/seed0', 'Toys/image/seed1', 'Toys/image/seed2', 'Grocery/text/seed0', 'Grocery/image/seed0', 'Grocery/image/seed1', 'Grocery/image/seed2', 'Reddit-S/text/seed0', 'Reddit-S/image/seed0', 'Reddit-S/text/seed1', 'Reddit-S/image/seed1', 'Reddit-S/text/seed2', 'Reddit-S/image/seed2'], and include zero in ['Movies/text/seed0', 'Movies/text/seed1', 'Movies/text/seed2', 'Toys/text/seed0', 'Toys/text/seed1', 'Toys/text/seed2', 'Grocery/text/seed1', 'Grocery/text/seed2', 'ele-fashion/text/seed0', 'ele-fashion/text/seed1', 'ele-fashion/image/seed1', 'ele-fashion/text/seed2', 'ele-fashion/image/seed2']. Best-action and global-fixed advantages remain separately reported; the data do not choose a deployment model.

## 9. Threats to validity

- This is a post-hoc audit, and it reuses the same validation data as V1.
- The diagnostic host is frozen, graph-free, and defines the response surface; another host may produce different responses.
- The matched nonneighbor bank is one null design and does not represent every possible control.
- Per-node response profiles and adaptive best-action quantities use validation labels and are not deployable router outputs.
- Three host seeds are robustness seeds, not statistical population replicates; no seed-level t-test is used.
- The study remains transductive with respect to graph topology and uses the V1 validation split, so results do not establish performance on a new split or an end-to-end model.

## 10. Final factual evidence table

| Question | Evidence reported | Uncertainty unit | Interpretation boundary |
|---|---|---|---|
| P1 gap vs reference noise | Corrected raw-feature gap, five-reference cross signal, modality instability, conservative adjusted gap | Validation receiver blocks | Source-matched percentile estimator and selected graph only |
| P2 directional utility | Cross-only and conditional factorial contrasts at α=.05/.1/.2, separate T2I/I2T | Same receiver averaged across host seeds, then receiver bootstrap | Frozen host local loss response |
| P3 structure response | Deterministic matched effect, standardized magnitude, receiver-block interval, text/image grades | Validation receiver blocks; re-match each draw | Structural bins and report interactions inherited from V1 |
| Functional vs null actions | Agreement, kappa, profile correlation/cosine, margin subgroups, paired headroom/advantages | Nodewise profile stability; receiver bootstrap for paired means | Label-aware frozen-host diagnostic response |

No final model or router is recommended by this audit. Full protocol, exact seed rows, bootstrap counts, and compact CSVs are provided beside this report.
