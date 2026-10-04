# MAG Problem Validation / Interaction Study V1

Status: **complete** (15/15 formal dataset-seed runs).

## 1. Exact protocol

- Host: graph-free, separate text/image projectors, hidden_dim=128, AdamW lr=0.001, weight_decay=0.0001, epochs≤200, patience=20; seeds=[0, 1, 2].
- Split: deterministic per-class 80/20 split of original train into diagnostic_fit/calib; original validation is the report split.
- Isolation: test_label_access=false; test_metric_access=false. Test indices are hashed/masked; no test label values or test metrics are used.
- Full transductive topology and unlabelled node features are retained. No operator-specific classifier is trained.
- P1: source_matched_nonedge_cosine_percentile; raw features primary, host projected states robustness; thresholds are fit from train-endpoint edges.
- P2: neighbor_cap=32, alphas=[0.05, 0.1, 0.2]; gain=random-control delta CE − graph-neighbor delta CE.
- P3: directed interactions capped at 50000 per receiver split; alphas=[0.05, 0.1, 0.2]; train-edge thresholds; receiver-block bootstrap.
- Cross-modal alignment: train-only ridge λ=1.0; quality assessed on calibration and validation.

## 2. Dataset audit

| Dataset | Nodes | Directed edges | Text/Image dim | Train/val/test | Mean/median/p95/max degree | Raw feature GB | Estimated GPU peak GB | Status |
|---|---:|---:|---|---|---|---:|---:|---|
| Movies | 16672 | 160802 | 768/768 | 10003/3334/3335 | 9.6450/5.0000/33.0000/212.0000 | 0.1024 | 0.2506 | formal study |
| Toys | 20695 | 113402 | 768/768 | 12417/4139/4139 | 5.4797/3.0000/19.0000/179.0000 | 0.1272 | 0.3110 | formal study |
| Grocery | 17074 | 142262 | 768/768 | 10244/3415/3415 | 8.3321/4.0000/30.0000/429.0000 | 0.1049 | 0.2566 | formal study |
| Reddit-S | 15894 | 283080 | 768/768 | 9536/3179/3179 | 17.8105/3.0000/131.0000/191.0000 | 0.0977 | 0.2389 | formal study |
| ele-fashion | 97766 | 399172 | 512/512 | 58659/9777/29330 | 4.0829/2.0000/13.0000/661.0000 | 0.4004 | 1.0754 | formal study |
| books-nc | 685294 | 13682358 | 512/512 | 406689/67805/203428 | 19.9657/13.0000/63.0000/1698.0000 | 2.8070 | 7.4855 | audit only |

Books-nc was audited only; a full study is excluded by the protocol.

## 3. Diagnostic host quality

| Dataset | Seed | Fit Acc/F1 | Calib Acc/F1 | Validation Acc/F1 | Majority val Acc | Low quality |
|---|---:|---|---|---|---:|---|
| Grocery | 0 | 0.9144/0.8560 | 0.7689/0.6799 | 0.7722/0.6758 | 0.1414 | False |
| Grocery | 1 | 0.9343/0.8916 | 0.7782/0.6955 | 0.7716/0.6830 | 0.1414 | False |
| Grocery | 2 | 0.9047/0.8320 | 0.7694/0.6699 | 0.7681/0.6806 | 0.1414 | False |
| Movies | 0 | 0.6709/0.5423 | 0.5230/0.3696 | 0.5087/0.3367 | 0.3293 | False |
| Movies | 1 | 0.6695/0.5528 | 0.5125/0.3710 | 0.5063/0.3441 | 0.3293 | False |
| Movies | 2 | 0.6846/0.5713 | 0.5050/0.3595 | 0.5024/0.3548 | 0.3293 | False |
| Reddit-S | 0 | 0.9916/0.9796 | 0.9244/0.8622 | 0.9210/0.8706 | 0.0856 | False |
| Reddit-S | 1 | 0.9906/0.9765 | 0.9228/0.8595 | 0.9239/0.8750 | 0.0856 | False |
| Reddit-S | 2 | 0.9883/0.9698 | 0.9249/0.8607 | 0.9207/0.8728 | 0.0856 | False |
| Toys | 0 | 0.8670/0.8520 | 0.7271/0.6987 | 0.7367/0.7070 | 0.1034 | False |
| Toys | 1 | 0.8568/0.8369 | 0.7299/0.6952 | 0.7415/0.7154 | 0.1034 | False |
| Toys | 2 | 0.8574/0.8390 | 0.7267/0.6972 | 0.7376/0.7069 | 0.1034 | False |
| ele-fashion | 0 | 0.8981/0.7448 | 0.8627/0.6742 | 0.8693/0.6796 | 0.2865 | False |
| ele-fashion | 1 | 0.8973/0.7446 | 0.8612/0.6712 | 0.8667/0.6709 | 0.2865 | False |
| ele-fashion | 2 | 0.8958/0.7435 | 0.8618/0.6747 | 0.8685/0.6767 | 0.2865 | False |

Low host quality means report accuracy is less than 0.02 above the fit-derived majority baseline; this flags lower response confidence and does not select models.

## 4. P1 — modality-specific edge interaction heterogeneity

- Grocery: Spearman=0.1740; raw mean/median |pT−pI|=0.3035/0.2500; receiver-block CI=[0.2993, 0.3068]; projected mean gap across seeds=0.1682; high-T/low-I=0.0390; low-T/high-I=0.0399.
- Movies: Spearman=0.2253; raw mean/median |pT−pI|=0.2899/0.2031; receiver-block CI=[0.2856, 0.2939]; projected mean gap across seeds=0.2193; high-T/low-I=0.0233; low-T/high-I=0.0000.
- Reddit-S: Spearman=0.1014; raw mean/median |pT−pI|=0.2231/0.1250; receiver-block CI=[0.2171, 0.2286]; projected mean gap across seeds=0.1378; high-T/low-I=0.0358; low-T/high-I=0.0000.
- Toys: Spearman=0.1637; raw mean/median |pT−pI|=0.2832/0.1875; receiver-block CI=[0.2787, 0.2873]; projected mean gap across seeds=0.1085; high-T/low-I=0.0000; low-T/high-I=0.0000.
- ele-fashion: Spearman=0.3578; raw mean/median |pT−pI|=0.1329/0.0312; receiver-block CI=[0.1301, 0.1357]; projected mean gap across seeds=0.0976; high-T/low-I=0.0000; low-T/high-I=0.0000.

Decision: **Supported**. Per-dataset support requires raw mean absolute percentile gap ≥ 0.05 and CI lower bound > 0. Raw cosine distributions are not directly compared. Same-label-by-quadrant is descriptive only, not edge utility.

## 5. P2 — directional cross-modal complementarity

- Movies: α=0.1 T2I gain=0.0131 (positive receivers=0.5841), I2T gain=0.0069 (positive receivers=0.5771); T→I−I→T gain=0.0062, mean seed CI=[0.0038, 0.0086], alignment val cosine T→I/I→T=0.6196/0.9385, status=Supported, alignment-low=False.
- Toys: α=0.1 T2I gain=0.0228 (positive receivers=0.7031), I2T gain=0.0212 (positive receivers=0.7234); T→I−I→T gain=0.0016, mean seed CI=[-0.0007, 0.0041], alignment val cosine T→I/I→T=0.7339/0.8142, status=Mixed, alignment-low=False.
- Grocery: α=0.1 T2I gain=0.0259 (positive receivers=0.7483), I2T gain=0.0293 (positive receivers=0.7542); T→I−I→T gain=-0.0034, mean seed CI=[-0.0065, -0.0003], alignment val cosine T→I/I→T=0.7955/0.8594, status=Supported, alignment-low=False.
- Reddit-S: α=0.1 T2I gain=0.0197 (positive receivers=0.7113), I2T gain=0.0143 (positive receivers=0.8829); T→I−I→T gain=0.0053, mean seed CI=[0.0028, 0.0079], alignment val cosine T→I/I→T=0.6082/0.9063, status=Supported, alignment-low=False.
- ele-fashion: α=0.1 T2I gain=0.0024 (positive receivers=0.6777), I2T gain=0.0036 (positive receivers=0.7198); T→I−I→T gain=-0.0012, mean seed CI=[-0.0026, 0.0001], alignment val cosine T→I/I→T=0.8289/0.7236, status=Mixed, alignment-low=False.

Decision: **Supported**. Positive graph_gain means real neighbors caused a smaller CE increase than equal-count random sources in the frozen host. Directional asymmetry is computed from graph-minus-random gains. Low alignment quality blocks a Supported label.

## 6. P3 — structure-conditioned interaction response

- Grocery seed 0 text: high-minus-low embeddedness D_same=-0.0658, repeated-subsample interval=[-0.0722, -0.0593], matched strata=32, interactions/arm=5114.
- Grocery seed 0 image: high-minus-low embeddedness D_same=-0.0176, repeated-subsample interval=[-0.0212, -0.0140], matched strata=32, interactions/arm=5114.
- Grocery seed 1 text: high-minus-low embeddedness D_same=-0.0309, repeated-subsample interval=[-0.0344, -0.0275], matched strata=32, interactions/arm=5114.
- Grocery seed 1 image: high-minus-low embeddedness D_same=0.0408, repeated-subsample interval=[0.0368, 0.0447], matched strata=32, interactions/arm=5114.
- Grocery seed 2 text: high-minus-low embeddedness D_same=-0.0556, repeated-subsample interval=[-0.0601, -0.0512], matched strata=32, interactions/arm=5114.
- Grocery seed 2 image: high-minus-low embeddedness D_same=-0.0039, repeated-subsample interval=[-0.0099, 0.0021], matched strata=32, interactions/arm=5114.
- Movies seed 0 text: high-minus-low embeddedness D_same=-0.0289, repeated-subsample interval=[-0.0315, -0.0262], matched strata=32, interactions/arm=6790.
- Movies seed 0 image: high-minus-low embeddedness D_same=-0.0374, repeated-subsample interval=[-0.0396, -0.0351], matched strata=32, interactions/arm=6790.
- Movies seed 1 text: high-minus-low embeddedness D_same=-0.0215, repeated-subsample interval=[-0.0244, -0.0186], matched strata=32, interactions/arm=6790.
- Movies seed 1 image: high-minus-low embeddedness D_same=-0.0558, repeated-subsample interval=[-0.0579, -0.0537], matched strata=32, interactions/arm=6790.
- Movies seed 2 text: high-minus-low embeddedness D_same=-0.0618, repeated-subsample interval=[-0.0651, -0.0586], matched strata=32, interactions/arm=6790.
- Movies seed 2 image: high-minus-low embeddedness D_same=-0.0327, repeated-subsample interval=[-0.0355, -0.0299], matched strata=32, interactions/arm=6790.
- Reddit-S seed 0 text: high-minus-low embeddedness D_same=None, repeated-subsample interval=[None, None], matched strata=0, interactions/arm=0.
- Reddit-S seed 0 image: high-minus-low embeddedness D_same=None, repeated-subsample interval=[None, None], matched strata=0, interactions/arm=0.
- Reddit-S seed 1 text: high-minus-low embeddedness D_same=None, repeated-subsample interval=[None, None], matched strata=0, interactions/arm=0.
- Reddit-S seed 1 image: high-minus-low embeddedness D_same=None, repeated-subsample interval=[None, None], matched strata=0, interactions/arm=0.
- Reddit-S seed 2 text: high-minus-low embeddedness D_same=None, repeated-subsample interval=[None, None], matched strata=0, interactions/arm=0.
- Reddit-S seed 2 image: high-minus-low embeddedness D_same=None, repeated-subsample interval=[None, None], matched strata=0, interactions/arm=0.
- Toys seed 0 text: high-minus-low embeddedness D_same=-0.1019, repeated-subsample interval=[-0.1063, -0.0975], matched strata=18, interactions/arm=4187.
- Toys seed 0 image: high-minus-low embeddedness D_same=-0.0864, repeated-subsample interval=[-0.0897, -0.0830], matched strata=9, interactions/arm=4220.
- Toys seed 1 text: high-minus-low embeddedness D_same=-0.0881, repeated-subsample interval=[-0.0919, -0.0843], matched strata=18, interactions/arm=4187.
- Toys seed 1 image: high-minus-low embeddedness D_same=-0.1077, repeated-subsample interval=[-0.1112, -0.1042], matched strata=9, interactions/arm=4220.
- Toys seed 2 text: high-minus-low embeddedness D_same=-0.0999, repeated-subsample interval=[-0.1037, -0.0961], matched strata=18, interactions/arm=4187.
- Toys seed 2 image: high-minus-low embeddedness D_same=-0.1100, repeated-subsample interval=[-0.1145, -0.1055], matched strata=9, interactions/arm=4220.
- ele-fashion seed 0 text: high-minus-low embeddedness D_same=0.0013, repeated-subsample interval=[0.0000, 0.0026], matched strata=9, interactions/arm=7459.
- ele-fashion seed 0 image: high-minus-low embeddedness D_same=-0.0112, repeated-subsample interval=[-0.0123, -0.0101], matched strata=9, interactions/arm=7459.
- ele-fashion seed 1 text: high-minus-low embeddedness D_same=0.0294, repeated-subsample interval=[0.0272, 0.0316], matched strata=9, interactions/arm=7459.
- ele-fashion seed 1 image: high-minus-low embeddedness D_same=0.0010, repeated-subsample interval=[-0.0003, 0.0022], matched strata=9, interactions/arm=7459.
- ele-fashion seed 2 text: high-minus-low embeddedness D_same=0.0150, repeated-subsample interval=[0.0134, 0.0165], matched strata=9, interactions/arm=7459.
- ele-fashion seed 2 image: high-minus-low embeddedness D_same=-0.0030, repeated-subsample interval=[-0.0044, -0.0016], matched strata=9, interactions/arm=7459.
- Decision: **Supported**. The primary table compares D_same across embeddedness quartiles within each fixed semantic bin. Matching controls semantic bin plus source/receiver degree quartiles. D_same < 0 means moving toward the same-modality neighbor lowers local loss; D_same > 0 means it raises loss. This is frozen-host local response, not a causal effect.

### Incremental explanatory signal

- Grocery seed 0 text: validation R² semantic/local-context=-0.0079, plus structure=-0.0036, ΔR²=0.0042; MAE 0.7533→0.7553. Not router learnability.
- Grocery seed 0 image: validation R² semantic/local-context=-0.0351, plus structure=-0.0314, ΔR²=0.0038; MAE 0.8227→0.8287. Not router learnability.
- Grocery seed 1 text: validation R² semantic/local-context=-0.0074, plus structure=-0.0028, ΔR²=0.0046; MAE 0.6907→0.6944. Not router learnability.
- Grocery seed 1 image: validation R² semantic/local-context=-0.0458, plus structure=-0.0415, ΔR²=0.0043; MAE 0.7518→0.7558. Not router learnability.
- Grocery seed 2 text: validation R² semantic/local-context=-0.0043, plus structure=-0.0006, ΔR²=0.0037; MAE 0.7577→0.7585. Not router learnability.
- Grocery seed 2 image: validation R² semantic/local-context=-0.0354, plus structure=-0.0303, ΔR²=0.0051; MAE 0.8114→0.8160. Not router learnability.
- Movies seed 0 text: validation R² semantic/local-context=-0.0032, plus structure=-0.0038, ΔR²=-0.0006; MAE 0.7724→0.7783. Not router learnability.
- Movies seed 0 image: validation R² semantic/local-context=-0.0292, plus structure=-0.0279, ΔR²=0.0012; MAE 0.7521→0.7542. Not router learnability.
- Movies seed 1 text: validation R² semantic/local-context=-0.0054, plus structure=-0.0099, ΔR²=-0.0045; MAE 0.7484→0.7546. Not router learnability.
- Movies seed 1 image: validation R² semantic/local-context=-0.0266, plus structure=-0.0234, ΔR²=0.0033; MAE 0.7696→0.7706. Not router learnability.
- Movies seed 2 text: validation R² semantic/local-context=-0.0040, plus structure=-0.0056, ΔR²=-0.0016; MAE 0.8180→0.8231. Not router learnability.
- Movies seed 2 image: validation R² semantic/local-context=-0.0259, plus structure=-0.0254, ΔR²=0.0005; MAE 0.7888→0.7900. Not router learnability.
- Reddit-S seed 0 text: validation R² semantic/local-context=-0.0107, plus structure=-0.0092, ΔR²=0.0015; MAE 0.1433→0.1434. Not router learnability.
- Reddit-S seed 0 image: validation R² semantic/local-context=0.0046, plus structure=0.0046, ΔR²=0.0000; MAE 0.2653→0.2653. Not router learnability.
- Reddit-S seed 1 text: validation R² semantic/local-context=-0.0110, plus structure=-0.0099, ΔR²=0.0011; MAE 0.1416→0.1418. Not router learnability.
- Reddit-S seed 1 image: validation R² semantic/local-context=0.0086, plus structure=0.0092, ΔR²=0.0005; MAE 0.2576→0.2574. Not router learnability.
- Reddit-S seed 2 text: validation R² semantic/local-context=-0.0124, plus structure=-0.0106, ΔR²=0.0018; MAE 0.1386→0.1392. Not router learnability.
- Reddit-S seed 2 image: validation R² semantic/local-context=0.0161, plus structure=0.0169, ΔR²=0.0009; MAE 0.2549→0.2550. Not router learnability.
- Toys seed 0 text: validation R² semantic/local-context=-0.0091, plus structure=-0.0096, ΔR²=-0.0005; MAE 0.5950→0.5966. Not router learnability.
- Toys seed 0 image: validation R² semantic/local-context=-0.0425, plus structure=-0.0457, ΔR²=-0.0032; MAE 0.6788→0.6865. Not router learnability.
- Toys seed 1 text: validation R² semantic/local-context=-0.0085, plus structure=-0.0113, ΔR²=-0.0029; MAE 0.5860→0.5888. Not router learnability.
- Toys seed 1 image: validation R² semantic/local-context=-0.0359, plus structure=-0.0392, ΔR²=-0.0033; MAE 0.6245→0.6299. Not router learnability.
- Toys seed 2 text: validation R² semantic/local-context=-0.0078, plus structure=-0.0103, ΔR²=-0.0025; MAE 0.5905→0.5941. Not router learnability.
- Toys seed 2 image: validation R² semantic/local-context=-0.0327, plus structure=-0.0335, ΔR²=-0.0008; MAE 0.6490→0.6542. Not router learnability.
- ele-fashion seed 0 text: validation R² semantic/local-context=-0.0005, plus structure=0.0001, ΔR²=0.0007; MAE 0.2946→0.2958. Not router learnability.
- ele-fashion seed 0 image: validation R² semantic/local-context=-0.0021, plus structure=-0.0039, ΔR²=-0.0019; MAE 0.2165→0.2173. Not router learnability.
- ele-fashion seed 1 text: validation R² semantic/local-context=-0.0002, plus structure=0.0006, ΔR²=0.0008; MAE 0.3166→0.3176. Not router learnability.
- ele-fashion seed 1 image: validation R² semantic/local-context=-0.0039, plus structure=-0.0063, ΔR²=-0.0023; MAE 0.2394→0.2394. Not router learnability.
- ele-fashion seed 2 text: validation R² semantic/local-context=-0.0002, plus structure=-0.0030, ΔR²=-0.0028; MAE 0.2832→0.2835. Not router learnability.
- ele-fashion seed 2 image: validation R² semantic/local-context=-0.0033, plus structure=-0.0049, ΔR²=-0.0017; MAE 0.2048→0.2050. Not router learnability.
- Across 30 comparisons, mean ΔR²=0.0003; R² improved in 17/30; MAE improved in 3/30. R² is near or below zero in most rows, so this secondary regression provides little stable validation explanatory signal.

## 7. Operator-response study

- Grocery seed 0 text: calibration-selected fixed action=Same-Smooth; report action ΔCE={"Cross-Neighbor": -0.012114781886339188, "NoOp": 0.0, "Same-Contrast": 0.05207524448633194, "Same-Smooth": -0.03584069013595581}; headroom=0.0493 [0.0444, 0.0539]; median best/second margin=0.0108.
- Grocery seed 0 image: calibration-selected fixed action=Same-Smooth; report action ΔCE={"Cross-Neighbor": -0.03330051526427269, "NoOp": 0.0, "Same-Contrast": 0.07318293303251266, "Same-Smooth": -0.05668531730771065}; headroom=0.0390 [0.0349, 0.0432]; median best/second margin=0.0104.
- Grocery seed 1 text: calibration-selected fixed action=Same-Smooth; report action ΔCE={"Cross-Neighbor": -0.008860521018505096, "NoOp": 0.0, "Same-Contrast": 0.04513058811426163, "Same-Smooth": -0.033024728298187256}; headroom=0.0437 [0.0395, 0.0483]; median best/second margin=0.0091.
- Grocery seed 1 image: calibration-selected fixed action=Same-Smooth; report action ΔCE={"Cross-Neighbor": -0.033839039504528046, "NoOp": 0.0, "Same-Contrast": 0.06585253775119781, "Same-Smooth": -0.05335327982902527}; headroom=0.0343 [0.0311, 0.0380]; median best/second margin=0.0091.
- Grocery seed 2 text: calibration-selected fixed action=Same-Smooth; report action ΔCE={"Cross-Neighbor": -0.007702951785176992, "NoOp": 0.0, "Same-Contrast": 0.05100946128368378, "Same-Smooth": -0.033398523926734924}; headroom=0.0510 [0.0462, 0.0562]; median best/second margin=0.0113.
- Grocery seed 2 image: calibration-selected fixed action=Same-Smooth; report action ΔCE={"Cross-Neighbor": -0.03586304187774658, "NoOp": 0.0, "Same-Contrast": 0.07472304254770279, "Same-Smooth": -0.05827798321843147}; headroom=0.0369 [0.0330, 0.0409]; median best/second margin=0.0113.
- Movies seed 0 text: calibration-selected fixed action=Same-Smooth; report action ΔCE={"Cross-Neighbor": 0.00579537870362401, "NoOp": 0.0, "Same-Contrast": 0.01265446562319994, "Same-Smooth": -0.002361885504797101}; headroom=0.0845 [0.0794, 0.0895]; median best/second margin=0.0303.
- Movies seed 0 image: calibration-selected fixed action=Same-Smooth; report action ΔCE={"Cross-Neighbor": -0.014702926389873028, "NoOp": 0.0, "Same-Contrast": 0.04026584327220917, "Same-Smooth": -0.03449088707566261}; headroom=0.0530 [0.0496, 0.0564]; median best/second margin=0.0334.
- Movies seed 1 text: calibration-selected fixed action=Same-Smooth; report action ΔCE={"Cross-Neighbor": 0.005488058552145958, "NoOp": 0.0, "Same-Contrast": 0.012743913568556309, "Same-Smooth": -0.002617280464619398}; headroom=0.0816 [0.0772, 0.0861]; median best/second margin=0.0284.
- Movies seed 1 image: calibration-selected fixed action=Same-Smooth; report action ΔCE={"Cross-Neighbor": -0.013576364144682884, "NoOp": 0.0, "Same-Contrast": 0.0405798964202404, "Same-Smooth": -0.034350644797086716}; headroom=0.0544 [0.0509, 0.0579]; median best/second margin=0.0318.
- Movies seed 2 text: calibration-selected fixed action=Same-Smooth; report action ΔCE={"Cross-Neighbor": 0.00666075199842453, "NoOp": 0.0, "Same-Contrast": 0.013495068997144699, "Same-Smooth": -0.0027993195690214634}; headroom=0.0867 [0.0818, 0.0914]; median best/second margin=0.0320.
- Movies seed 2 image: calibration-selected fixed action=Same-Smooth; report action ΔCE={"Cross-Neighbor": -0.016866443678736687, "NoOp": 0.0, "Same-Contrast": 0.04440226033329964, "Same-Smooth": -0.038296300917863846}; headroom=0.0543 [0.0511, 0.0574]; median best/second margin=0.0343.
- Reddit-S seed 0 text: calibration-selected fixed action=Same-Smooth; report action ΔCE={"Cross-Neighbor": -0.010152304545044899, "NoOp": 0.0, "Same-Contrast": 0.026594312861561775, "Same-Smooth": -0.021967681124806404}; headroom=0.0134 [0.0112, 0.0159]; median best/second margin=0.0004.
- Reddit-S seed 0 image: calibration-selected fixed action=Same-Smooth; report action ΔCE={"Cross-Neighbor": -0.02024964988231659, "NoOp": 0.0, "Same-Contrast": 0.04645857587456703, "Same-Smooth": -0.03807216137647629}; headroom=0.0131 [0.0104, 0.0159]; median best/second margin=0.0014.
- Reddit-S seed 1 text: calibration-selected fixed action=Same-Smooth; report action ΔCE={"Cross-Neighbor": -0.011091590858995914, "NoOp": 0.0, "Same-Contrast": 0.027640623971819878, "Same-Smooth": -0.022832877933979034}; headroom=0.0120 [0.0100, 0.0142]; median best/second margin=0.0005.
- Reddit-S seed 1 image: calibration-selected fixed action=Same-Smooth; report action ΔCE={"Cross-Neighbor": -0.018273938447237015, "NoOp": 0.0, "Same-Contrast": 0.04405071586370468, "Same-Smooth": -0.036318469792604446}; headroom=0.0112 [0.0091, 0.0137]; median best/second margin=0.0014.
- Reddit-S seed 2 text: calibration-selected fixed action=Same-Smooth; report action ΔCE={"Cross-Neighbor": -0.011245799250900745, "NoOp": 0.0, "Same-Contrast": 0.026998737826943398, "Same-Smooth": -0.02265581302344799}; headroom=0.0119 [0.0096, 0.0144]; median best/second margin=0.0005.
- Reddit-S seed 2 image: calibration-selected fixed action=Same-Smooth; report action ΔCE={"Cross-Neighbor": -0.019877934828400612, "NoOp": 0.0, "Same-Contrast": 0.046313874423503876, "Same-Smooth": -0.03757254779338837}; headroom=0.0124 [0.0096, 0.0151]; median best/second margin=0.0016.
- Toys seed 0 text: calibration-selected fixed action=Same-Smooth; report action ΔCE={"Cross-Neighbor": -0.0099410405382514, "NoOp": 0.0, "Same-Contrast": 0.033073220402002335, "Same-Smooth": -0.02440720610320568}; headroom=0.0424 [0.0390, 0.0459]; median best/second margin=0.0114.
- Toys seed 0 image: calibration-selected fixed action=Same-Smooth; report action ΔCE={"Cross-Neighbor": -0.028100090101361275, "NoOp": 0.0, "Same-Contrast": 0.0540386326611042, "Same-Smooth": -0.044618573039770126}; headroom=0.0351 [0.0322, 0.0380]; median best/second margin=0.0131.
- Toys seed 1 text: calibration-selected fixed action=Same-Smooth; report action ΔCE={"Cross-Neighbor": -0.008548415265977383, "NoOp": 0.0, "Same-Contrast": 0.03059248998761177, "Same-Smooth": -0.022455953061580658}; headroom=0.0408 [0.0376, 0.0445]; median best/second margin=0.0104.
- Toys seed 1 image: calibration-selected fixed action=Same-Smooth; report action ΔCE={"Cross-Neighbor": -0.027132924646139145, "NoOp": 0.0, "Same-Contrast": 0.04978835955262184, "Same-Smooth": -0.04172572121024132}; headroom=0.0325 [0.0296, 0.0352]; median best/second margin=0.0125.
- Toys seed 2 text: calibration-selected fixed action=Same-Smooth; report action ΔCE={"Cross-Neighbor": -0.008453231304883957, "NoOp": 0.0, "Same-Contrast": 0.03212428092956543, "Same-Smooth": -0.02335786446928978}; headroom=0.0428 [0.0395, 0.0465]; median best/second margin=0.0112.
- Toys seed 2 image: calibration-selected fixed action=Same-Smooth; report action ΔCE={"Cross-Neighbor": -0.027889840304851532, "NoOp": 0.0, "Same-Contrast": 0.05240769311785698, "Same-Smooth": -0.043331339955329895}; headroom=0.0350 [0.0322, 0.0380]; median best/second margin=0.0138.
- ele-fashion seed 0 text: calibration-selected fixed action=Same-Smooth; report action ΔCE={"Cross-Neighbor": -0.001572893699631095, "NoOp": 0.0, "Same-Contrast": 0.010083568282425404, "Same-Smooth": -0.005775908939540386}; headroom=0.0283 [0.0267, 0.0300]; median best/second margin=0.0036.
- ele-fashion seed 0 image: calibration-selected fixed action=Same-Smooth; report action ΔCE={"Cross-Neighbor": -0.002526214113458991, "NoOp": 0.0, "Same-Contrast": 0.0064628589898347855, "Same-Smooth": -0.0037098736502230167}; headroom=0.0244 [0.0229, 0.0259]; median best/second margin=0.0027.
- ele-fashion seed 1 text: calibration-selected fixed action=Same-Smooth; report action ΔCE={"Cross-Neighbor": -0.0027185403741896152, "NoOp": 0.0, "Same-Contrast": 0.010765710845589638, "Same-Smooth": -0.00611905986443162}; headroom=0.0298 [0.0283, 0.0312]; median best/second margin=0.0042.
- ele-fashion seed 1 image: calibration-selected fixed action=Same-Smooth; report action ΔCE={"Cross-Neighbor": -0.0033040884882211685, "NoOp": 0.0, "Same-Contrast": 0.007266581058502197, "Same-Smooth": -0.004119908437132835}; headroom=0.0260 [0.0246, 0.0275]; median best/second margin=0.0031.
- ele-fashion seed 2 text: calibration-selected fixed action=Same-Smooth; report action ΔCE={"Cross-Neighbor": -0.0022500064224004745, "NoOp": 0.0, "Same-Contrast": 0.010021328926086426, "Same-Smooth": -0.0056645809672772884}; headroom=0.0292 [0.0275, 0.0309]; median best/second margin=0.0039.
- ele-fashion seed 2 image: calibration-selected fixed action=Same-Smooth; report action ΔCE={"Cross-Neighbor": -0.002337961457669735, "NoOp": 0.0, "Same-Contrast": 0.00652922922745347, "Same-Smooth": -0.0038741370663046837}; headroom=0.0233 [0.0221, 0.0249]; median best/second margin=0.0027.

The adaptive response upper bound is non-deployable, label-aware, specific to this frozen host and action bank, not an oracle for a future MoE, and does not show an end-to-end router can realize the gap. Argmin action is not a ground-truth operator label.

## 8. Cross-dataset consistency

The overall rule requires at least 3 of five datasets to meet the preregistered dataset criterion. The per-dataset decisions and all seed rows are in the compact CSVs. P2 direction signs, P3 seed consistency, and projected P1 robustness are retained in full.

## 9. Alpha and seed stability

Dataset mean P2 asymmetry is positive for Movies, Toys, Reddit-S and negative for Grocery, ele-fashion; the sign is not universal across datasets.
- P2 alpha=0.05: mean asymmetry=0.0011; finite runs=15/15.
- P2 alpha=0.1: mean asymmetry=0.0017; finite runs=15/15.
- P2 alpha=0.2: mean asymmetry=0.0012; finite runs=15/15.
- P3 Movies: matched effect span across seed×modality=[-0.0618, -0.0215].
- P3 Toys: matched effect span across seed×modality=[-0.1100, -0.0864].
- P3 Grocery: matched effect span across seed×modality=[-0.0658, 0.0408].
- P3 Reddit-S: matched high/low effect unavailable in all seed×modality cells because no common matched strata were available.
- P3 ele-fashion: matched effect span across seed×modality=[-0.0112, 0.0294].
- Low host quality flags: 0/15; low alignment runs: 0/15.

## 10. Failure cases

No formal run failure was recorded.

## 11. Threats to validity

- The graph-free host and frozen classifier define the response space; another host can produce different responses.
- Full graph topology and unlabelled features are transductive; this is not an inductive evaluation.
- Source-matched percentile estimates depend on finite non-edge samples; fixed ridge regularization may fit directions unevenly.
- P3 uses sampled directed edges, train-edge cut points, finite-difference alpha, and a structural embeddedness definition; these choices bound interpretation.
- Receiver-block intervals capture within-dataset receiver clustering, not all between-dataset uncertainty.
- Validation is report-only: host fitting/early stopping and alignment fitting do not use its labels.

## 12. Factual conclusions

- P1: Supported under the raw percentile-gap criterion.
- P2: Supported under graph-versus-random directional asymmetry with an alignment gate.
- P3: Supported under matched high/low embeddedness response and seed consistency.
- Operator results describe the frozen host's response profile; any headroom is a label-aware validation upper bound for this action bank.

## 13. Conclusions not supported by this evidence

- No ground-truth operator, oracle expert, or causal effect is established.
- The results do not establish that an end-to-end router improves NC, generalizes, or realizes the response upper bound.
- Same-label edges are not treated as useful edges, and no test-label-based claim is made.

## Decision Table

| Hypothesis | Supported / Mixed / Unsupported | Evidence | Caveat |
|---|---|---|---|
| P1: modality-specific relation on the same physical edge | Supported | Raw source-matched percentile gap and receiver-block CI | Train-edge bins; sampled non-edge references |
| P2: T→I versus I→T utility asymmetry | Supported | Graph-minus-random gains with source-count matching | Fixed ridge and frozen host; alignment gate |
| P3: structure response after semantic control | Supported | Within-bin D_same tables and matched high/low embeddedness | Sampled local response; not causal |
