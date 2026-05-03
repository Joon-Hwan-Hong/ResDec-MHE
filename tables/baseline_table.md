# Paper Baseline Table

Per-fold metrics are reported as mean ± std (ddof=1) across 5 outer folds unless noted. Rows are grouped into (1) external baselines sorted by R², (2) our canonical + ablations sorted by R². Pending or missing rows are retained so the table is idempotent across re-runs.

| Model | N folds | R² | MAE | RMSE | Pearson r | Spearman ρ | Source | Notes |
|---|---|---|---|---|---|---|---|---|
| TabPFN-2.6 standalone (top-2K features) | 5 | 0.3994 ± 0.1012 | 0.6852 ± 0.0381 | 0.8935 ± 0.0668 | 0.6426 ± 0.0712 | 0.6121 ± 0.0376 | data/redesign | outer-fold R² computed from tabpfn_outer_fold{f}.npz |
| XGBoost [A+C+E] | 5 | 0.3584 ± 0.0531 | 0.7291 ± 0.0533 | 0.9116 ± 0.0701 | 0.6100 ± 0.0468 | 0.5656 ± 0.1058 | outputs/pipeline/baseline_results_classical.csv | feature set: A+C+E |
| XGBoost [A] | 5 | 0.3518 ± 0.0535 | 0.7301 ± 0.0539 | 0.9163 ± 0.0714 | 0.6032 ± 0.0479 | 0.5588 ± 0.0962 | outputs/pipeline/baseline_results_classical.csv | feature set: A |
| RandomForest [A+C+E] | 5 | 0.3136 ± 0.0670 | 0.7464 ± 0.0406 | 0.9418 ± 0.0615 | 0.5749 ± 0.0690 | 0.5183 ± 0.0526 | outputs/pipeline/baseline_results_classical.csv | feature set: A+C+E |
| RandomForest [A] | 5 | 0.3076 ± 0.0669 | 0.7483 ± 0.0409 | 0.9459 ± 0.0622 | 0.5678 ± 0.0693 | 0.5126 ± 0.0492 | outputs/pipeline/baseline_results_classical.csv | feature set: A |
| Ridge [A+C+E] | 5 | 0.2697 ± 0.0815 | 0.7799 ± 0.0629 | 0.9726 ± 0.0933 | 0.5286 ± 0.0927 | 0.5088 ± 0.1050 | outputs/pipeline/baseline_results_classical.csv | feature set: A+C+E |
| Ridge [A] | 5 | 0.2695 ± 0.0815 | 0.7800 ± 0.0630 | 0.9728 ± 0.0933 | 0.5285 ± 0.0927 | 0.5090 ± 0.1048 | outputs/pipeline/baseline_results_classical.csv | feature set: A |
| PLS [A+C+E] | 5 | 0.2694 ± 0.0847 | 0.7808 ± 0.0643 | 0.9729 ± 0.0971 | 0.5275 ± 0.0976 | 0.5110 ± 0.1014 | outputs/pipeline/baseline_results_classical.csv | feature set: A+C+E |
| PLS [A] | 5 | 0.2692 ± 0.0847 | 0.7809 ± 0.0643 | 0.9731 ± 0.0971 | 0.5273 ± 0.0976 | 0.5110 ± 0.1005 | outputs/pipeline/baseline_results_classical.csv | feature set: A |
| MixMIL (Engelmann et al. 2024) | 5 | 0.1572 ± 0.0753 | 0.8546 ± 0.0149 | 1.0608 ± 0.0241 | 0.4098 ± 0.0929 | 0.3883 ± 0.0845 | outputs/baselines/mixmil/results.csv |  |
| GPIO | 5 | 0.1529 ± 0.0701 | 0.8442 ± 0.0288 | 1.0639 ± 0.0345 | 0.3955 ± 0.0909 | 0.3870 ± 0.0945 | outputs/baselines/gpio/results.csv |  |
| ElasticNet [A] | 5 | 0.1287 ± 0.0583 | 0.8625 ± 0.0511 | 1.0623 ± 0.0695 | 0.3937 ± 0.1224 | 0.3874 ± 0.1417 | outputs/pipeline/baseline_results_classical.csv | feature set: A |
| ElasticNet [A+C+E] | 5 | 0.1274 ± 0.0342 | 0.8626 ± 0.0552 | 1.0638 ± 0.0721 | 0.3851 ± 0.0641 | 0.3807 ± 0.0621 | outputs/pipeline/baseline_results_classical.csv | feature set: A+C+E |
| Perceiver-IO | 5 | 0.1246 ± 0.0547 | 0.8820 ± 0.0309 | 1.0822 ± 0.0355 | 0.3620 ± 0.0803 | 0.3503 ± 0.0972 | outputs/baselines/perceiver_io/results.csv |  |
| CloudPred (per-type) | 5 | 0.1027 ± 0.0815 | 0.8703 ± 0.0456 | 1.0955 ± 0.0615 | 0.3141 ± 0.1143 | 0.3108 ± 0.1482 | outputs/baselines/cloudpred_pertype/results.csv |  |
| CloudPred | 5 | 0.0500 ± 0.0251 | 0.9156 ± 0.0235 | 1.1283 ± 0.0453 | 0.2319 ± 0.0537 | 0.2497 ± 0.0751 | outputs/baselines/cloudpred/results.csv |  |
| RandomForest [C] | 5 | 0.0367 ± 0.0774 | 0.8932 ± 0.0768 | 1.1175 ± 0.0914 | 0.2183 ± 0.1298 | 0.2205 ± 0.1249 | outputs/pipeline/baseline_results_classical.csv | feature set: C |
| Ridge [C] | 5 | 0.0139 ± 0.0302 | 0.9130 ± 0.0497 | 1.1302 ± 0.0600 | 0.1767 ± 0.1267 | 0.1860 ± 0.0546 | outputs/pipeline/baseline_results_classical.csv | feature set: C |
| ElasticNet [C] | 5 | -0.0004 ± 0.0232 | 0.9202 ± 0.0520 | 1.1387 ± 0.0663 | 0.1694 ± 0.0592 | 0.2124 ± 0.0062 | outputs/pipeline/baseline_results_classical.csv | feature set: C |
| PLS [C] | 5 | -0.0120 ± 0.0990 | 0.9119 ± 0.0331 | 1.1426 ± 0.0491 | 0.1648 ± 0.1326 | 0.1803 ± 0.0563 | outputs/pipeline/baseline_results_classical.csv | feature set: C |
| XGBoost [C] | 5 | -0.0129 ± 0.0638 | 0.9196 ± 0.0467 | 1.1446 ± 0.0556 | 0.1622 ± 0.0747 | 0.1788 ± 0.0654 | outputs/pipeline/baseline_results_classical.csv | feature set: C |
| scPhase (Berson et al. 2025) | 5 | -0.0742 ± 0.0525 | 0.9377 ± 0.0454 | 1.2008 ± 0.0788 | -0.0315 ± 0.0712 | 0.0709 ± 0.0788 | outputs/baselines/scphase/results.csv |  |
| Ablation: top-k=1000 | 5 | 0.4499 ± 0.0789 | 0.6660 ± 0.0224 | 0.8553 ± 0.0313 | 0.6783 ± 0.0542 | 0.6501 ± 0.0196 | outputs/redesign/p5_ablation_topk_1000/best_vs_tabpfn_summary.json |  |
| ResDec-MHE (canonical, p5_canonical_seed42) | 5 | 0.4436 ± 0.0996 | 0.6697 ± 0.0527 | 0.8592 ± 0.0590 | 0.6723 ± 0.0684 | 0.6646 ± 0.0458 | outputs/redesign/p5_canonical_seed42/best_vs_tabpfn_summary.json |  |
| Ablation: no aug-U n=2 | 5 | 0.4427 ± 0.0881 | 0.6688 ± 0.0458 | 0.8609 ± 0.0556 | 0.6727 ± 0.0547 | 0.6471 ± 0.0401 | outputs/redesign/p5_ablation_no_aug_u_n2/best_vs_tabpfn_summary.json |  |
| Ablation: top-k=4000 | 5 | 0.4404 ± 0.0665 | 0.6600 ± 0.0385 | 0.8637 ± 0.0311 | 0.6762 ± 0.0330 | 0.6523 ± 0.0395 | outputs/redesign/p5_ablation_topk_4000/best_vs_tabpfn_summary.json |  |
| Ablation: per-feature z-score | 5 | 0.4375 ± 0.0819 | 0.6812 ± 0.0248 | 0.8652 ± 0.0455 | 0.6687 ± 0.0542 | 0.6465 ± 0.0279 | outputs/redesign/p5_ablation_zscore/best_vs_tabpfn_summary.json |  |
| ResDec-MHE with DiffAttn | 5 | 0.4373 ± 0.0948 | 0.6678 ± 0.0465 | 0.8646 ± 0.0578 | 0.6651 ± 0.0647 | 0.6407 ± 0.0370 | outputs/redesign/p5_phase3_1stage_with_tabm/best_vs_tabpfn_summary.json |  |
| Ablation: k_tabm=1 | 5 | 0.4342 ± 0.0854 | 0.6863 ± 0.0529 | 0.8676 ± 0.0492 | 0.6681 ± 0.0621 | 0.6427 ± 0.0403 | outputs/redesign/p5_ablation_k1/best_vs_tabpfn_summary.json |  |
| ResDec-MHE + FiLM with real metadata | 5 | 0.4333 ± 0.0835 | 0.6889 ± 0.0337 | 0.8684 ± 0.0481 | 0.6629 ± 0.0582 | 0.6440 ± 0.0319 | outputs/redesign/p5_filmwired_5fold_seed42/best_vs_tabpfn_summary.json |  |
| Ablation: no FiLM | 5 | 0.4328 ± 0.0972 | 0.6731 ± 0.0404 | 0.8680 ± 0.0610 | 0.6706 ± 0.0581 | 0.6470 ± 0.0363 | outputs/redesign/p5_ablation_no_film/best_vs_tabpfn_summary.json |  |
| ResDec-MHE n_stages=3 | 5 | 0.4310 ± 0.0925 | 0.6758 ± 0.0504 | 0.8697 ± 0.0594 | 0.6639 ± 0.0590 | 0.6414 ± 0.0355 | outputs/redesign/p5_phase3_3stage/fold{0..4}/val_predictions_best.npz | per-fold val_predictions_best.npz (summary JSON absent) |
| Ablation: no HyperConn | 5 | 0.4305 ± 0.0902 | 0.6865 ± 0.0415 | 0.8703 ± 0.0568 | 0.6620 ± 0.0616 | 0.6438 ± 0.0433 | outputs/redesign/p5_ablation_no_hyper_conn/best_vs_tabpfn_summary.json |  |
| ResDec-MHE n_stages=2 | 5 | 0.4305 ± 0.0877 | 0.6761 ± 0.0420 | 0.8705 ± 0.0565 | 0.6624 ± 0.0578 | 0.6327 ± 0.0267 | outputs/redesign/p5_phase3_2stage/best_vs_tabpfn_summary.json |  |
| Current encoder alone (mean-only reference) | 0 | 0.2860 | — | — | — | — | (legacy training run; no per-fold CSV archived) | reference R² only (0.286) from an earlier 5-fold run; per-fold data unavailable |
| Ablation: no TabPFN residual | 5 | 0.2659 ± 0.0432 | 0.7622 ± 0.0378 | 0.9920 ± 0.0559 | 0.5300 ± 0.0425 | 0.4690 ± 0.0223 | outputs/redesign/p5_ablation_no_tabpfn/best_vs_tabpfn_summary.json |  |
