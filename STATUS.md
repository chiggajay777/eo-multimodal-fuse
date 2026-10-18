# STATUS — eo-multimodal-fuse v0.2.0

**Runnable:** yes. Default command downloads a 294 MB Sen1Floods11 hand-labeled subset (40 train / 20 in-domain test / 15 Bolivia chips) and trains unimodal CNNs, index logistic regressions, late fusion, and cross-attention over seeds 0–4.

**Reported run:** CPU, 4 threads, PyTorch 2.14.1+cpu, 308 seconds including download. Metrics are in `results/`. Figures are in `figures/`.

**Result, briefly:** on the in-domain chip hold-out, fusion does not beat an optical CNN or an optical-index logistic regression (macro-F1 about 0.92, gaps inside the seed noise). Without modality dropout, zeroing a sensor collapses fusion hard-predictions to one class (accuracy falls to the positive rate) even though AUROC stays high. With modality dropout that collapse goes away. On the Bolivia hold-out, a SAR-index logistic regression is as strong as the best fusion net. No metric in this run is NaN.

**Offline fallback:** `python -m eo_multimodal_fuse --demo` (synthetic features, not the numbers above).

**Not claimed:** full Sen1Floods11 segmentation scores, So2Sat LCZ42, or a tuned model.
