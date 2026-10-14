# eo-multimodal-fuse

**Multimodal EO fusion toy experiment** — optical + SAR-like proxies, unimodal baselines vs late / cross-attention fusion.

## Problem

Hazard / change screening often has **missing modalities** (clouds kill optical; SAR is noisy). A portfolio-ready fusion demo should (1) beat strong unimodal baselines and (2) show what happens when one modality is dropped.

## Method

| Modality | Features (synthetic) |
|----------|----------------------|
| Optical-like | NDVI proxy, brightness, green/blue ratios |
| SAR-like | VV/VH mean proxies, VV std, texture |

Models: `optical_only` · `sar_only` · `late_fusion` · `cross_attention_fusion` · robustness check `late_fusion_sar_missing`.

## How to run

```bash
cd /workspace/phd-github-portfolio/eo-multimodal-fuse
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

python -m eo_multimodal_fuse --out output/demo
```

Outputs: `metrics.json`, `metrics_table.csv`, `figures/fusion_accuracy.png`.

## Limitations

- Synthetic features (not real Sentinel-1/2 stacks).
- Small MLPs — pedagogical, not SOTA.
- Next upgrade: real tiny open chips (e.g. SEN12MS subset) + modality dropout schedule.

## License

MIT
