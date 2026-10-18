"""Sen1Floods11 hand-labeled Sentinel-1 / Sentinel-2 subset.

The full hand-labeled archive is public Cloud-Optimized GeoTIFFs. This loader
downloads a deterministic chip subset (a few hundred MB), cuts non-overlapping
patches, and caches numpy arrays under the data directory.

Labels are the analyst ``LabelHand`` layer only. Weak Sentinel-1 Otsu labels
are not used, so the SAR image is not copied into the target.
"""

from __future__ import annotations

import json
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

GCS_ROOT = "https://storage.googleapis.com/sen1floods11/v1.1"
SPLIT_NAMES = {
    "train": "splits/flood_handlabeled/flood_train_data.csv",
    "test": "splits/flood_handlabeled/flood_test_data.csv",
    "bolivia": "splits/flood_handlabeled/flood_bolivia_data.csv",
}
HAND = "data/flood_events/HandLabeled"
# Rasterio band indexes (1-based) for B2, B3, B4, B8, B11, B12.
S2_INDEXES = (2, 3, 4, 8, 12, 13)
S2_BAND_NAMES = ("B2", "B3", "B4", "B8", "B11", "B12")
S1_BAND_NAMES = ("VV", "VH")
S2_SCALE = 10000.0
S1_DB_MIN = -30.0
S1_DB_MAX = 5.0
MIN_VALID_FRACTION = 0.9
MAX_NONFINITE_FRACTION = 0.01
CACHE_VERSION = 1
USER_AGENT = "eo-multimodal-fuse/0.2"

OPTICAL_FEATURE_NAMES = (
    "B2_mean",
    "B3_mean",
    "B4_mean",
    "B8_mean",
    "B11_mean",
    "B12_mean",
    "NDWI_mean",
    "MNDWI_mean",
    "NDVI_mean",
    "B8_std",
    "B11_std",
)
SAR_FEATURE_NAMES = ("VV_mean", "VH_mean", "VH_minus_VV_mean", "VV_std", "VH_std")


@dataclass(frozen=True)
class Chip:
    event: str
    chip_id: int
    s1_name: str
    label_name: str

    @property
    def base(self) -> str:
        return self.s1_name[: -len("_S1Hand.tif")]

    @property
    def s2_name(self) -> str:
        return f"{self.base}_S2Hand.tif"


def systematic_indices(n: int, k: int) -> list[int]:
    """``k`` spread-out indexes in ``range(n)``.

    Uses evenly spaced positions and, if rounding collides, the nearest free
    index. Deterministic. Returns every index when ``k >= n``.
    """
    if k <= 0 or n <= 0:
        return []
    if k >= n:
        return list(range(n))
    if k == 1:
        raw = [n // 2]
    else:
        raw = [int(round(i * (n - 1) / (k - 1))) for i in range(k)]
    chosen: list[int] = []
    used: set[int] = set()
    for idx in raw:
        if idx not in used:
            chosen.append(idx)
            used.add(idx)
            continue
        for delta in range(1, n):
            found = None
            for cand in (idx - delta, idx + delta):
                if 0 <= cand < n and cand not in used:
                    found = cand
                    break
            if found is not None:
                chosen.append(found)
                used.add(found)
                break
    return sorted(chosen)


def parse_split(text: str) -> list[Chip]:
    chips: list[Chip] = []
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        s1_name, label_name = [part.strip() for part in line.split(",")]
        base = s1_name[: -len("_S1Hand.tif")]
        event, chip_s = base.rsplit("_", 1)
        chips.append(Chip(event=event, chip_id=int(chip_s), s1_name=s1_name, label_name=label_name))
    return chips


def select_chips(chips: list[Chip], per_event: int, limit_events: int | None = None) -> list[Chip]:
    grouped: dict[str, list[Chip]] = defaultdict(list)
    for chip in chips:
        grouped[chip.event].append(chip)
    events = sorted(grouped)
    if limit_events is not None:
        events = events[:limit_events]
    selected: list[Chip] = []
    for event in events:
        group = sorted(grouped[event], key=lambda chip: chip.chip_id)
        for index in systematic_indices(len(group), per_event):
            selected.append(group[index])
    return selected


def normalize_optical(dn: np.ndarray) -> np.ndarray:
    """TOA digital numbers to reflectance clipped to [0, 1]."""
    return np.clip(np.asarray(dn, dtype=np.float32) / np.float32(S2_SCALE), 0.0, 1.0)


def normalize_sar(db: np.ndarray) -> np.ndarray:
    """Sentinel-1 dB clipped to [S1_DB_MIN, S1_DB_MAX] and scaled to [0, 1]."""
    span = np.float32(S1_DB_MAX - S1_DB_MIN)
    scaled = (np.asarray(db, dtype=np.float32) - np.float32(S1_DB_MIN)) / span
    return np.clip(scaled, 0.0, 1.0)


def _ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b) / np.maximum(a + b, np.float32(1e-6))


def optical_features(refl: np.ndarray) -> np.ndarray:
    """Classical water/vegetation features from reflectance (6, H, W)."""
    blue, green, red, nir, swir1, swir2 = refl
    ndwi = _ratio(green, nir)
    mndwi = _ratio(green, swir1)
    ndvi = _ratio(nir, red)
    return np.asarray(
        [
            blue.mean(),
            green.mean(),
            red.mean(),
            nir.mean(),
            swir1.mean(),
            swir2.mean(),
            ndwi.mean(),
            mndwi.mean(),
            ndvi.mean(),
            nir.std(),
            swir1.std(),
        ],
        dtype=np.float32,
    )


def sar_features(db: np.ndarray) -> np.ndarray:
    """Backscatter summaries from dB (2, H, W), VV then VH."""
    vv, vh = db
    return np.asarray(
        [vv.mean(), vh.mean(), (vh - vv).mean(), vv.std(), vh.std()],
        dtype=np.float32,
    )


def patch_label(lab: np.ndarray, water_fraction: float, min_valid: float = MIN_VALID_FRACTION) -> int | None:
    """Binary water presence, or None when too many pixels are nodata.

    Valid labels are 0 (not water) and 1 (water). -1 is nodata. A patch is
    positive when the water fraction among valid pixels is at least
    ``water_fraction``.
    """
    valid = (lab == 0) | (lab == 1)
    if float(valid.mean()) < min_valid:
        return None
    fraction = float((lab[valid] == 1).mean())
    return int(fraction >= water_fraction)


def _fill_nonfinite(arr: np.ndarray) -> np.ndarray:
    out = np.array(arr, copy=True)
    for band_i in range(out.shape[0]):
        band = out[band_i]
        bad = ~np.isfinite(band)
        if not bad.any():
            continue
        finite = band[np.isfinite(band)]
        fill = float(np.median(finite)) if finite.size else 0.0
        band[bad] = fill
        out[band_i] = band
    return out


def iter_patches(
    optical_dn: np.ndarray,
    sar_db: np.ndarray,
    label: np.ndarray,
    *,
    patch_size: int,
    water_fraction: float,
):
    """Yield normalized patches and index features from one chip."""
    if optical_dn.shape[1:] != label.shape or sar_db.shape[1:] != label.shape:
        raise ValueError("optical, SAR, and label chip shapes do not match")
    height, width = label.shape
    if height % patch_size or width % patch_size:
        raise ValueError(f"patch size {patch_size} does not tile {height}x{width}")
    for y0 in range(0, height, patch_size):
        for x0 in range(0, width, patch_size):
            lab = label[y0 : y0 + patch_size, x0 : x0 + patch_size]
            y = patch_label(lab, water_fraction)
            if y is None:
                continue
            opt = optical_dn[:, y0 : y0 + patch_size, x0 : x0 + patch_size]
            sar = sar_db[:, y0 : y0 + patch_size, x0 : x0 + patch_size]
            if (
                float((~np.isfinite(opt)).mean()) > MAX_NONFINITE_FRACTION
                or float((~np.isfinite(sar)).mean()) > MAX_NONFINITE_FRACTION
            ):
                continue
            opt = _fill_nonfinite(opt)
            sar = _fill_nonfinite(sar)
            refl = opt / np.float32(S2_SCALE)
            yield (
                normalize_optical(opt),
                normalize_sar(sar),
                optical_features(refl),
                sar_features(sar),
                y,
            )


def _request(url: str, method: str = "GET") -> urllib.request.Request:
    return urllib.request.Request(url, method=method, headers={"User-Agent": USER_AGENT})


def _remote_size(url: str) -> int | None:
    try:
        with urllib.request.urlopen(_request(url, "HEAD"), timeout=60) as resp:
            length = resp.headers.get("Content-Length")
            return int(length) if length is not None else None
    except Exception:
        return None


def download(url: str, dest: Path, retries: int = 4) -> int:
    """Download ``url`` to ``dest`` if missing or the wrong size. Return bytes on disk."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        expected = _remote_size(url)
        if expected is None or dest.stat().st_size == expected:
            return dest.stat().st_size
    tmp = dest.with_suffix(dest.suffix + ".partial")
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            if tmp.exists():
                tmp.unlink()
            with urllib.request.urlopen(_request(url), timeout=180) as resp, tmp.open("wb") as handle:
                total = 0
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    handle.write(chunk)
                    total += len(chunk)
                length = resp.headers.get("Content-Length")
                if length is not None and total != int(length):
                    raise OSError(f"short read {total} != {length}")
            tmp.replace(dest)
            return dest.stat().st_size
        except Exception as exc:  # noqa: BLE001 — retry network errors
            last_error = exc
            if tmp.exists():
                tmp.unlink()
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to download {url}: {last_error}")


def _read_text(url: str) -> str:
    with urllib.request.urlopen(_request(url), timeout=60) as resp:
        return resp.read().decode("utf-8")


def _chip_paths(root: Path, chip: Chip) -> dict[str, Path]:
    base = root / "chips"
    return {
        "s1": base / "S1Hand" / chip.s1_name,
        "s2": base / "S2Hand" / chip.s2_name,
        "label": base / "LabelHand" / chip.label_name,
    }


def _chip_urls(chip: Chip) -> dict[str, str]:
    return {
        "s1": f"{GCS_ROOT}/{HAND}/S1Hand/{chip.s1_name}",
        "s2": f"{GCS_ROOT}/{HAND}/S2Hand/{chip.s2_name}",
        "label": f"{GCS_ROOT}/{HAND}/LabelHand/{chip.label_name}",
    }


def _download_chips(root: Path, chips: list[Chip], workers: int = 8) -> int:
    jobs = []
    for chip in chips:
        paths = _chip_paths(root, chip)
        urls = _chip_urls(chip)
        for key in ("s1", "s2", "label"):
            jobs.append((urls[key], paths[key]))
    total = 0
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(download, url, path) for url, path in jobs]
        for future in as_completed(futures):
            total += future.result()
            done += 1
            if done % 15 == 0 or done == len(futures):
                print(f"  files {done}/{len(futures)}  {total / 1e6:.1f} MB", flush=True)
    return total


def _read_chip(paths: dict[str, Path]):
    import rasterio

    with rasterio.open(paths["s1"]) as src:
        sar = src.read().astype(np.float32)
    with rasterio.open(paths["s2"]) as src:
        optical = src.read(indexes=S2_INDEXES).astype(np.float32)
    with rasterio.open(paths["label"]) as src:
        label = src.read(1)
    if sar.shape[0] != 2 or optical.shape[0] != 6:
        raise RuntimeError(f"unexpected band count in {paths['s1'].name}")
    return optical, sar, label


def _stack(rows: list[tuple], channels: int, patch: int, dtype) -> np.ndarray:
    if not rows:
        return np.zeros((0, channels, patch, patch), dtype=dtype)
    return np.stack(rows, axis=0).astype(dtype, copy=False)


def _extract_split(root: Path, chips: list[Chip], patch_size: int, water_fraction: float) -> dict:
    opt_imgs, sar_imgs, opt_feats, sar_feats, labels = [], [], [], [], []
    kept_per_chip = []
    for chip in chips:
        optical, sar, label = _read_chip(_chip_paths(root, chip))
        n_before = len(labels)
        for opt_i, sar_i, opt_f, sar_f, y in iter_patches(
            optical, sar, label, patch_size=patch_size, water_fraction=water_fraction
        ):
            opt_imgs.append(opt_i.astype(np.float16))
            sar_imgs.append(sar_i.astype(np.float16))
            opt_feats.append(opt_f)
            sar_feats.append(sar_f)
            labels.append(y)
        kept_per_chip.append({"base": chip.base, "n_patches": len(labels) - n_before})
    patch = patch_size
    return {
        "optical": _stack(opt_imgs, 6, patch, np.float16),
        "sar": _stack(sar_imgs, 2, patch, np.float16),
        "optical_features": np.stack(opt_feats, axis=0) if opt_feats else np.zeros((0, 11), np.float32),
        "sar_features": np.stack(sar_feats, axis=0) if sar_feats else np.zeros((0, 5), np.float32),
        "y": np.asarray(labels, dtype=np.int64),
        "chips": [asdict(chip) for chip in chips],
        "kept_per_chip": kept_per_chip,
    }


def _protocol(train, test, bolivia, patch_size, water_fraction, limit_events) -> dict:
    return {
        "cache_version": CACHE_VERSION,
        "patch_size": patch_size,
        "water_fraction": water_fraction,
        "min_valid_fraction": MIN_VALID_FRACTION,
        "limit_events": limit_events,
        "train": [chip.base for chip in train],
        "test": [chip.base for chip in test],
        "bolivia": [chip.base for chip in bolivia],
        "s2_bands": list(S2_BAND_NAMES),
        "s1_bands": list(S1_BAND_NAMES),
        "s1_db_clip": [S1_DB_MIN, S1_DB_MAX],
    }


def prepare_dataset(
    root: str | Path,
    *,
    train_chips_per_event: int = 4,
    test_chips_per_event: int = 2,
    patch_size: int = 64,
    water_fraction: float = 0.05,
    include_bolivia: bool = True,
    limit_events: int | None = None,
) -> dict:
    """Download the subset if needed and return train/test/(bolivia) arrays."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    print("Sen1Floods11: resolving official hand-labeled splits...", flush=True)
    train_all = parse_split(_read_text(f"{GCS_ROOT}/{SPLIT_NAMES['train']}"))
    test_all = parse_split(_read_text(f"{GCS_ROOT}/{SPLIT_NAMES['test']}"))
    bolivia_all = parse_split(_read_text(f"{GCS_ROOT}/{SPLIT_NAMES['bolivia']}"))
    train = select_chips(train_all, train_chips_per_event, limit_events)
    test = select_chips(test_all, test_chips_per_event, limit_events)
    bolivia = sorted(bolivia_all, key=lambda chip: chip.chip_id) if include_bolivia else []
    train_ids = {chip.base for chip in train}
    test_ids = {chip.base for chip in test}
    if train_ids & test_ids:
        raise RuntimeError(f"train/test chip overlap: {sorted(train_ids & test_ids)}")
    if train_ids & {chip.base for chip in bolivia}:
        raise RuntimeError("Bolivia chips overlapped the training subset")

    protocol = _protocol(train, test, bolivia, patch_size, water_fraction, limit_events)
    manifest_path = root / "subset_manifest.json"
    npz_path = root / "patches.npz"
    if manifest_path.exists() and npz_path.exists():
        cached = json.loads(manifest_path.read_text(encoding="utf-8"))
        if cached.get("protocol") == protocol:
            print(f"Sen1Floods11: using cached patches {npz_path}", flush=True)
            return _load_cache(root, cached)

    chips = train + test + bolivia
    print(f"Sen1Floods11: downloading {len(chips)} chips (train {len(train)}, test {len(test)}, bolivia {len(bolivia)})", flush=True)
    download_bytes = _download_chips(root, chips)
    print("Sen1Floods11: extracting patches...", flush=True)
    splits = {
        "train": _extract_split(root, train, patch_size, water_fraction),
        "test": _extract_split(root, test, patch_size, water_fraction),
    }
    if bolivia:
        splits["bolivia"] = _extract_split(root, bolivia, patch_size, water_fraction)
    payload = {"protocol": protocol, "download_bytes": download_bytes, "splits": {}}
    arrays = {}
    for name, split in splits.items():
        if split["y"].size == 0:
            raise RuntimeError(f"{name} split produced zero patches; check filters")
        arrays[f"{name}_optical"] = split["optical"]
        arrays[f"{name}_sar"] = split["sar"]
        arrays[f"{name}_optical_features"] = split["optical_features"]
        arrays[f"{name}_sar_features"] = split["sar_features"]
        arrays[f"{name}_y"] = split["y"]
        payload["splits"][name] = {
            "n_patches": int(split["y"].size),
            "positive_rate": float(split["y"].mean()),
            "n_chips": len(split["chips"]),
            "chips": split["chips"],
            "kept_per_chip": split["kept_per_chip"],
        }
        print(
            f"  {name}: {split['y'].size} patches, "
            f"positive rate {float(split['y'].mean()):.3f}, chips {len(split['chips'])}",
            flush=True,
        )
    np.savez(npz_path, **arrays)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    payload["root"] = str(root)
    payload["arrays"] = arrays
    return payload


def _load_cache(root: Path, manifest: dict) -> dict:
    with np.load(root / "patches.npz") as data:
        arrays = {key: np.array(data[key]) for key in data.files}
    manifest = dict(manifest)
    manifest["root"] = str(root)
    manifest["arrays"] = arrays
    # Cached manifests store download_bytes from the original download.
    if "download_bytes" not in manifest:
        manifest["download_bytes"] = _directory_bytes(root / "chips")
    return manifest


def _directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
