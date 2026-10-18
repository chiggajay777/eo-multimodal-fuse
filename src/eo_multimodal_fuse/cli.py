"""Command line entry point.

The default command downloads the Sen1Floods11 subset and runs the multi-seed
benchmark. ``--demo`` stays on synthetic features and does not download data.
"""

from __future__ import annotations

import argparse
import sys

from eo_multimodal_fuse.metrics import format_mean_std


def _print_summary(payload: dict) -> None:
    print(f"device: {payload.get('hardware', {}).get('device', payload.get('device'))}")
    if "runtime_seconds" in payload:
        print(f"runtime_seconds: {payload['runtime_seconds']:.1f}")
    for row in payload["summary"]:
        if row["condition"] not in ("full", "optical_missing", "sar_missing"):
            continue
        print(
            f"{row['split']:10} {row['method']:32} {row['condition']:16} "
            f"acc {format_mean_std(row['accuracy_mean'], row['accuracy_std'])}  "
            f"f1 {format_mean_std(row['macro_f1_mean'], row['macro_f1_std'])}  "
            f"auroc {format_mean_std(row['auroc_mean'], row['auroc_std'])}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eo-multimodal-fuse",
        description=(
            "Optical + SAR fusion vs unimodal baselines. "
            "Default: Sen1Floods11 hand-labeled subset. "
            "--demo: synthetic offline fallback."
        ),
    )
    parser.add_argument("--demo", action="store_true", help="synthetic data, no download")
    parser.add_argument("--out", type=str, default=None, help="output directory")
    parser.add_argument("--device", type=str, default=None, help="torch device (default: cpu)")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--modality-dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42, help="single seed used by --demo")
    parser.add_argument("--n", type=int, default=2000, help="synthetic sample count for --demo")
    parser.add_argument("--data-dir", type=str, default="data/sen1floods11")
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
        help="comma-separated training seeds",
    )
    parser.add_argument("--train-chips-per-event", type=int, default=4)
    parser.add_argument("--test-chips-per-event", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--water-fraction", type=float, default=0.05)
    parser.add_argument("--no-bolivia", action="store_true", help="skip the Bolivia hold-out")
    parser.add_argument(
        "--limit-events",
        type=int,
        default=None,
        help="debug only: keep the first N events alphabetically",
    )
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--publish-figures",
        type=str,
        default=None,
        help="also copy png figures to this directory",
    )
    args = parser.parse_args(argv)

    if args.demo:
        from eo_multimodal_fuse.train import run_demo

        out = args.out or "output/demo"
        print("eo-multimodal-fuse: synthetic demo (not the reported benchmark)")
        payload = run_demo(
            out,
            n=args.n,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device or "cpu",
            modality_dropout=args.modality_dropout,
            lr=args.lr,
        )
    else:
        from eo_multimodal_fuse.benchmark import run_benchmark

        seeds = [int(part) for part in args.seeds.split(",") if part.strip() != ""]
        out = args.out or "results"
        payload = run_benchmark(
            out,
            args.data_dir,
            seeds=seeds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            train_chips_per_event=args.train_chips_per_event,
            test_chips_per_event=args.test_chips_per_event,
            modality_dropout=args.modality_dropout,
            patch_size=args.patch_size,
            water_fraction=args.water_fraction,
            include_bolivia=not args.no_bolivia,
            limit_events=args.limit_events,
            device=args.device or "cpu",
            num_threads=args.threads,
            publish_figures=args.publish_figures,
            argv=sys.argv[1:] if argv is None else argv,
        )
    _print_summary(payload)
    print(f"artifacts: {payload['artifacts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
