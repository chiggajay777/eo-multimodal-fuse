from __future__ import annotations

import argparse
import json

from eo_multimodal_fuse.train import run_demo


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="eo-multimodal-fuse")
    p.add_argument("--out", type=str, default="output/demo")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args(argv)
    print("eo-multimodal-fuse: starting demo...")
    results = run_demo(
        args.out,
        n=args.n,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps({"device": results["device"], "methods": results["methods"]}, indent=2))
    print(f"artifacts: {results['artifacts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
