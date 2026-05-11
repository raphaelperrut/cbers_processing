from __future__ import annotations

import argparse
import torch

from cbers_colorize.tools.sr.rsinet_x4 import (
    build_rsinet_x4,
    load_rsinet_x4_checkpoint,
    count_parameters,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--strict", action="store_true", default=False)
    ap.add_argument("--h", type=int, default=128)
    ap.add_argument("--w", type=int, default=128)
    ap.add_argument("--end_mode", default="last", choices=["last", "mean", "sum"])
    ap.add_argument("--model_half", action="store_true", default=False)
    return ap.parse_args()


def main():
    args = parse_args()

    model = build_rsinet_x4()
    print("Params:", count_parameters(model))

    model, report = load_rsinet_x4_checkpoint(
        checkpoint_path=args.checkpoint,
        device=args.device,
        strict=args.strict,
        model_half=args.model_half,
    )

    print("missing_keys:", len(report.missing_keys))
    for k in report.missing_keys[:50]:
        print("  MISSING:", k)

    print("unexpected_keys:", len(report.unexpected_keys))
    for k in report.unexpected_keys[:50]:
        print("  UNEXPECTED:", k)

    x_dtype = torch.float16 if (args.model_half and str(args.device).startswith("cuda")) else torch.float32
    x = torch.randn(1, 3, args.h, args.w, device=args.device, dtype=x_dtype)

    with torch.inference_mode():
        y = model(x, end_mode=args.end_mode)

    print("Input :", tuple(x.shape), x.dtype)
    print("Output:", tuple(y.shape), y.dtype)


if __name__ == "__main__":
    main()