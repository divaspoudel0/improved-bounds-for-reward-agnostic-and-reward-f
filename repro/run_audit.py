#!/usr/bin/env python3
"""Standalone runner for the ICML 2026 paper #11370 numerical audits.

Self-contained (no relative imports) so it can run identically on a local
worktree and on a Hugging Face Job. Prints a structured JSON report to stdout
-- the single evidence channel for the logbook.

Usage:
    python3 run_audit.py [--scale small|full|large] [--out report.json]

Ridel & Cohen, "Improved Bounds for Reward-Agnostic and Reward-Free
Exploration", arXiv 2602.16363 (ICML 2026, OpenReview dmTyOwKAmZ).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "audit"))

from audit import claim1, claim3, claim4, claim25  # noqa: E402


def audit_claim1_scaled(scale):
    cfg = {
        "small": dict(S=4, A=3, H=4),
        "full": dict(S=6, A=4, H=5),
        "large": dict(S=8, A=5, H=6),
    }[scale]
    return claim1.audit_claim1(**cfg)


def audit_claim3_scaled(scale):
    cfg = {
        "small": dict(S=4, A=3, H=4, T=300, n_mdps=6),
        "full": dict(S=6, A=4, H=5, T=500, n_mdps=8),
        "large": dict(S=8, A=5, H=6, T=800, n_mdps=10),
    }[scale]
    return claim3.audit_claim3(**cfg)


def audit_claim4_scaled(scale):
    cfg = {
        "small": dict(S=4, A=3, H=4, n_mdps=8),
        "full": dict(S=6, A=4, H=5, n_mdps=10),
        "large": dict(S=8, A=5, H=6, n_mdps=12),
    }[scale]
    return claim4.audit_claim4(**cfg)


def audit_claim25_scaled(scale):
    cfg = {
        "small": dict(S=15, A=3, H=16, eps=0.1),
        "full": dict(S=31, A=3, H=32, eps=0.1),
        "large": dict(S=63, A=3, H=64, eps=0.1),
    }[scale]
    return claim25.audit_claims25(**cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=["small", "full", "large"], default="full")
    ap.add_argument("--out", default=None, help="optional path to write JSON report")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="claim keys to skip, e.g. claim4")
    args = ap.parse_args()

    t0 = time.time()
    report = {
        "paper": "Improved Bounds for Reward-Agnostic and Reward-Free Exploration",
        "arxiv": "2602.16363",
        "openreview": "dmTyOwKAmZ",
        "audit_type": "numerical audit (theory paper, no GPU needed)",
        "scale": args.scale,
        "claims": {},
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
    }
    if "claim4" not in args.skip:
        report["claims"]["claim4_logbarrier"] = audit_claim4_scaled(args.scale)
    if "claim3" not in args.skip:
        report["claims"]["claim3_reward_sequence"] = audit_claim3_scaled(args.scale)
    if "claim1" not in args.skip:
        report["claims"]["claim1_sample_complexity"] = audit_claim1_scaled(args.scale)
    if "claim25" not in args.skip:
        report["claims"]["claims25_lower_bound"] = audit_claim25_scaled(args.scale)
    report["wall_clock_s"] = round(time.time() - t0, 2)

    out = json.dumps(report, indent=2, default=str)
    print(out)
    if args.out:
        with open(args.out, "w") as f:
            f.write(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
