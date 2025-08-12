#!/usr/bin/env python3
# tools/train_launcher.py
from __future__ import annotations
import argparse, glob, json, os, shutil, subprocess, sys, time
from pathlib import Path

def backup_if_exists(p: Path):
    if p.exists():
        ts = time.strftime("%Y%m%d-%H%M%S")
        bak = p.with_suffix(p.suffix + f".bak.{ts}")
        shutil.copy2(p, bak)
        print(f"[backup] {p} -> {bak}")

def main():
    ap = argparse.ArgumentParser(description="Train Stockfish weights from telemetry logs with auto-backup.")
    ap.add_argument("--glob", default="logs/telemetry_*.jsonl")
    ap.add_argument("--out", default="Models/weights.json")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    logs = sorted(glob.glob(args.glob))
    if not logs:
        print(f"[error] no logs match {args.glob}")
        sys.exit(2)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    backup_if_exists(out)

    cmd = [args.python, "../tools/train_weights.py", *logs, "--out", str(out)]
    print("[run]", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        sys.exit(proc.returncode)
    print("[ok] weights saved to", out)

if __name__ == "__main__":
    main()
