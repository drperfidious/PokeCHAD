#!/usr/bin/env python3
# tools/fetch_ps_data.py
from __future__ import annotations

import argparse, hashlib, os, sys, time
from pathlib import Path
from urllib.request import urlopen

BASE = "https://play.pokemonshowdown.com/data/"

DEFAULT_FILES = [
    "pokedex.json",
    "moves.json",
    "learnsets.json",
    "items.js",
    "abilities.js",
    "formats.js",
    "formats-data.js",
    "typechart.js",
]

def _read(url: str) -> bytes:
    with urlopen(url, timeout=30) as r:
        return r.read()

def _md5(b: bytes) -> str:
    import hashlib
    return hashlib.md5(b).hexdigest()

def backup_if_changed(dst: Path, new_bytes: bytes) -> None:
    if dst.exists():
        old = dst.read_bytes()
        if _md5(old) != _md5(new_bytes):
            ts = time.strftime("%Y%m%d-%H%M%S")
            bak = dst.with_suffix(dst.suffix + f".bak.{ts}")
            dst.rename(bak)
            print(f"[backup] {dst} -> {bak.name}")

def download(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    data = _read(url)
    backup_if_changed(out, data)
    out.write_bytes(data)
    print(f"[ok] {out.relative_to(Path.cwd())} ({len(data)} bytes)")

def main():
    p = argparse.ArgumentParser(description="Fetch latest Pokémon Showdown data files.")
    p.add_argument("--out", default="Data/showdown", help="output directory (default: Data/showdown)")
    p.add_argument("--gen", default="9", help="generation for /data/sets/<gen>.json (default: 9)")
    p.add_argument("--extra-sets", action="store_true", help="also fetch /data/sets/gen<gen>.json")
    args = p.parse_args()

    outdir = Path(args.out).resolve()
    files = list(DEFAULT_FILES)
    if args.extra_sets:
        files.append(f"sets/gen{args.gen}.json")

    print(f"Fetching {len(files)} file(s) into {outdir} …")
    for rel in files:
        url = BASE + rel
        dst = outdir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            download(url, dst)
        except Exception as e:
            print(f"[warn] failed to fetch {url}: {e}")

    print("Done.")

if __name__ == "__main__":
    main()
