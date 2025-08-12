
# tools/train_weights.py
"""
Train scoring weights from telemetry logs (JSONL from the UI).

Usage:
    python tools/train_weights.py logs/telemetry_*.jsonl --out Models/weights.json

We compute a simple linear regression from per-turn features to next-turn HP delta reward.
"""
from __future__ import annotations
import os, sys, json, glob
from collections import defaultdict
from typing import Dict, List, Any
import math

try:
    import numpy as np
except Exception as e:
    print("NumPy required: pip install numpy")
    raise

def _features(entry: Dict[str, Any]) -> List[float]:
    """Extract features for the PICKED candidate."""
    p = entry.get("picked", {}) or {}
    order = entry.get("order", {}) or {}
    return [
        float(p.get("expected", 0.0)),
        float(p.get("acc_mult", 1.0)),
        float(p.get("effectiveness", 1.0)),
        float(order.get("p_user_first", 0.5)),
        float(p.get("opp_counter_ev", 0.0)),
    ]

def _reward(prev: Dict[str, Any], nxt: Dict[str, Any]) -> float:
    """Reward = (opponent total HP loss) - (our total HP loss) between snapshots."""
    def tot(team: Dict[str, Any]) -> float:
        return sum(float(p.get("hp_fraction") or 0.0) for p in team.values()) if team else 0.0
    p_my = prev.get("snapshot", {}).get("my_team", {})
    p_opp = prev.get("snapshot", {}).get("opp_team", {})
    n_my = nxt.get("snapshot", {}).get("my_team", {})
    n_opp = nxt.get("snapshot", {}).get("opp_team", {})
    return (tot(p_opp) - tot(n_opp)) - (tot(p_my) - tot(n_my))

def main():
    argv = sys.argv[1:]
    if not argv:
        print(__doc__)
        return
    paths = []
    out = "Models/weights.json"
    i = 0
    while i < len(argv):
        if argv[i] == "--out":
            out = argv[i+1]; i += 2
        else:
            paths.extend(glob.glob(argv[i])); i += 1

    # Load logs
    entries: List[Dict[str, Any]] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass
    if not entries:
        print("No telemetry entries found.")
        return

    # Group by battle and turn to pair rewards
    entries.sort(key=lambda e: (e.get("battle_tag",""), int(e.get("turn") or 0)))
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in entries:
        grouped[str(e.get("battle_tag",""))].append(e)

    X = []
    y = []
    for tag, arr in grouped.items():
        for i in range(len(arr)-1):
            cur, nxt = arr[i], arr[i+1]
            X.append(_features(cur))
            y.append(_reward(cur, nxt))

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = X.shape[0]
    if n < 10:
        print(f"Warning: very few samples ({n}). Results may be noisy.")

    # Add bias term
    Xb = np.concatenate([X, np.ones((n,1))], axis=1)
    # Ridge regularization
    lam = 1e-3
    XtX = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    Xty = Xb.T @ y
    w = np.linalg.solve(XtX, Xty)

    # Map to our model's weight names best-effort
    names = ["expected", "acc_mult", "effectiveness", "p_user_first", "opp_counter_ev", "bias"]
    weights = {k: float(v) for k, v in zip(names, w)}
    print("Fitted raw weights:", weights)

    # Translate to our runtime weights roughly
    # We scale/assign:
    out_weights = {
        "go_first_bonus": float(weights["p_user_first"]),
        "opp_dmg_penalty": max(0.0, float(-weights["opp_counter_ev"])),
        "survival_bonus": max(0.0, float(weights["bias"])),
        # Keep switch weights unchanged for now
    }

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(out_weights, f, indent=2)
    print(f"Saved weights to {out}")

if __name__ == "__main__":
    main()
