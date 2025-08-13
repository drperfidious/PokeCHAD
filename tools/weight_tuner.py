"""Evolutionary weight tuner for StockfishModel evaluation function.

Usage:
  python tools/weight_tuner.py \
      --format gen9randombattle \
      --population 32 \
      --generations 50 \
      --seeds 32 \
      --out Models/weights.json

Strategy:
  (1+λ) style ES around current champion (loaded from --out or defaults).
  Each generation: sample λ mutation candidates, evaluate vs champion across paired seeds; keep best.
  If best beats champion by margin > epsilon (binomial lower CI), promote.
  Adaptive sigma: shrink on no-improvement, expand on improvement streak.

Logs:
  - JSONL per candidate in logs/weight_tuning.jsonl
  - Champion snapshots: Models/weights_champion_<timestamp>.json (when improved)
"""
from __future__ import annotations
import argparse, json, math, random, time
from pathlib import Path
from typing import Dict, List, Tuple

from Models.stockfish_model import StockfishModel  # type: ignore
from tools.weight_spaces import WEIGHT_BOUNDS, DEFAULT_MUTATION_SCALE, SEARCH_KEYS
from tools.self_play import run_self_play

# Reintroduce log directory constant (was removed during refactor)
from pathlib import Path as _P
# LOG_DIR relative to repo root, not CWD
_BASE_DIR = _P(__file__).resolve().parent.parent
LOG_DIR = _BASE_DIR / 'logs'
LOG_DIR.mkdir(exist_ok=True)


def load_champion(path: Path) -> Dict[str, float]:
    engine = StockfishModel()
    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                engine.set_weights(raw)
        except Exception:
            pass
    return dict(engine._W)  # type: ignore


def clip_weights(w: Dict[str, float]) -> Dict[str, float]:
    for k,(lo,hi) in WEIGHT_BOUNDS.items():
        if k in w:
            if w[k] < lo: w[k]=lo
            elif w[k] > hi: w[k]=hi
    return w


def mutate(base: Dict[str,float], sigma_scale: float, rng: random.Random) -> Dict[str,float]:
    child = dict(base)
    for k in SEARCH_KEYS:
        lo,hi = WEIGHT_BOUNDS[k]
        scale = DEFAULT_MUTATION_SCALE[k] * sigma_scale
        child[k] = child.get(k,0.0) + rng.gauss(0.0, scale)
        if child[k] < lo: child[k] = lo
        if child[k] > hi: child[k] = hi
    return child


def evaluate_candidate(format_id: str, seeds: List[int], champion: Dict[str,float], candidate: Dict[str,float], *, offline: bool) -> Tuple[float,int,int]:
    # Candidate as P1 vs champion as P2, then swap (paired seeds)
    results_forward = run_self_play(format_id, seeds, candidate, champion, offline=offline)
    results_reverse = run_self_play(format_id, seeds, champion, candidate, offline=offline)
    wins = 0; games = 0
    for r in results_forward:
        if r.p1_won is True: wins += 1
        elif r.p1_won is None: pass
        games += 1
    for r in results_reverse:
        if r.p1_won is False:  # champion as p1 lost => candidate won as p2
            wins += 1
        elif r.p1_won is None: pass
        games += 1
        # If repeated network errors, break early
        if r.error and 'ConnectionRefusedError' in (r.error or ''):
            break
    win_rate = wins / games if games else 0.0
    return win_rate, wins, games


def proportion_ci_lower(p: float, n: int, z: float = 1.96) -> float:
    if n == 0: return 0.0
    # Wilson lower bound
    denom = 1 + z*z/n
    centre = p + z*z/(2*n)
    adj = z*math.sqrt((p*(1-p) + z*z/(4*n))/n)
    lower = (centre - adj)/denom
    return lower


def main():
    ap = argparse.ArgumentParser(description='Evolutionary tuner for evaluation weights.')
    ap.add_argument('--format', default='gen9randombattle')
    ap.add_argument('--population', type=int, default=32)
    ap.add_argument('--generations', type=int, default=50)
    ap.add_argument('--seeds', type=int, default=32, help='Number of base seeds (each used twice: forward+reverse)')
    ap.add_argument('--out', default='Models/weights.json')
    ap.add_argument('--sigma', type=float, default=1.0, help='Initial sigma scale multiplier')
    ap.add_argument('--patience', type=int, default=10, help='Generations without improvement before stop')
    ap.add_argument('--min-promote-diff', type=float, default=0.02, help='Minimum raw win-rate advantage required to consider promotion')
    ap.add_argument('--confidence', type=float, default=0.95, help='Confidence for Wilson lower bound on improvement')
    ap.add_argument('--seed', type=int, default=0)
    # Default is offline self-play; use --online to force remote Showdown server
    ap.add_argument('--online', action='store_true', help='Use online Showdown server (default offline self-play).')
    # Backwards compatibility flag (ignored; offline is default)
    ap.add_argument('--offline', action='store_true', help=argparse.SUPPRESS)
    args = ap.parse_args()

    rng = random.Random(args.seed or int(time.time()))

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = _BASE_DIR / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    champion = load_champion(out_path)
    base_sigma = args.sigma
    sigma = base_sigma

    # Pre-generate seed list
    base_seeds = [rng.randint(1, 1_000_000_000) for _ in range(args.seeds)]

    log_path = LOG_DIR / 'weight_tuning.jsonl'

    def log_event(obj: Dict):
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(obj) + '\n')
        except Exception:
            pass

    gen = 0
    no_improve = 0
    champion_wr = 0.5  # baseline expectation; updated on promotions

    print(f"[init] Champion weights: {champion}")
    offline_flag = not args.online  # offline by default unless --online supplied
    log_event({'event':'init','champion':champion, 'offline': offline_flag, 'format': args.format, 'out': str(out_path), 'seeds': len(base_seeds)*2, 'sigma': sigma})

    while gen < args.generations and no_improve < args.patience:
        gen += 1
        gen_start_time = time.time()
        # Log generation start snapshot
        log_event({'event':'gen_start','generation':gen,'champion_wr':champion_wr,'champion':champion,'sigma':sigma,'offline':offline_flag})
        candidates: List[Tuple[Dict[str,float], float, int, int]] = []
        for i in range(args.population):
            cand = mutate(champion, sigma, rng)
            wr, wins, games = evaluate_candidate(args.format, base_seeds, champion, cand, offline=offline_flag)
            candidates.append((cand, wr, wins, games))
            log_event({'event':'candidate','generation':gen,'idx':i,'weights':cand,'win_rate':wr,'wins':wins,'games':games,'sigma':sigma,'champion_wr_at_gen_start':champion_wr,'offline':offline_flag})
            print(f"[gen {gen}] cand {i} wr={wr:.3f} wins={wins}/{games} sigma={sigma:.3f}")
        # Select best
        candidates.sort(key=lambda t: t[1], reverse=True)
        best, best_wr, best_wins, best_games = candidates[0]
        diff = best_wr - champion_wr
        promote = False
        if diff >= args.min_promote_diff:
            z = 1.96 if args.confidence >= 0.95 else 1.645
            lower = proportion_ci_lower(best_wr, best_games, z=z)
            if lower - champion_wr >= 0:
                promote = True
        gen_seconds = time.time() - gen_start_time
        if promote:
            timestamp = time.strftime('%Y%m%d-%H%M%S')
            backup = out_path.with_name(f"weights_champion_{timestamp}.json")
            try:
                with open(backup,'w',encoding='utf-8') as f: json.dump(best, f, indent=2)
            except Exception: pass
            try:
                with open(out_path,'w',encoding='utf-8') as f: json.dump(best, f, indent=2)
            except Exception: pass
            prev_champ = champion
            champion = best
            champion_wr = best_wr
            no_improve = 0
            sigma = min(sigma * 1.1, 3.0)
            log_event({
                'event':'promote',
                'generation':gen,
                'new_champion':champion,
                'prev_champion': prev_champ,
                'champion_wr':champion_wr,
                'backup':str(backup),
                'gen_seconds':gen_seconds,
                'sigma':sigma,
                'offline':offline_flag,
                'wins': int(best_wins),
                'games': int(best_games),
                'win_rate': float(best_wr),
                'diff': float(diff),
                'format': args.format,
                'seeds': len(base_seeds)*2,
            })
            print(f"[gen {gen}] PROMOTION wr={best_wr:.3f} diff={diff:.3f} sigma->{sigma:.3f} time={gen_seconds:.1f}s")
        else:
            no_improve += 1
            sigma = max(sigma * 0.9, 0.05)
            log_event({'event':'no_improve','generation':gen,'best_wr':best_wr,'diff':diff,'no_improve':no_improve,'sigma':sigma,'gen_seconds':gen_seconds,'offline':offline_flag})
            print(f"[gen {gen}] no improvement best_wr={best_wr:.3f} diff={diff:.3f} no_improve={no_improve} sigma->{sigma:.3f} time={gen_seconds:.1f}s")

    print(f"[done] generations={gen} final_champion_wr={champion_wr:.3f} weights={champion}")
    log_event({'event':'done','generations':gen,'final_champion':champion,'champion_wr':champion_wr,'offline':offline_flag,'sigma':sigma,'format':args.format})

if __name__ == '__main__':
    main()
