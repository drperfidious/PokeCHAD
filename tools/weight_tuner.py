"""Evolutionary weight tuner for StockfishModel evaluation function.

Usage:
  python tools/weight_tuner.py \
      --format gen9randombattle \
      --population 32 \
      --generations 50 \
      --seeds 128 \
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


def calculate_confidence_intervals(p: float, n: int, confidence: float = 0.95) -> Dict[str, float]:
    """Phase 3: Calculate comprehensive confidence intervals for training metrics"""
    if n == 0:
        return {'lower': 0.0, 'upper': 0.0, 'margin_of_error': 0.0, 'width': 0.0}
    
    # Determine z-score based on confidence level
    z_scores = {
        0.90: 1.645,
        0.95: 1.96,
        0.99: 2.576
    }
    z = z_scores.get(confidence, 1.96)
    
    # Wilson confidence interval (more accurate for proportions)
    denom = 1 + z*z/n
    centre = p + z*z/(2*n)
    adj = z*math.sqrt((p*(1-p) + z*z/(4*n))/n)
    wilson_lower = (centre - adj)/denom
    wilson_upper = (centre + adj)/denom
    
    # Ensure bounds are valid
    wilson_lower = max(0.0, wilson_lower)
    wilson_upper = min(1.0, wilson_upper)
    
    margin_of_error = (wilson_upper - wilson_lower) / 2
    interval_width = wilson_upper - wilson_lower
    
    return {
        'lower': wilson_lower,
        'upper': wilson_upper,
        'margin_of_error': margin_of_error,
        'width': interval_width,
        'confidence_level': confidence,
        'z_score': z,
        'sample_size': n
    }


def analyze_training_metrics_with_ci(candidates: List[Tuple[Dict[str,float], float, int, int]], 
                                   confidence: float = 0.95) -> Dict[str, Any]:
    """Phase 3: Analyze training metrics with confidence intervals"""
    if not candidates:
        return {'error': 'No candidates to analyze'}
    
    win_rates = [wr for _, wr, _, _ in candidates]
    game_counts = [games for _, _, _, games in candidates]
    
    import statistics
    
    # Basic statistics
    mean_wr = statistics.mean(win_rates)
    median_wr = statistics.median(win_rates)
    std_wr = statistics.stdev(win_rates) if len(win_rates) > 1 else 0.0
    min_wr = min(win_rates)
    max_wr = max(win_rates)
    
    # Calculate confidence intervals for each candidate
    candidate_cis = []
    for i, (_, wr, wins, games) in enumerate(candidates):
        ci = calculate_confidence_intervals(wr, games, confidence)
        ci['candidate_index'] = i
        ci['win_rate'] = wr
        candidate_cis.append(ci)
    
    # Calculate overall confidence interval (treating all games as pooled)
    total_wins = sum(wins for _, _, wins, _ in candidates)
    total_games = sum(games for _, _, _, games in candidates)
    overall_ci = calculate_confidence_intervals(total_wins / max(total_games, 1), total_games, confidence)
    
    # Identify candidates with non-overlapping confidence intervals (potentially different performance)
    significant_differences = []
    for i in range(len(candidate_cis)):
        for j in range(i + 1, len(candidate_cis)):
            ci_i = candidate_cis[i]
            ci_j = candidate_cis[j]
            
            # Check if confidence intervals don't overlap
            no_overlap = (ci_i['upper'] < ci_j['lower']) or (ci_j['upper'] < ci_i['lower'])
            if no_overlap:
                significant_differences.append({
                    'candidate_i': i,
                    'candidate_j': j,
                    'win_rate_i': ci_i['win_rate'],
                    'win_rate_j': ci_j['win_rate'],
                    'ci_i': [ci_i['lower'], ci_i['upper']],
                    'ci_j': [ci_j['lower'], ci_j['upper']],
                    'difference': abs(ci_i['win_rate'] - ci_j['win_rate'])
                })
    
    analysis = {
        'basic_stats': {
            'mean': mean_wr,
            'median': median_wr,
            'std': std_wr,
            'min': min_wr,
            'max': max_wr,
            'count': len(candidates)
        },
        'overall_confidence_interval': overall_ci,
        'candidate_confidence_intervals': candidate_cis,
        'significant_differences': significant_differences,
        'confidence_level': confidence,
        'notes': {
            'total_games': total_games,
            'avg_games_per_candidate': statistics.mean(game_counts),
            'min_games': min(game_counts),
            'max_games': max(game_counts)
        }
    }
    
    return analysis


def validate_tuning_parameters(args) -> List[str]:
    """Phase 1: Validate weight tuning parameters before starting runs"""
    errors = []
    
    # Validate population size
    if args.population < 1:
        errors.append(f"Population size must be at least 1, got {args.population}")
    elif args.population > 200:
        errors.append(f"Population size is unusually large ({args.population}), max recommended is 200")
    
    # Validate number of generations
    if args.generations < 1:
        errors.append(f"Generations must be at least 1, got {args.generations}")
    elif args.generations > 1000:
        errors.append(f"Generations is unusually large ({args.generations}), max recommended is 1000")
    
    # Validate number of seeds
    if args.seeds < 2:
        errors.append(f"Number of seeds must be at least 2 for meaningful evaluation, got {args.seeds}")
    elif args.seeds > 1000:
        errors.append(f"Number of seeds is unusually large ({args.seeds}), max recommended is 1000")
    
    # Validate sigma
    if args.sigma <= 0:
        errors.append(f"Sigma must be positive, got {args.sigma}")
    elif args.sigma > 10:
        errors.append(f"Sigma is unusually large ({args.sigma}), max recommended is 10")
    
    # Validate confidence
    if not (0.5 <= args.confidence <= 0.999):
        errors.append(f"Confidence must be between 0.5 and 0.999, got {args.confidence}")
    
    # Validate min-promote-diff
    if not (0.0 <= args.min_promote_diff <= 0.5):
        errors.append(f"Min-promote-diff must be between 0.0 and 0.5, got {args.min_promote_diff}")
    
    # Validate patience
    if args.patience < 1:
        errors.append(f"Patience must be at least 1, got {args.patience}")
    elif args.patience > args.generations:
        errors.append(f"Patience ({args.patience}) should not exceed generations ({args.generations})")
    
    # Validate output path
    try:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = _BASE_DIR / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Test write access
        test_file = out_path.with_suffix('.test')
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            test_file.unlink()
        except Exception as e:
            errors.append(f"Cannot write to output path {out_path}: {e}")
    except Exception as e:
        errors.append(f"Invalid output path {args.out}: {e}")
    
    # Validate format
    valid_formats = ['gen9randombattle', 'gen9ou', 'gen8randombattle', 'gen8ou']
    if args.format not in valid_formats:
        errors.append(f"Unsupported format {args.format}, valid options: {valid_formats}")
    
    return errors


def validate_candidate_result(wr: float, wins: int, games: int) -> bool:
    """Phase 1: Validate candidate evaluation results for corruption detection"""
    try:
        # Check for impossible win rates
        if not (0.0 <= wr <= 1.0):
            return False
        
        # Check for impossible win/game counts
        if wins < 0 or games < 0 or wins > games:
            return False
        
        # Check for consistency between win rate and win/game counts
        if games > 0:
            expected_wr = wins / games
            if abs(wr - expected_wr) > 1e-6:  # Allow for floating point precision
                return False
        elif games == 0 and wr != 0.0:
            return False
        
        # Check for suspiciously extreme results that might indicate corruption
        if games > 10 and (wr == 0.0 or wr == 1.0):
            # Perfect scores with sufficient sample size are suspicious
            return False
        
        return True
    except Exception:
        return False


def detect_outliers_in_generation(candidates: List[Tuple[Dict[str,float], float, int, int]]) -> List[int]:
    """Phase 3: Detect outliers in candidate win rates using statistical methods"""
    if len(candidates) < 3:
        return []  # Need at least 3 candidates for meaningful outlier detection
    
    win_rates = [wr for _, wr, _, _ in candidates]
    
    # Calculate IQR (Interquartile Range) method
    sorted_rates = sorted(win_rates)
    n = len(sorted_rates)
    q1_idx = int(n * 0.25)
    q3_idx = int(n * 0.75)
    
    q1 = sorted_rates[q1_idx]
    q3 = sorted_rates[q3_idx]
    iqr = q3 - q1
    
    # Calculate outlier boundaries (1.5 * IQR method)
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Also use z-score method for additional validation
    import statistics
    try:
        mean_wr = statistics.mean(win_rates)
        stdev_wr = statistics.stdev(win_rates) if len(win_rates) > 1 else 0
    except Exception:
        return []
    
    outlier_indices = []
    for i, (_, wr, wins, games) in enumerate(candidates):
        is_outlier = False
        
        # IQR-based outlier detection
        if wr < lower_bound or wr > upper_bound:
            is_outlier = True
        
        # Z-score based outlier detection (z > 2.5 is considered outlier)
        if stdev_wr > 0:
            z_score = abs(wr - mean_wr) / stdev_wr
            if z_score > 2.5:
                is_outlier = True
        
        # Additional check: extreme win rates with sufficient sample size
        if games >= 50:  # Only for large sample sizes
            if wr <= 0.05 or wr >= 0.95:  # Less than 5% or more than 95% win rate
                is_outlier = True
        
        if is_outlier:
            outlier_indices.append(i)
    
    return outlier_indices


def get_outlier_stats(candidates: List[Tuple[Dict[str,float], float, int, int]], outlier_indices: List[int]) -> Dict[str, Any]:
    """Phase 3: Generate statistics about detected outliers"""
    if not outlier_indices:
        return {'outlier_count': 0, 'outlier_rate': 0.0}
    
    outlier_rates = [candidates[i][1] for i in outlier_indices]
    normal_rates = [candidates[i][1] for i in range(len(candidates)) if i not in outlier_indices]
    
    import statistics
    
    stats = {
        'outlier_count': len(outlier_indices),
        'outlier_rate': len(outlier_indices) / len(candidates),
        'outlier_win_rates': outlier_rates,
        'normal_win_rates': normal_rates,
        'outlier_mean': statistics.mean(outlier_rates) if outlier_rates else 0,
        'normal_mean': statistics.mean(normal_rates) if normal_rates else 0,
        'outlier_indices': outlier_indices
    }
    
    return stats


def cross_validate_champion(format_id: str, champion: Dict[str, float], baseline: Dict[str, float], 
                          k_folds: int = 5, seeds_per_fold: int = 32, offline: bool = True) -> Dict[str, Any]:
    """Phase 3: Cross-validation for weight stability testing"""
    import random
    
    # Generate seeds for cross-validation
    cv_seeds = [random.randint(1, 1_000_000_000) for _ in range(k_folds * seeds_per_fold)]
    
    fold_results = []
    
    for fold in range(k_folds):
        # Get seeds for this fold
        fold_start = fold * seeds_per_fold
        fold_end = fold_start + seeds_per_fold
        fold_seeds = cv_seeds[fold_start:fold_end]
        
        try:
            # Evaluate champion vs baseline on this fold
            wr, wins, games = evaluate_candidate(format_id, fold_seeds, baseline, champion, offline=offline)
            
            fold_result = {
                'fold': fold,
                'win_rate': wr,
                'wins': wins,
                'games': games,
                'seeds': fold_seeds
            }
            fold_results.append(fold_result)
            
        except Exception as e:
            fold_result = {
                'fold': fold,
                'win_rate': 0.0,
                'wins': 0,
                'games': 0,
                'error': str(e),
                'seeds': fold_seeds
            }
            fold_results.append(fold_result)
    
    # Calculate cross-validation statistics
    valid_results = [r for r in fold_results if 'error' not in r and r['games'] > 0]
    
    if not valid_results:
        return {
            'k_folds': k_folds,
            'seeds_per_fold': seeds_per_fold,
            'fold_results': fold_results,
            'cv_mean': 0.0,
            'cv_std': 0.0,
            'cv_stability': 0.0,
            'is_stable': False,
            'error': 'No valid folds completed'
        }
    
    import statistics
    win_rates = [r['win_rate'] for r in valid_results]
    
    cv_mean = statistics.mean(win_rates)
    cv_std = statistics.stdev(win_rates) if len(win_rates) > 1 else 0.0
    
    # Calculate stability score (lower std = more stable)
    cv_stability = 1.0 - min(cv_std * 2, 1.0)  # Normalize to 0-1 scale
    
    # Consider stable if std dev < 0.1 and mean > 0.52
    is_stable = cv_std < 0.1 and cv_mean > 0.52
    
    cv_stats = {
        'k_folds': k_folds,
        'seeds_per_fold': seeds_per_fold,
        'valid_folds': len(valid_results),
        'fold_results': fold_results,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'cv_stability': cv_stability,
        'is_stable': is_stable,
        'confidence_interval_95': [
            max(0.0, cv_mean - 1.96 * cv_std / math.sqrt(len(valid_results))),
            min(1.0, cv_mean + 1.96 * cv_std / math.sqrt(len(valid_results)))
        ]
    }
    
    return cv_stats


def main():
    ap = argparse.ArgumentParser(description='Evolutionary tuner for evaluation weights.')
    ap.add_argument('--format', default='gen9randombattle')
    ap.add_argument('--population', type=int, default=32)
    ap.add_argument('--generations', type=int, default=50)
    ap.add_argument('--seeds', type=int, default=128, help='Number of base seeds (each used twice: forward+reverse) - Phase 3: Increased from 32 to 128 for better statistical power')
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
    # Phase 3: Cross-validation options
    ap.add_argument('--cross-validate', action='store_true', help='Enable cross-validation for champion stability testing')
    ap.add_argument('--cv-frequency', type=int, default=10, help='Run cross-validation every N generations')
    ap.add_argument('--cv-folds', type=int, default=5, help='Number of folds for cross-validation')
    args = ap.parse_args()

    # Phase 1: Data validation before weight tuning runs
    validation_errors = validate_tuning_parameters(args)
    if validation_errors:
        print(f"[ERROR] Weight tuning validation failed:")
        for error in validation_errors:
            print(f"  - {error}")
        print("[ERROR] Aborting weight tuning due to validation failures")
        return 1

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
            
            # Phase 1: Validate candidate results for corruption detection
            if not validate_candidate_result(wr, wins, games):
                print(f"[gen {gen}] cand {i} INVALID RESULT: wr={wr:.3f} wins={wins}/{games} - marking as failed")
                log_event({'event':'candidate_invalid','generation':gen,'idx':i,'weights':cand,'win_rate':wr,'wins':wins,'games':games,'sigma':sigma,'reason':'validation_failed','offline':offline_flag})
                # Use a poor but valid result to continue the evolution process
                wr, wins, games = 0.0, 0, max(1, games)
            
            candidates.append((cand, wr, wins, games))
            log_event({'event':'candidate','generation':gen,'idx':i,'weights':cand,'win_rate':wr,'wins':wins,'games':games,'sigma':sigma,'champion_wr_at_gen_start':champion_wr,'offline':offline_flag})
            print(f"[gen {gen}] cand {i} wr={wr:.3f} wins={wins}/{games} sigma={sigma:.3f}")
        # Phase 3: Outlier detection for extreme win rates
        outlier_indices = detect_outliers_in_generation(candidates)
        outlier_stats = get_outlier_stats(candidates, outlier_indices)
        
        if outlier_indices:
            print(f"[gen {gen}] Detected {len(outlier_indices)} outliers: {[f'cand{i}' for i in outlier_indices]}")
            log_event({
                'event': 'outliers_detected',
                'generation': gen,
                'outlier_stats': outlier_stats,
                'offline': offline_flag
            })
        
        # Phase 3: Confidence interval analysis for training metrics
        ci_analysis = analyze_training_metrics_with_ci(candidates, confidence=args.confidence)
        
        # Report significant differences
        if ci_analysis.get('significant_differences'):
            sig_diffs = ci_analysis['significant_differences']
            print(f"[gen {gen}] Found {len(sig_diffs)} statistically significant performance differences")
            
        # Log comprehensive confidence interval analysis
        log_event({
            'event': 'confidence_interval_analysis',
            'generation': gen,
            'ci_analysis': ci_analysis,
            'offline': offline_flag
        })
        
        # Select best
        candidates.sort(key=lambda t: t[1], reverse=True)
        best, best_wr, best_wins, best_games = candidates[0]
        
        # Check if best candidate is an outlier (potential corruption)
        if 0 in outlier_indices:  # Index 0 after sorting is the best candidate
            print(f"[gen {gen}] WARNING: Best candidate (wr={best_wr:.3f}) detected as outlier - reviewing")
            # Use second best if available and not an outlier
            for i, (cand, wr, wins, games) in enumerate(candidates[1:], 1):
                if i not in outlier_indices:
                    print(f"[gen {gen}] Using candidate {i} (wr={wr:.3f}) instead of outlier")
                    best, best_wr, best_wins, best_games = cand, wr, wins, games
                    break
        
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
            
            # Phase 3: Cross-validation after champion promotion
            if args.cross_validate:
                print(f"[gen {gen}] Running cross-validation for promoted champion...")
                try:
                    # Use default weights as baseline for cross-validation
                    baseline_engine = StockfishModel()
                    baseline_weights = dict(baseline_engine._W)
                    
                    cv_results = cross_validate_champion(
                        args.format, champion, baseline_weights,
                        k_folds=args.cv_folds, seeds_per_fold=32, offline=offline_flag
                    )
                    
                    log_event({
                        'event': 'cross_validation',
                        'generation': gen,
                        'cv_results': cv_results,
                        'offline': offline_flag
                    })
                    
                    print(f"[gen {gen}] CV: mean={cv_results['cv_mean']:.3f} std={cv_results['cv_std']:.3f} stability={cv_results['cv_stability']:.3f} stable={cv_results['is_stable']}")
                    
                    # Warn if champion is not stable
                    if not cv_results['is_stable']:
                        print(f"[gen {gen}] WARNING: Champion shows poor cross-validation stability")
                        
                except Exception as e:
                    print(f"[gen {gen}] Cross-validation failed: {e}")
                    
        else:
            no_improve += 1
            sigma = max(sigma * 0.9, 0.05)
            log_event({'event':'no_improve','generation':gen,'best_wr':best_wr,'diff':diff,'no_improve':no_improve,'sigma':sigma,'gen_seconds':gen_seconds,'offline':offline_flag})
            print(f"[gen {gen}] no improvement best_wr={best_wr:.3f} diff={diff:.3f} no_improve={no_improve} sigma->{sigma:.3f} time={gen_seconds:.1f}s")
        
        # Phase 3: Periodic cross-validation check (even without promotion)
        if args.cross_validate and gen % args.cv_frequency == 0 and not promote:
            print(f"[gen {gen}] Running periodic cross-validation check...")
            try:
                baseline_engine = StockfishModel()
                baseline_weights = dict(baseline_engine._W)
                
                cv_results = cross_validate_champion(
                    args.format, champion, baseline_weights,
                    k_folds=args.cv_folds, seeds_per_fold=24, offline=offline_flag  # Smaller CV for periodic checks
                )
                
                log_event({
                    'event': 'periodic_cross_validation',
                    'generation': gen,
                    'cv_results': cv_results,
                    'offline': offline_flag
                })
                
                print(f"[gen {gen}] Periodic CV: mean={cv_results['cv_mean']:.3f} std={cv_results['cv_std']:.3f} stable={cv_results['is_stable']}")
                
            except Exception as e:
                print(f"[gen {gen}] Periodic cross-validation failed: {e}")

    print(f"[done] generations={gen} final_champion_wr={champion_wr:.3f} weights={champion}")
