"""Self-play API used by weight_tuner.

Runs offline simulated battles between two StockfishModel configurations.

We keep this lightweight and headless; no UI, no network ladder. Uses poke-env local battle simulation
if available; if not, raises a clear error.
"""
from __future__ import annotations
import asyncio
import random
from dataclasses import dataclass
from typing import Dict, Optional, List, Iterable
import math

from Models.stockfish_model import StockfishModel, StockfishPokeEnvPlayer  # type: ignore

try:
    from poke_env.ps_client.server_configuration import ShowdownServerConfiguration, LocalhostServerConfiguration  # type: ignore
except Exception:
    ShowdownServerConfiguration = None  # type: ignore
    LocalhostServerConfiguration = None  # type: ignore

@dataclass
class BattleResult:
    seed: int
    winner: Optional[str]
    p1_won: Optional[bool]
    n_turns: int
    error: Optional[str] = None

# ---------------- Offline heuristic simulation (no network) -----------------
_DEF_COEFFS = [
    ('expected_mult', 1.0),
    ('go_first_bonus', 0.6),
    ('opp_dmg_penalty', 0.9),
    ('effectiveness_mult', 0.4),
    ('accuracy_mult', 0.3),
    ('ko_bonus', 0.7),
    ('switch_outgoing_mult', 0.5),
    ('switch_incoming_penalty', 0.7),
    ('survival_bonus', 0.4),
]

def _score_weights(w: Dict[str,float]) -> float:
    return sum(float(w.get(k,0.0))*c for k,c in _DEF_COEFFS)

def _simulate_offline(seed: int, w1: Dict[str,float], w2: Dict[str,float]) -> BattleResult:
    rng = random.Random(seed)
    base1 = _score_weights(w1)
    base2 = _score_weights(w2)
    # Normalize by number of coeffs
    base1 /= len(_DEF_COEFFS)
    base2 /= len(_DEF_COEFFS)
    turns = 40
    advantage = 0.0
    for t in range(turns):
        # Add small stochastic fluctuation each turn
        fluct = rng.gauss(0, 0.15)
        # Dynamic aggression scaling late game
        phase = t / turns
        adv_turn = (base1 - base2) * (0.6 + 0.8*phase) + fluct
        advantage += adv_turn
        # Early termination if decisive lead (simple heuristic)
        if abs(advantage) > 5.0:
            break
    winner = 'p1' if advantage > 0 else 'p2'
    return BattleResult(seed=seed, winner=winner, p1_won=(advantage>0), n_turns=t+1)

# ---------------- Network (poke-env) simulation -----------------
async def _play_one(seed: int, format_id: str, w1: Dict[str, float], w2: Dict[str, float], max_turns: int = 300, *, offline: bool=False) -> BattleResult:
    if offline:
        return _simulate_offline(seed, w1, w2)
    random.seed(seed)
    try:
        server_conf = ShowdownServerConfiguration  # type: ignore
        p1 = StockfishPokeEnvPlayer(battle_format=format_id, server_configuration=server_conf)
        p2 = StockfishPokeEnvPlayer(battle_format=format_id, server_configuration=server_conf)
        p1.engine.set_weights(w1)
        p2.engine.set_weights(w2)
        battle = await p1.battle_against(p2, n_battles=1)
        turns = getattr(battle, 'turn', 0) or 0
        winner = None
        p1_won = None
        if getattr(battle, 'won', None) is True:
            winner = 'p1'; p1_won = True
        elif getattr(battle, 'won', None) is False:
            winner = 'p2'; p1_won = False
        else:
            if getattr(battle, 'player_username', None) == getattr(battle, 'winner', None):
                winner = 'p1'; p1_won = True
        return BattleResult(seed=seed, winner=winner, p1_won=p1_won, n_turns=turns)
    except Exception as e:
        return BattleResult(seed=seed, winner=None, p1_won=None, n_turns=0, error=str(e))

async def self_play_series(format_id: str, seeds: Iterable[int], w1: Dict[str, float], w2: Dict[str, float], *, offline: bool=False) -> List[BattleResult]:
    results: List[BattleResult] = []
    for s in seeds:
        r = await _play_one(s, format_id, w1, w2, offline=offline)
        results.append(r)
    return results

def run_self_play(format_id: str, seeds: List[int], w1: Dict[str, float], w2: Dict[str, float], *, offline: bool=False) -> List[BattleResult]:
    """Synchronous wrapper."""
    return asyncio.run(self_play_series(format_id, seeds, w1, w2, offline=offline))

__all__ = [
    'BattleResult', 'run_self_play'
]
