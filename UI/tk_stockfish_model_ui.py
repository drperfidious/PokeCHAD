
from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import subprocess
import sys
import threading
from typing import Any, Dict, List, Optional

import tkinter as tk
from tkinter import ttk, messagebox

# Ensure repo root is on sys.path so `Data` and `Models` can be imported
_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# -- Optional runtime patch to the model (accept on_think, set engine_depth, robust forfeit)
try:
    from UI.ui_think_patch import patch_stockfish_player  # noqa: F401
except Exception:  # patch file might not be present yet
    patch_stockfish_player = None  # type: ignore

from Models.stockfish_model import StockfishPokeEnvPlayer  # type: ignore
from Data.poke_env_battle_environment import snapshot as snapshot_battle  # type: ignore
from poke_env.ps_client.account_configuration import AccountConfiguration  # type: ignore
from poke_env.ps_client.server_configuration import (  # type: ignore
    ShowdownServerConfiguration,
    LocalhostServerConfiguration,
)

# ----------------------- Formats dropdown ---------------------------------
KNOWN_FORMATS = [
    "gen9randombattle",
    "gen9unratedrandombattle",
    "gen9randomdoublesbattle",
    "gen9hackmonscup",
    "gen9ou", "gen9ubers", "gen9uu", "gen9ru", "gen9nu", "gen9pu", "gen9lc", "gen9monotype",
    "gen9doublesou",
    "vgc2025regh",
]

# --------------------------- Small helpers --------------------------------
class QueueLogHandler(logging.Handler):
    def __init__(self, q: "queue.Queue[str]"):
        super().__init__()
        self.q = q
    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            self.q.put_nowait(msg)
        except Exception:
            pass

def pretty_boosts(boosts: Dict[str, int] | None) -> str:
    if not boosts:
        return ""
    ordered = ["atk", "def", "spa", "spd", "spe", "acc", "evasion"]
    return ", ".join(f"{k}+{v}" if v > 0 else f"{k}{v}" for k, v in ((k, boosts.get(k, 0)) for k in ordered) if v != 0)

# --------------------------------- Window ---------------------------------
class StockfishWindow(tk.Toplevel):
    def __init__(self, parent: tk.Tk, username: str, password: Optional[str], server_mode: str,
                 custom_ws: Optional[str], battle_format: str):
        super().__init__(parent)
        self.title("PokeCHAD — Stockfish Model")
        self.geometry("1180x740")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.username = username
        self.password = password
        self.server_mode = server_mode
        self.custom_ws = custom_ws
        self.battle_format = battle_format

        # Telemetry file
        os.makedirs("logs", exist_ok=True)
        self._telemetry_path = os.path.join("logs", f"telemetry_{os.getpid()}.jsonl")

        # Async loop thread
        self.loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self._loop_thread.start()

        # Player
        self.player: Optional[StockfishPokeEnvPlayer] = None

        # Logging pane & handler
        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.log_handler = QueueLogHandler(self.log_queue)
        self.log_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

        # UI State
        self._scheduled_tasks: List[str] = []
        self._latest_think: Dict[str, Any] = {}
        self._latest_snapshot: Dict[str, Any] = {}
        self._last_fallback_turn: Optional[int] = None
        self._last_real_think_turn: Optional[int] = None
        self._root_log_handler_attached = False

        self._build_ui()
        self._pump_logs()

        # Bootstrap: connect immediately
        self._submit(self._async_connect())

    def _submit(self, coro: "asyncio.coroutines.Coroutine[Any, Any, Any]"):
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        def handle_result():
            try:
                future.result(timeout=0.1)
            except asyncio.TimeoutError:
                self.after(100, handle_result)
            except Exception as e:
                error_msg = f"Action failed: {e}"
                self._append_log(error_msg)
                messagebox.showerror("Error", error_msg)
        self.after(100, handle_result)

    def _call_on_main(self, fn, delay_ms: int = 0):
        if delay_ms <= 0:
            return self.after(0, fn)
        return self.after(delay_ms, fn)

    # ---------- UI construction ----------
    def _build_ui(self):
        nb = ttk.Notebook(self); nb.pack(fill=tk.BOTH, expand=True)

        # --- Dashboard tab
        dash = ttk.Frame(nb); nb.add(dash, text="Dashboard")
        controls = ttk.LabelFrame(dash, text="Controls"); controls.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        ttk.Label(controls, text=f"User: {self.username}").pack(side=tk.LEFT, padx=4)
        ttk.Label(controls, text="Format:").pack(side=tk.LEFT, padx=(14, 2))
        self.format_var = tk.StringVar(value=self.battle_format)
        self.format_combo = ttk.Combobox(controls, textvariable=self.format_var, values=KNOWN_FORMATS, width=26, state="readonly")
        self.format_combo.pack(side=tk.LEFT, padx=4)

        ttk.Label(controls, text="Depth:").pack(side=tk.LEFT, padx=(14, 2))
        self.depth_var = tk.IntVar(value=1)
        self.depth_spin = ttk.Spinbox(controls, from_=1, to=3, textvariable=self.depth_var, width=4, command=self._on_depth_changed)
        self.depth_spin.pack(side=tk.LEFT, padx=4)

        ttk.Button(controls, text="Ladder 1", command=lambda: self._submit(self._ladder(1))).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Challenge…", command=self._challenge_dialog).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Accept 1", command=lambda: self._submit(self._accept(1))).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Start Timer", command=lambda: self._submit(self._timer_all(True))).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Forfeit", command=lambda: self._submit(self._forfeit_all())).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Train Weights", command=self._train_weights).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Reload Weights", command=self._reload_weights).pack(side=tk.LEFT, padx=4)

        # Team panes
        teams = ttk.Frame(dash); teams.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))
        self.team_tree = self._make_team_tree(teams, "Your team")
        self.opp_tree = self._make_team_tree(teams, "Opponent team")

        # --- Thinking tab
        think_tab = ttk.Frame(nb); nb.add(think_tab, text="Thinking")
        self.cand_tree = self._make_cand_tree(think_tab, "Move candidates")
        self.switch_tree = self._make_switch_tree(think_tab, "Switch candidates")

        # --- Logs tab
        logs_tab = ttk.Frame(nb); nb.add(logs_tab, text="Logs")
        self.logs_text = tk.Text(logs_tab, height=20, wrap="word")
        self.logs_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

    def _make_team_tree(self, parent, title: str) -> ttk.Treeview:
        frame = ttk.LabelFrame(parent, text=title)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)
        cols = ("slot", "species", "hp", "status", "boosts")
        tree = ttk.Treeview(frame, columns=cols, show="headings", height=12)
        for c, w in zip(cols, (60, 160, 60, 80, 220)):
            tree.heading(c, text=c.upper())
            tree.column(c, width=w, anchor=tk.W)
        tree.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        return tree

    def _make_cand_tree(self, parent, title: str) -> ttk.Treeview:
        frame = ttk.LabelFrame(parent, text=title); frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=6)
        cols = ("move", "score", "exp_dmg", "acc", "eff", "first", "opp", "note")
        tree = ttk.Treeview(frame, columns=cols, show="headings", height=12)
        hdrs = ("MOVE", "SCORE", "EXP", "ACC", "EFF", "FIRST", "OPP", "WHY/NOTE")
        widths = (200, 80, 70, 60, 60, 60, 60, 260)
        for c, h, w in zip(cols, hdrs, widths):
            tree.heading(c, text=h)
            tree.column(c, width=w, anchor=tk.W)
        tree.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        return tree

    def _make_switch_tree(self, parent, title: str) -> ttk.Treeview:
        frame = ttk.LabelFrame(parent, text=title); frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=6)
        cols = ("species", "score", "base", "out", "in", "haz", "hp")
        tree = ttk.Treeview(frame, columns=cols, show="headings", height=8)
        headers = ("SPECIES", "SCORE", "BASE", "OUT", "IN", "HAZ", "HP")
        widths = (140, 70, 70, 60, 60, 50, 60)
        for c, h, w in zip(cols, headers, widths):
            tree.heading(c, text=h)
            tree.column(c, width=w, anchor=tk.W)
        tree.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        return tree

    # ---------- Connect ----------
    async def _async_connect(self):
        try:
            if patch_stockfish_player:
                patch_stockfish_player()
        except Exception as e:
            self._append_log(f"Model patch failed (non-fatal): {e}")

        account = AccountConfiguration(self.username, self.password)
        if self.server_mode == "Showdown":
            server = ShowdownServerConfiguration
        elif self.server_mode == "Localhost":
            server = LocalhostServerConfiguration
        else:
            server = ShowdownServerConfiguration
            if self.custom_ws:
                try:
                    server = type("CustomServerConf", (tuple,), {})((self.custom_ws, ShowdownServerConfiguration[1]))
                except Exception:
                    server = ShowdownServerConfiguration

        common = dict(
            account_configuration=account,
            server_configuration=server,
            battle_format=self.battle_format,
            log_level=logging.INFO,
        )

        try:
            depth_val = int(self.depth_var.get())
        except Exception:
            depth_val = None

        player = None; last_error = None
        sigs = [
            dict(on_think=self._on_think, engine_depth=depth_val, start_listening=True),
            dict(on_think=self._on_think, engine_depth=depth_val),
            dict(on_think=self._on_think),
            dict(engine_depth=depth_val, start_listening=True),
            dict(engine_depth=depth_val),
            dict(),
        ]
        for extra in sigs:
            try:
                kv = {k: v for k, v in extra.items() if v is not None}
                player = StockfishPokeEnvPlayer(**common, **kv); break
            except TypeError as e:
                last_error = e; continue
        if player is None:
            self._append_log(f"Failed to construct player with extended kwargs; retrying minimal. Last error: {last_error}")
            player = StockfishPokeEnvPlayer(**common)

        self.player = player

        try:
            attached = False
            for attr in ("on_think", "think_callback", "on_think_hook"):
                if hasattr(self.player, attr):
                    setattr(self.player, attr, self._on_think)
                    attached = True; break
            if not attached:
                for meth in ("set_on_think", "set_think_callback", "register_think_callback", "on_think_connect"):
                    fn = getattr(self.player, meth, None)
                    if callable(fn):
                        fn(self._on_think); attached = True; break
            if not attached:
                self._append_log("Note: model exposes no on_think hook; Thinking tab will use fallback per turn.")
        except Exception as e:
            self._append_log(f"Could not attach think callback: {e}")

        try:
            self.player.logger.addHandler(self.log_handler)
            self.player.logger.setLevel(logging.INFO)
        except Exception:
            pass
        try:
            if not getattr(self, "_root_log_handler_attached", False):
                root = logging.getLogger()
                root.addHandler(self.log_handler)
                if root.level > logging.INFO:
                    root.setLevel(logging.INFO)
                self._root_log_handler_attached = True
        except Exception:
            pass

        await self.player.ps_client.wait_for_login()
        self._append_log("Login confirmed. Ready.")
        self._call_on_main(self._poll_battle)

    # ---------- Actions ----------
    async def _ladder(self, n: int):
        if not self.player: return
        if getattr(self.player, "format", None) != self.format_var.get():
            try: self.player.format = self.format_var.get()
            except Exception: pass
        self._append_log(f"Starting ladder: {n} game(s)…")
        await self.player.ladder(n)

    async def _accept(self, n: int):
        if not self.player: return
        self._append_log(f"Accepting {n} challenge(s)…")
        await self.player.accept_challenges(opponent=None, n_challenges=n)

    def _challenge_dialog(self):
        if not self.player: return
        dlg = tk.Toplevel(self); dlg.title("Challenge a user")
        ttk.Label(dlg, text="Opponent username:").pack(side=tk.TOP, padx=8, pady=8)
        name_var = tk.StringVar(); ttk.Entry(dlg, textvariable=name_var, width=28).pack(side=tk.TOP, padx=8, pady=(0, 8))
        def go():
            opp = name_var.get().strip()
            if opp: self._submit(self.player.send_challenges(opp, n_challenges=1))
            dlg.destroy()
        ttk.Button(dlg, text="Challenge", command=go).pack(side=tk.TOP, padx=8, pady=8)

    async def _forfeit_all(self):
        p = self.player
        if not p: return
        try:
            m = getattr(p, "forfeit_all", None)
            if callable(m):
                await m(); self._append_log("Called player.forfeit_all()."); return
        except Exception as e:
            self._append_log(f"player.forfeit_all() failed: {e} — falling back to direct /forfeit.")
        try:
            client = getattr(p, "ps_client", None) or getattr(p, "_client", None)
            if not client: raise RuntimeError("PSClient missing on player")
            battles = getattr(p, "battles", {}) or {}
            rooms: List[str] = []
            for key, battle in list(battles.items()):
                room_id = getattr(battle, "battle_tag", None) or getattr(battle, "room_id", None) or str(key)
                if room_id: rooms.append(room_id)
            if not rooms:
                self._append_log("No active battle rooms found for /forfeit."); return
            sent = 0
            for r in rooms:
                try:
                    await client.send_message("/forfeit", room=r); sent += 1
                except Exception as e2:
                    self._append_log(f"Failed to send /forfeit to {r}: {e2}")
            self._append_log(f"Sent /forfeit to {sent} room(s).")
        except Exception as e:
            self._append_log(f"Forfeit fallback failed: {e}")

    async def _timer_all(self, on: bool):
        if self.player: await self.player.timer_all(on)

    def _on_depth_changed(self):
        if self.player:
            try:
                eng = getattr(self.player, "engine", None)
                if eng and hasattr(eng, "set_depth"):
                    eng.set_depth(int(self.depth_var.get()))
                elif hasattr(self.player, "set_depth"):
                    self.player.set_depth(int(self.depth_var.get()))
            except Exception:
                pass

    # ---------- Train / Reload ----------
    def _train_weights(self):
        def run():
            try:
                cmd = [sys.executable, "../tools/train_launcher.py"]
                self._append_log("[run] " + " ".join(cmd))
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.stdout: self._append_log(proc.stdout.strip())
                if proc.returncode != 0:
                    if proc.stderr: self._append_log(proc.stderr.strip())
                    messagebox.showerror("Training failed", proc.stderr or "trainer returned non-zero")
                else:
                    self._append_log("[ok] Training finished.")
            except Exception as e:
                self._append_log(f"Training failed: {e}")
        threading.Thread(target=run, daemon=True).start()

    def _reload_weights(self):
        try:
            eng = getattr(self.player, "engine", None)
            if eng and hasattr(eng, "reload_weights"):
                eng.reload_weights()
                messagebox.showinfo("Weights", "Weights reloaded from Models/weights.json")
            else:
                messagebox.showwarning("Weights", "Engine does not expose reload_weights()")
        except Exception as e:
            messagebox.showerror("Weights", f"Reload failed: {e}")

    # ---------- Think data from model ----------
    def _on_think(self, battle, think: Dict[str, Any]):
        self._latest_think = think or {}
        try: snap = think.get("snapshot")
        except Exception: snap = None
        self._latest_snapshot = snap or snapshot_battle(battle)
        try:
            self._last_real_think_turn = int(self._latest_snapshot.get("turn")) if self._latest_snapshot else None
        except Exception:
            pass
        # write JSONL telemetry if there is a picked decision
        try:
            if think.get("picked"):
                entry = {
                    "battle_tag": think.get("battle_tag") or getattr(battle, "battle_tag", None) or getattr(battle, "room_id", None),
                    "turn": think.get("turn") or self._latest_snapshot.get("turn"),
                    "picked": think.get("picked"),
                    "order": think.get("order"),
                    "snapshot": self._latest_snapshot,
                }
                with open(self._telemetry_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
        except Exception as e:
            self._append_log(f"telemetry write failed: {e}")

        self._call_on_main(self._refresh_thinking)
        self._call_on_main(self._refresh_teams)

    def _refresh_thinking(self):
        if not self.winfo_exists(): return
        # Candidates
        self._reload_tree(self.cand_tree)
        for d in self._latest_think.get("candidates", []):
            try:
                move = d.get("name") or d.get("id") or d.get("move") or d.get("move_id")
                s = d.get("score"); score = f"{float(s):.2f}" if s is not None else ""
                exp = ""; 
                for k in ("exp_dmg", "expected", "exp", "expdmg", "expected_damage"):
                    if d.get(k) is not None:
                        exp = f"{float(d.get(k)):.2f}"; break
                acc = ""; 
                for k in ("acc", "acc_mult", "accuracy", "hit_chance"):
                    if d.get(k) is not None:
                        acc = f"{float(d.get(k)):.2f}"; break
                eff = ""; 
                for k in ("eff", "effectiveness", "type_mult", "type_effectiveness"):
                    if d.get(k) is not None:
                        eff = f"{float(d.get(k)):.2f}"; break
                first = ""; 
                if d.get("first_prob") is not None: first = f"{float(d.get('first_prob')):.2f}"
                opp = ""; 
                if d.get("opp_counter_ev") is not None: opp = f"{float(d.get('opp_counter_ev')):.2f}"
                note = d.get("why_blocked") or d.get("note") or d.get("why") or ""
                self.cand_tree.insert("", tk.END, values=(move, score, exp, acc, eff, first, opp, note))
            except Exception:
                try:
                    self.cand_tree.insert("", tk.END, values=(str(d), "", "", "", "", "", "", ""))
                except Exception:
                    pass

        # Switches
        self._reload_tree(self.switch_tree)
        for d in self._latest_think.get("switches", []):
            try:
                species = d.get("species") or d.get("name") or d.get("id")
                s = d.get("score"); score = f"{float(s):.2f}" if s is not None else ""
                base = d.get("base_score"); base_s = f"{float(base):.2f}" if base is not None else ""
                out = d.get("outgoing_frac"); out_s = f"{float(out):.2f}" if out is not None else ""
                incoming = d.get("incoming_on_switch"); in_s = f"{float(incoming):.2f}" if incoming is not None else ""
                haz = d.get("hazards_frac"); haz_s = f"{float(haz):.2f}" if haz is not None else ""
                hp = d.get("hp_fraction"); hp_s = f"{int(round(float(hp) * 100))}%" if isinstance(hp, (int, float)) else ""
                self.switch_tree.insert("", tk.END, values=(species, score, base_s, out_s, in_s, haz_s, hp_s))
            except Exception:
                try:
                    species = d.get("species") or str(d)
                    score = f"{float(d.get('score', 0)):.2f}" if d.get('score') is not None else "0.00"
                    self.switch_tree.insert("", tk.END, values=(species, score, "", "", "", "", ""))
                except Exception:
                    pass

    def _refresh_teams(self):
        if not self.winfo_exists(): return
        snap = self._latest_snapshot or {}
        self._reload_tree(self.team_tree)
        for sid, p in (snap.get("my_team") or {}).items():
            boosts = pretty_boosts(p.get("boosts"))
            hp = p.get("hp_fraction")
            hp_s = f"{int(round(hp * 100))}%" if isinstance(hp, (int, float)) else ""
            self.team_tree.insert("", tk.END, values=(sid, p.get("species"), hp_s, str(p.get("status") or ""), boosts))
        self._reload_tree(self.opp_tree)
        for sid, p in (snap.get("opp_team") or {}).items():
            boosts = pretty_boosts(p.get("boosts"))
            hp = p.get("hp_fraction")
            hp_s = f"{int(round(hp * 100))}%" if isinstance(hp, (int, float)) else ""
            self.opp_tree.insert("", tk.END, values=(sid, p.get("species"), hp_s, str(p.get("status") or ""), boosts))

    def _reload_tree(self, tree: ttk.Treeview):
        try:
            for iid in tree.get_children():
                tree.delete(iid)
        except tk.TclError:
            pass

    # ---------- Polling ----------
    def _find_active_battle(self):
        p = getattr(self, "player", None)
        if not p: return None
        for name in ("current_battle", "battle", "active_battle"):
            b = getattr(p, name, None)
            if b is not None: return b
        battles = getattr(p, "battles", None)
        if isinstance(battles, dict) and battles:
            try:
                for b in battles.values():
                    if getattr(b, "active_pokemon", None) is not None or getattr(b, "turn", None):
                        return b
                return list(battles.values())[-1]
            except Exception:
                try: return next(iter(battles.values()))
                except Exception: return None
        return None

    def _poll_battle(self):
        try:
            if not self.winfo_exists(): return
            b = self._find_active_battle()
            if b is not None:
                try: snap = snapshot_battle(b); self._latest_snapshot = snap
                except Exception: snap = None
                try: self._refresh_teams()
                except Exception: pass
                try:
                    turn = int(snap.get("turn")) if snap else None
                except Exception:
                    turn = None
                if turn is not None and turn != self._last_fallback_turn and turn != self._last_real_think_turn:
                    self._emit_fallback_think(b, snap); self._last_fallback_turn = turn
        finally:
            if self.winfo_exists():
                try: h = self.after(500, self._poll_battle); self._scheduled_tasks.append(h)
                except Exception: pass

    def _emit_fallback_think(self, battle, snap: Optional[Dict[str, Any]]):
        try:
            cands = []
            for m in (getattr(battle, "available_moves", None) or []):
                try:
                    name = getattr(m, "name", None) or getattr(m, "id", None) or str(m)
                    bp = getattr(m, "base_power", None) or getattr(m, "basePower", None) or 0
                    acc = getattr(m, "accuracy", None)
                    if acc is True: acc_val = 1.0
                    elif isinstance(acc, (int, float)): acc_val = float(acc) / (100.0 if acc > 1 else 1.0)
                    else: acc_val = 1.0
                    expected = float(bp or 0) * float(acc_val)
                    cands.append({"name": name, "score": expected, "exp_dmg": expected, "acc": acc_val, "eff": "", "note": "synthetic"})
                except Exception: pass

            switches = []
            for pkm in (getattr(battle, "available_switches", None) or []):
                try:
                    species = getattr(pkm, "species", None) or getattr(pkm, "name", None) or str(pkm)
                    hp_frac = getattr(pkm, "hp_fraction", None) or getattr(pkm, "current_hp_fraction", None)
                    switches.append({"species": species, "score": float(hp_frac or 0.0), "hp_fraction": float(hp_frac or 0.0)})
                except Exception: pass

            think = {"candidates": sorted(cands, key=lambda d: d.get("score") or 0.0, reverse=True),
                     "switches": sorted(switches, key=lambda d: d.get("score") or 0.0, reverse=True),
                     "snapshot": snap or snapshot_battle(battle)}
            self._on_think(battle, think)
        except Exception as e:
            self._append_log(f"Fallback think failed: {e}")

    # ---------- Logs ----------
    def _append_log(self, msg: str):
        try:
            self.logs_text.insert(tk.END, msg + "\n")
            self.logs_text.see(tk.END)
        except Exception:
            pass

    def _pump_logs(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self._append_log(msg)
        except queue.Empty:
            pass
        if self.winfo_exists():
            h = self.after(200, self._pump_logs)
            self._scheduled_tasks.append(h)

    # ---------- Shutdown ----------
    def _on_close(self):
        for h in self._scheduled_tasks:
            try: self.after_cancel(h)
            except Exception: pass
        self._scheduled_tasks.clear()
        try:
            if self.loop.is_running():
                self.loop.call_soon_threadsafe(self.loop.stop)
        except Exception:
            pass
        self.destroy()

def launch_stockfish_window(root: tk.Tk, username: str, password: Optional[str],
                            server_mode: str, custom_ws: Optional[str], battle_format: str) -> StockfishWindow:
    return StockfishWindow(root, username=username, password=password,
                           server_mode=server_mode, custom_ws=custom_ws, battle_format=battle_format)
