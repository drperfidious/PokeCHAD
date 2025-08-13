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
from pathlib import Path

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
    CONFIG_PATH = Path.home() / '.pokechad_ui_settings.json'
    def __init__(self, parent: tk.Tk, username: str, password: Optional[str], server_mode: str,
                 custom_ws: Optional[str], battle_format: str):
        super().__init__(parent)
        self.title("PokeCHAD — Stockfish Model")
        self.geometry("1180x740")
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        # Load persisted prefs early
        self._prefs = self._load_prefs()

        self.username = username
        self.password = password
        self.server_mode = server_mode
        self.custom_ws = custom_ws
        self.battle_format = battle_format or self._prefs.get('stockfish_ui', {}).get('format') or battle_format

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
        self._finished_battles: set[str] = set()
        self._active_battle_id: Optional[str] = None

        self._build_ui()
        self._apply_loaded_prefs()  # set widget values from prefs after UI built
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
    def _load_prefs(self) -> dict:
        try:
            if self.CONFIG_PATH.exists():
                with open(self.CONFIG_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
        return {}

    def _save_prefs(self):
        try:
            base = {}
            if self.CONFIG_PATH.exists():
                try:
                    with open(self.CONFIG_PATH, 'r', encoding='utf-8') as f:
                        base = json.load(f) or {}
                        if not isinstance(base, dict):
                            base = {}
                except Exception:
                    base = {}
            ui = base.get('stockfish_ui', {}) if isinstance(base.get('stockfish_ui'), dict) else {}
            ui.update({
                'depth': int(self.depth_var.get()) if hasattr(self, 'depth_var') else None,
                'branching': int(self.branch_var.get()) if hasattr(self, 'branch_var') else None,
                'softmin_temp': float(self.softmin_temp_var.get()) if hasattr(self, 'softmin_temp_var') else None,
                'verbose': bool(self.verbose_var.get()) if hasattr(self, 'verbose_var') else None,
                'tree_trace': bool(self.tree_trace_var.get()) if hasattr(self, 'tree_trace_var') else None,
                'format': self.format_var.get().strip() if hasattr(self, 'format_var') else None,
            })
            base['stockfish_ui'] = ui
            with open(self.CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(base, f, indent=2)
        except Exception:
            pass

    def _apply_loaded_prefs(self):
        ui = self._prefs.get('stockfish_ui', {}) if isinstance(self._prefs.get('stockfish_ui'), dict) else {}
        try:
            if ui.get('format') and hasattr(self, 'format_var'):
                self.format_var.set(ui.get('format'))
            if ui.get('depth') and hasattr(self, 'depth_var'):
                self.depth_var.set(int(ui.get('depth')))
            if ui.get('branching') and hasattr(self, 'branch_var'):
                self.branch_var.set(int(ui.get('branching')))
            if ui.get('softmin_temp') is not None and hasattr(self, 'softmin_temp_var'):
                self.softmin_temp_var.set(float(ui.get('softmin_temp')))
            if ui.get('verbose') is not None and hasattr(self, 'verbose_var'):
                self.verbose_var.set(bool(ui.get('verbose')))
            if ui.get('tree_trace') is not None and hasattr(self, 'tree_trace_var'):
                self.tree_trace_var.set(bool(ui.get('tree_trace')))
        except Exception:
            pass

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
        self.depth_var = tk.IntVar(value=self._prefs.get('stockfish_ui', {}).get('depth', 1))
        self.depth_spin = ttk.Spinbox(controls, from_=1, to=10, textvariable=self.depth_var, width=4, command=self._on_depth_changed)
        self.depth_spin.pack(side=tk.LEFT, padx=4)
        # Branching width spinner (top-K moves considered in depth projection)
        ttk.Label(controls, text="Branch:").pack(side=tk.LEFT, padx=(14, 2))
        self.branch_var = tk.IntVar(value=self._prefs.get('stockfish_ui', {}).get('branching', 3))
        self.branch_spin = ttk.Spinbox(controls, from_=1, to=10, textvariable=self.branch_var, width=4, command=self._on_branch_changed)
        self.branch_spin.pack(side=tk.LEFT, padx=4)
        # Softmin temperature spinner
        ttk.Label(controls, text="Softmin T:").pack(side=tk.LEFT, padx=(14,2))
        self.softmin_temp_var = tk.DoubleVar(value=self._prefs.get('stockfish_ui', {}).get('softmin_temp', 0.0))
        self.softmin_spin = ttk.Spinbox(controls, from_=0.0, to=5.0, increment=0.1, textvariable=self.softmin_temp_var, width=5, command=self._on_softmin_changed)
        self.softmin_spin.pack(side=tk.LEFT, padx=4)
        self.softmin_spin.bind('<Return>', lambda e: self._on_softmin_changed())
        # Verbose think checkbox
        self.verbose_var = tk.BooleanVar(value=self._prefs.get('stockfish_ui', {}).get('verbose', bool(int(os.environ.get('POKECHAD_THINK_DEBUG','0')))))
        self.verbose_chk = ttk.Checkbutton(controls, text="Verbose Think", variable=self.verbose_var, command=self._on_verbose_toggle)
        self.verbose_chk.pack(side=tk.LEFT, padx=(14,4))
        # Tree trace checkbox (detailed minimax branch logging)
        self.tree_trace_var = tk.BooleanVar(value=self._prefs.get('stockfish_ui', {}).get('tree_trace', bool(int(os.environ.get('POKECHAD_TREE_TRACE','0')))))
        self.tree_trace_chk = ttk.Checkbutton(controls, text="Tree Trace", variable=self.tree_trace_var, command=self._on_tree_trace_toggle)
        self.tree_trace_chk.pack(side=tk.LEFT, padx=(4,4))

        ttk.Button(controls, text="Ladder 1", command=lambda: self._submit(self._ladder(1))).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Challenge…", command=self._challenge_dialog).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Accept 1", command=lambda: self._submit(self._accept(1))).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Start Timer", command=lambda: self._submit(self._timer_all(True))).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Forfeit", command=lambda: self._submit(self._forfeit_all())).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Train Weights", command=self._train_weights).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Reload Weights", command=self._reload_weights).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Reset UI", command=self._reset_ui).pack(side=tk.LEFT, padx=(14,4))
        # Battle selector (multi-game support)
        ttk.Label(controls, text="Battle:").pack(side=tk.LEFT, padx=(14,2))
        self.battle_choice_var = tk.StringVar(value="")
        self.battle_combo = ttk.Combobox(controls, textvariable=self.battle_choice_var, values=[], width=24, state="readonly")
        self.battle_combo.pack(side=tk.LEFT, padx=4)
        self.battle_combo.bind('<<ComboboxSelected>>', lambda e: self._on_battle_select())

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
        # Added 'active' column
        cols = ("slot", "active", "species", "hp", "status", "boosts")
        tree = ttk.Treeview(frame, columns=cols, show="headings", height=12)
        headers = ("SLOT", "ACT", "SPECIES", "HP", "STATUS", "BOOSTS")
        widths = (60, 40, 160, 60, 80, 220)
        for c, h, w in zip(cols, headers, widths):
            tree.heading(c, text=h)
            tree.column(c, width=w, anchor=tk.W)
        tree.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        return tree

    def _make_cand_tree(self, parent, title: str) -> ttk.Treeview:
        frame = ttk.LabelFrame(parent, text=title); frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=6)
        # Added depth-adjusted score (DSCORE) and future projection (FUT) columns
        cols = ("move", "score", "dscore", "future", "exp_dmg", "acc", "eff", "first", "opp", "note")
        tree = ttk.Treeview(frame, columns=cols, show="headings", height=12)
        hdrs = ("MOVE", "SCORE", "DSCORE", "FUT", "EXP", "ACC", "EFF", "FIRST", "OPP", "WHY/NOTE")
        widths = (180, 70, 70, 60, 60, 50, 50, 60, 60, 240)
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

        # Apply verbose flag if set in UI/env
        try:
            if bool(self.verbose_var.get()) and getattr(self.player, 'engine', None):
                self.player.engine.set_verbose(True)
            # apply initial softmin temp
            if getattr(self.player, 'engine', None) and hasattr(self.player.engine, 'set_softmin_temperature'):
                try: self.player.engine.set_softmin_temperature(float(self.softmin_temp_var.get()))
                except Exception: pass
        except Exception:
            pass

        try:
            self.player.logger.addHandler(self.log_handler)
            self.player.logger.setLevel(logging.INFO)
        except Exception:
            pass
        try:
            # Also capture ThinkVerbose logger output
            tv = logging.getLogger('ThinkVerbose')
            tv.addHandler(self.log_handler)
            if tv.level > logging.INFO:
                tv.setLevel(logging.INFO)
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
        p = self.player
        if not p:
            return
        # Try native player method first
        try:
            m = getattr(p, "timer_all", None)
            if callable(m):
                await m(on)
                self._append_log(f"Called player.timer_all({on}).")
                return
        except Exception as e:
            self._append_log(f"player.timer_all({on}) failed: {e} — falling back to direct /timer command.")
        # Fallback: manually send /timer on|off to all battle rooms
        try:
            client = getattr(p, "ps_client", None) or getattr(p, "_client", None)
            if not client:
                raise RuntimeError("PSClient missing on player")
            battles = getattr(p, "battles", {}) or {}
            rooms: List[str] = []
            for key, battle in list(battles.items()):
                room_id = getattr(battle, "battle_tag", None) or getattr(battle, "room_id", None) or str(key)
                if room_id:
                    rooms.append(room_id)
            if not rooms:
                self._append_log("No active battle rooms found for /timer.")
                return
            cmd = "/timer on" if on else "/timer off"
            sent = 0
            for r in rooms:
                try:
                    await client.send_message(cmd, room=r)
                    sent += 1
                except Exception as e2:
                    self._append_log(f"Failed to send {cmd} to {r}: {e2}")
            self._append_log(f"Sent {cmd} to {sent} room(s).")
        except Exception as e:
            self._append_log(f"Timer fallback failed: {e}")

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
        self._save_prefs()

    def _on_branch_changed(self):
        if self.player:
            try:
                eng = getattr(self.player, "engine", None)
                if eng and hasattr(eng, "set_branching"):
                    eng.set_branching(int(self.branch_var.get()))
            except Exception:
                pass
        self._save_prefs()

    def _on_softmin_changed(self):
        if self.player:
            try:
                eng = getattr(self.player, 'engine', None)
                if eng and hasattr(eng, 'set_softmin_temperature'):
                    eng.set_softmin_temperature(float(self.softmin_temp_var.get()))
            except Exception:
                pass
        self._save_prefs()

    def _on_battle_select(self):
        # User manually picked a battle; update active battle id and refresh snapshot
        bid = self.battle_choice_var.get().strip()
        if not bid or not self.player: return
        self._active_battle_id = bid
        try:
            battle = self.player.battles.get(bid)
            if battle:
                from Data.poke_env_battle_environment import snapshot as snapshot_battle  # local import to avoid cycle
                try: self._latest_snapshot = snapshot_battle(battle)
                except Exception: pass
                self._refresh_teams(); self._refresh_thinking()
        except Exception: pass

    def _on_verbose_toggle(self):
        v = bool(self.verbose_var.get())
        try:
            if self.player and getattr(self.player, 'engine', None):
                try: self.player.engine.set_verbose(v)
                except Exception: pass
        except Exception: pass
        try: os.environ['POKECHAD_THINK_DEBUG'] = '1' if v else '0'
        except Exception: pass
        self._append_log(f"Verbose think {'ENABLED' if v else 'DISABLED'}")
        self._save_prefs()

    def _on_tree_trace_toggle(self):
        t = bool(self.tree_trace_var.get())
        try: os.environ['POKECHAD_TREE_TRACE'] = '1' if t else '0'
        except Exception: pass
        self._append_log(f"Tree trace {'ENABLED' if t else 'DISABLED'} (takes effect next think cycle)")
        self._save_prefs()

    def _reset_ui(self):
        """Clear UI state so next battle starts with a clean slate."""
        try:
            self._latest_think = {}
            self._latest_snapshot = {}
            self._last_fallback_turn = None
            self._last_real_think_turn = None
            # Clear trees
            for tree in (self.cand_tree, self.switch_tree, self.team_tree, self.opp_tree):
                try: self._reload_tree(tree)
                except Exception: pass
            # Clear logs text (keep telemetry file)
            try:
                self.logs_text.delete('1.0', tk.END)
            except Exception: pass
            self._append_log("[reset] UI state cleared; ready for next battle.")
        except Exception as e:
            self._append_log(f"[reset] Failed: {e}")

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
                    "switch_meta": think.get("switch_meta"),  # added for switch weight training
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
                raw_score = d.get("score")
                depth_score = d.get("score_depth", raw_score)
                future = d.get("future_proj")
                score = f"{float(raw_score):.2f}" if raw_score is not None else ""
                dscore = f"{float(depth_score):.2f}" if depth_score is not None else ""
                fut = f"{float(future):.2f}" if future is not None else ""
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
                self.cand_tree.insert("", tk.END, values=(move, score, dscore, fut, exp, acc, eff, first, opp, note))
            except Exception:
                try:
                    self.cand_tree.insert("", tk.END, values=(str(d), "", "", "", "", "", "", "", "", ""))
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
            active_flag = "*" if p.get("is_active") else ""
            self.team_tree.insert("", tk.END, values=(sid, active_flag, p.get("species"), hp_s, str(p.get("status") or ""), boosts))
        self._reload_tree(self.opp_tree)
        for sid, p in (snap.get("opp_team") or {}).items():
            boosts = pretty_boosts(p.get("boosts"))
            hp = p.get("hp_fraction")
            hp_s = f"{int(round(hp * 100))}%" if isinstance(hp, (int, float)) else ""
            active_flag = "*" if p.get("is_active") else ""
            self.opp_tree.insert("", tk.END, values=(sid, active_flag, p.get("species"), hp_s, str(p.get("status") or ""), boosts))

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
        battles = getattr(p, "battles", None)
        if not isinstance(battles, dict) or not battles:
            return None
        # Update selector list with current battles
        ids = list(battles.keys())
        try:
            self.battle_combo.configure(values=ids)
            # Preserve selection; if none selected or selection gone, pick first unfinished
            if not self._active_battle_id or self._active_battle_id not in ids:
                # prefer unfinished battle
                for bid, b in battles.items():
                    if not getattr(b, 'finished', False):
                        self._active_battle_id = bid; break
                else:
                    # fallback last id
                    self._active_battle_id = ids[-1]
                self.battle_choice_var.set(self._active_battle_id)
        except Exception: pass
        # Choose the active battle respecting user override
        battle = battles.get(self._active_battle_id)
        # If chosen battle finished and there is another unfinished, switch automatically
        if battle and getattr(battle,'finished', False):
            for bid,b in battles.items():
                if not getattr(b,'finished', False):
                    self._active_battle_id = bid
                    self.battle_choice_var.set(bid)
                    battle = b
                    break
        return battle

    def _poll_battle(self):
        try:
            if not self.winfo_exists(): return
            b = self._find_active_battle()
            if b is not None:
                # Detect battle finished transition
                finished = bool(getattr(b,'finished', False))
                bid = getattr(b,'battle_tag', getattr(b,'room_id', None))
                if finished and bid and bid not in self._finished_battles:
                    self._finished_battles.add(bid)
                    self._append_log(f"[battle] Finished: {bid} (winner={getattr(b,'won', None)})")
                if not finished:
                    try:
                        from Data.poke_env_battle_environment import snapshot as snapshot_battle
                        snap = snapshot_battle(b); self._latest_snapshot = snap
                    except Exception: snap = None
                    try: self._refresh_teams()
                    except Exception: pass
                    try:
                        turn = int(snap.get("turn")) if snap else None
                    except Exception:
                        turn = None
                    if turn is not None and turn != self._last_fallback_turn and turn != self._last_real_think_turn:
                        self._emit_fallback_think(b, snap); self._last_fallback_turn = turn
                else:
                    # Finished: if there is another unfinished battle, UI will swap next poll; otherwise allow new games without restart
                    pass
            else:
                # No battles active; clear selection state (leave past logs) but enable new games
                if self._active_battle_id is not None:
                    self._append_log("[battle] No active battles. Ready for a new game.")
                self._active_battle_id = None
                self.battle_choice_var.set("")
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
        self._save_prefs()
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
