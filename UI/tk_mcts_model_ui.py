from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import sys
import threading
import time
from typing import Any, Dict, List, Optional
from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox

# Ensure repo root is on sys.path so `Data` and `Models` can be imported
_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Force standalone neural MCTS - no legacy/enhanced complexity
from UI.standalone_player import StandaloneNeuralMCTSPlayer
from Data.poke_env_battle_environment import snapshot as snapshot_battle  # type: ignore
from poke_env.ps_client.account_configuration import AccountConfiguration  # type: ignore
from poke_env.ps_client.server_configuration import (  # type: ignore
    ShowdownServerConfiguration,
    LocalhostServerConfiguration,
)

# Reuse formats list from stockfish UI if available; else provide a local fallback
try:
    from UI.tk_stockfish_model_ui import KNOWN_FORMATS  # type: ignore
except Exception:
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
class MCTSWindow(tk.Toplevel):
    CONFIG_PATH = Path.home() / '.pokechad_ui_settings.json'
    def __init__(self, parent: tk.Tk, username: str, password: Optional[str], server_mode: str,
                 custom_ws: Optional[str], battle_format: str):
        super().__init__(parent)
        self.title("PokeCHAD — MCTS Model")
        self.geometry("1180x740")
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        # Load persisted prefs early
        self._prefs = self._load_prefs()

        self.username = username
        self.password = password
        self.server_mode = server_mode
        self.custom_ws = custom_ws
        self.battle_format = battle_format or self._prefs.get('mcts_ui', {}).get('format') or battle_format

        # Telemetry file
        os.makedirs("logs", exist_ok=True)
        self._telemetry_path = os.path.join("logs", f"telemetry_{os.getpid()}_mcts.jsonl")

        # Async loop thread
        self.loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self._loop_thread.start()

        # Player
        self.player: Optional[StandaloneNeuralMCTSPlayer] = None

        # Logging pane & handler
        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.log_handler = QueueLogHandler(self.log_queue)
        self.log_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        self.log_handler.setLevel(logging.DEBUG)  # Capture all MCTS debug logs
        
        # Simple file logging - just create the file path
        self.log_file_path = None
        try:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            self.log_file_path = log_dir / f"live_battle_{time.strftime('%Y%m%d_%H%M%S')}.log"
            
            # Create the file immediately with a header
            with open(self.log_file_path, 'w', encoding='utf-8') as f:
                f.write(f"=== PokeCHAD Live Battle Log - {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                f.flush()
            
            print(f"🗂️ Live battle logging to: {self.log_file_path}")
        except Exception as e:
            print(f"❌ Failed to setup file logging: {e}")
            self.log_file_path = None

        # UI State
        self._scheduled_tasks: List[str] = []
        self._latest_think: Dict[str, Any] = {}
        self._latest_snapshot: Dict[str, Any] = {}
        self._last_fallback_turn: Optional[int] = None
        self._last_real_think_turn: Optional[int] = None
        self._finished_battles: set[str] = set()
        self._active_battle_id: Optional[str] = None
        
        # Auto Play State
        self._auto_play_active = False
        self._auto_play_games_remaining = 0
        self._auto_play_games_completed = 0
        
        # Auto Timer State
        self._auto_timer_enabled = False
        self._auto_timer_delay = 30.0  # seconds
        self._last_turn_time = None
        self._auto_timer_task = None
        self._timer_already_triggered = False  # Prevent infinite timer triggering

        self._build_ui()
        self._apply_loaded_prefs()  # set widget values from prefs after UI built
        self._pump_logs()

        # Bootstrap: connect immediately
        self._submit(self._async_connect())

    def _submit(self, coro):
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
            ui = base.get('mcts_ui', {}) if isinstance(base.get('mcts_ui'), dict) else {}
            ui.update({
                'sims': int(self.sims_var.get()) if hasattr(self, 'sims_var') else None,
                'c_puct': float(self.cpuct_var.get()) if hasattr(self, 'cpuct_var') else None,
                'rollout': int(self.rollout_var.get()) if hasattr(self, 'rollout_var') else None,
                'time_limit': float(self.time_limit_var.get()) if hasattr(self, 'time_limit_var') else None,
                'verbose': bool(self.verbose_var.get()) if hasattr(self, 'verbose_var') else None,
                'llm_enabled': bool(self.llm_var.get()) if hasattr(self, 'llm_var') else None,
                'enhanced_mcts': bool(self.enhanced_mcts_var.get()) if hasattr(self, 'enhanced_mcts_var') else None,
                'collect_training_data': bool(self.collect_data_var.get()) if hasattr(self, 'collect_data_var') else None,
                'auto_timer_enabled': bool(self.auto_timer_var.get()) if hasattr(self, 'auto_timer_var') else None,
                'openai_api_key': getattr(self, '_openai_api_key', '') if hasattr(self, '_openai_api_key') else '',
                'format': self.format_var.get().strip() if hasattr(self, 'format_var') else None,
            })
            base['mcts_ui'] = ui
            with open(self.CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(base, f, indent=2)
        except Exception:
            pass

    def _apply_loaded_prefs(self):
        ui = self._prefs.get('mcts_ui', {}) if isinstance(self._prefs.get('mcts_ui'), dict) else {}
        try:
            if ui.get('format') and hasattr(self, 'format_var'):
                self.format_var.set(ui.get('format'))
            if ui.get('sims') and hasattr(self, 'sims_var'):
                self.sims_var.set(int(ui.get('sims')))
            if ui.get('c_puct') is not None and hasattr(self, 'cpuct_var'):
                self.cpuct_var.set(float(ui.get('c_puct')))
            if ui.get('rollout') and hasattr(self, 'rollout_var'):
                self.rollout_var.set(int(ui.get('rollout')))
            if ui.get('time_limit') is not None and hasattr(self, 'time_limit_var'):
                self.time_limit_var.set(float(ui.get('time_limit')))
            if ui.get('verbose') is not None and hasattr(self, 'verbose_var'):
                self.verbose_var.set(bool(ui.get('verbose')))
            if ui.get('llm_enabled') is not None and hasattr(self, 'llm_var'):
                self.llm_var.set(bool(ui.get('llm_enabled')))
            if ui.get('enhanced_mcts') is not None and hasattr(self, 'enhanced_mcts_var'):
                self.enhanced_mcts_var.set(bool(ui.get('enhanced_mcts')))
            if ui.get('collect_training_data') is not None and hasattr(self, 'collect_data_var'):
                self.collect_data_var.set(bool(ui.get('collect_training_data')))
            if ui.get('auto_timer_enabled') is not None and hasattr(self, 'auto_timer_var'):
                self.auto_timer_var.set(bool(ui.get('auto_timer_enabled')))
                self._auto_timer_enabled = bool(ui.get('auto_timer_enabled'))
            # Load API key
            self._openai_api_key = ui.get('openai_api_key', '')
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

        # Neural MCTS params (research-optimized)
        ttk.Label(controls, text="🚀 Neural Sims:").pack(side=tk.LEFT, padx=(14, 2))
        self.sims_var = tk.IntVar(value=self._prefs.get('mcts_ui', {}).get('sims', 150))  # Timeout-optimized: 150 (was 400)
        self.sims_spin = ttk.Spinbox(controls, from_=100, to=1600, increment=50, textvariable=self.sims_var, width=6, command=self._on_sims_changed)
        self.sims_spin.pack(side=tk.LEFT, padx=4)

        ttk.Label(controls, text="Exploration:").pack(side=tk.LEFT, padx=(14, 2))
        self.cpuct_var = tk.DoubleVar(value=self._prefs.get('mcts_ui', {}).get('c_puct', 1.25))  # Pokemon-optimized vs AlphaZero 1.0
        self.cpuct_spin = ttk.Spinbox(controls, from_=0.5, to=3.0, increment=0.05, textvariable=self.cpuct_var, width=5, command=self._on_cpuct_changed)
        self.cpuct_spin.pack(side=tk.LEFT, padx=4)

        ttk.Label(controls, text="Time(s):").pack(side=tk.LEFT, padx=(14,2))
        self.time_limit_var = tk.DoubleVar(value=self._prefs.get('mcts_ui', {}).get('time_limit', 30.0))  # Competitive time limit
        self.time_spin = ttk.Spinbox(controls, from_=5.0, to=120.0, increment=5.0, textvariable=self.time_limit_var, width=5, command=self._on_time_changed)
        self.time_spin.pack(side=tk.LEFT, padx=4)

        # Core AI settings
        self.evaluation_mode_var = tk.BooleanVar(value=self._prefs.get('mcts_ui', {}).get('evaluation_mode', True))  # ON by default for competitive
        self.evaluation_mode_chk = ttk.Checkbutton(controls, text="🎯 Competitive Mode", variable=self.evaluation_mode_var, command=self._on_evaluation_mode_toggle)
        self.evaluation_mode_chk.pack(side=tk.LEFT, padx=(14,4))
        
        self.parallel_mcts_var = tk.BooleanVar(value=self._prefs.get('mcts_ui', {}).get('parallel_mcts', True))  # ON for performance
        self.parallel_mcts_chk = ttk.Checkbutton(controls, text="⚡ Parallel", variable=self.parallel_mcts_var, command=self._on_parallel_mcts_toggle)
        self.parallel_mcts_chk.pack(side=tk.LEFT, padx=(4,4))
        
        # Training data collection
        self.collect_data_var = tk.BooleanVar(value=self._prefs.get('mcts_ui', {}).get('collect_training_data', True))
        self.collect_data_chk = ttk.Checkbutton(controls, text="📊 Collect Data", variable=self.collect_data_var, command=self._on_collect_data_toggle)
        self.collect_data_chk.pack(side=tk.LEFT, padx=(4,4))
        
        # Optional features
        self.verbose_var = tk.BooleanVar(value=self._prefs.get('mcts_ui', {}).get('verbose', False))
        self.verbose_chk = ttk.Checkbutton(controls, text="📝 Verbose", variable=self.verbose_var, command=self._on_verbose_toggle)
        self.verbose_chk.pack(side=tk.LEFT, padx=(4,4))
        
        self.auto_timer_var = tk.BooleanVar(value=self._prefs.get('mcts_ui', {}).get('auto_timer_enabled', False))
        self.auto_timer_chk = ttk.Checkbutton(controls, text="⏰ Auto Timer", variable=self.auto_timer_var, command=self._on_auto_timer_toggle)
        self.auto_timer_chk.pack(side=tk.LEFT, padx=(4,4))

        ttk.Button(controls, text="Ladder 1", command=lambda: self._submit(self._ladder(1))).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Challenge…", command=self._challenge_dialog).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Accept 1", command=lambda: self._submit(self._accept(1))).pack(side=tk.LEFT, padx=4)
        
        # Training controls row
        training_controls = ttk.Frame(dash)
        training_controls.pack(fill=tk.X, pady=2)
        
        ttk.Label(training_controls, text="Neural Network Training:", font=('TkDefaultFont', 9, 'bold')).pack(side=tk.LEFT, padx=(0,8))
        ttk.Button(training_controls, text="View Training Data", command=self._view_training_data).pack(side=tk.LEFT, padx=4)
        ttk.Button(training_controls, text="Train Neural Network", command=self._start_neural_training).pack(side=tk.LEFT, padx=4)
        ttk.Button(training_controls, text="Training Progress", command=self._view_training_progress).pack(side=tk.LEFT, padx=4)
        
        # Auto Play controls
        ttk.Label(training_controls, text="Auto Play:", font=('TkDefaultFont', 9)).pack(side=tk.LEFT, padx=(14,2))
        self.auto_play_games_var = tk.IntVar(value=self._prefs.get('mcts_ui', {}).get('auto_play_games', 100))
        self.auto_play_spin = ttk.Spinbox(training_controls, from_=1, to=10000, textvariable=self.auto_play_games_var, width=6, command=self._on_auto_play_games_changed)
        self.auto_play_spin.pack(side=tk.LEFT, padx=2)
        ttk.Label(training_controls, text="games").pack(side=tk.LEFT, padx=(2,4))
        self.auto_play_button = ttk.Button(training_controls, text="Start Auto Play", command=self._toggle_auto_play)
        self.auto_play_button.pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Start Timer", command=lambda: self._submit(self._timer_all(True))).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Forfeit", command=lambda: self._submit(self._forfeit_all())).pack(side=tk.LEFT, padx=4)
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
        cols = ("move", "score", "exp_dmg", "acc", "eff", "first", "opp", "note")
        tree = ttk.Treeview(frame, columns=cols, show="headings", height=12)
        hdrs = ("MOVE", "SCORE", "EXP", "ACC", "EFF", "FIRST", "OPP", "WHY/NOTE")
        widths = (180, 70, 60, 50, 50, 60, 60, 240)
        for c, h, w in zip(cols, hdrs, widths):
            tree.heading(c, text=h)
            tree.column(c, width=w, anchor=tk.W)
        tree.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        return tree

    def _make_switch_tree(self, parent, title: str) -> ttk.Treeview:
        frame = ttk.LabelFrame(parent, text=title); frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=6)
        cols = ("species", "score", "out", "in", "haz", "hp")
        tree = ttk.Treeview(frame, columns=cols, show="headings", height=8)
        headers = ("SPECIES", "SCORE", "OUT", "IN", "HAZ", "HP")
        widths = (140, 70, 60, 60, 50, 60)
        for c, h, w in zip(cols, headers, widths):
            tree.heading(c, text=h)
            tree.column(c, width=w, anchor=tk.W)
        tree.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        return tree

    # ---------- Connect ----------
    async def _async_connect(self):
        account = AccountConfiguration(self.username, self.password)
        if self.server_mode == "Showdown":
            server = ShowdownServerConfiguration
        elif self.server_mode == "Localhost":
            server = LocalhostServerConfiguration
        else:
            server = ShowdownServerConfiguration

        common = dict(
            account_configuration=account,
            server_configuration=server,
            battle_format=self.battle_format,
            log_level=logging.DEBUG,  # Enable DEBUG for MCTS logs
        )

        # Apply initial params if possible
        player = None; last_error = None
        sigs = [
            dict(on_think=self._on_think, simulations=int(self.sims_var.get()), start_listening=True),
            dict(on_think=self._on_think, simulations=int(self.sims_var.get())),
            dict(on_think=self._on_think),
            dict(),
        ]
        # Create pure standalone neural MCTS player
        standalone_config = {
            # 'device': removed - let standalone player auto-detect CUDA vs CPU
            'simulations': int(self.sims_var.get()),
            'collect_training_data': bool(self.collect_data_var.get()),
            'evaluation_mode': bool(self.evaluation_mode_var.get()),  # Pass mode selection to player
            'enable_parallel': bool(self.parallel_mcts_var.get()),  # Pass parallel setting to player
            'on_think': self._on_think,  # Add UI callback for battle state updates
            'ui_log_handler': self.log_handler  # Pass UI log handler for MCTS debug logs
        }
        
        try:
            player = StandaloneNeuralMCTSPlayer(**common, **standalone_config)
            self._append_log("[✅ STANDALONE] Pure neural MCTS player created")
            self._append_log(f"[✅ STANDALONE] Config: {int(self.sims_var.get())} simulations, training: {bool(self.collect_data_var.get())}")
            self._append_log("[✅ STANDALONE] Saves .pt tensor files to UI/training_data/standalone/")
        except Exception as e:
            self._append_log(f"[❌ STANDALONE] Failed to create player: {e}")
            player = StandaloneNeuralMCTSPlayer(**common)

        self.player = player
        return

        # Apply verbose flag and other params on engine
        try:
            eng = getattr(self.player, 'engine', None)
            if eng:
                try: eng.set_verbose(bool(self.verbose_var.get()))
                except Exception: pass
                try: eng.set_c_puct(float(self.cpuct_var.get()))
                except Exception: pass
                try: eng.set_rollout_depth(int(self.rollout_var.get()))
                except Exception: pass
                try: eng.set_time_limit(float(self.time_limit_var.get()))
                except Exception: pass
                # Configure LLM
                try: 
                    llm_enabled = bool(self.llm_var.get())
                    self._append_log(f"[DEBUG] LLM checkbox state: {llm_enabled}")
                    eng.set_llm_enabled(llm_enabled)
                    self._append_log(f"[DEBUG] LLM set_llm_enabled called")
                    if llm_enabled:
                        self._append_log(f"[ENGINE] LLM Enhancement enabled on new engine")
                    else:
                        self._append_log(f"[ENGINE] LLM Enhancement disabled on new engine")
                except Exception as e: 
                    self._append_log(f"[ENGINE ERROR] Failed to enable LLM: {e}")
                try: 
                    api_key = getattr(self, '_openai_api_key', '')
                    if api_key:
                        eng.set_openai_api_key(api_key)
                        self._append_log(f"[ENGINE] API key configured")
                except Exception as e: 
                    self._append_log(f"[ENGINE ERROR] Failed to set API key: {e}")
                
                # Configure Enhanced MCTS
                try: 
                    enhanced_enabled = bool(self.enhanced_mcts_var.get())
                    self._append_log(f"[DEBUG] Enhanced MCTS checkbox state: {enhanced_enabled}")
                    eng.set_enhancements_enabled(enhanced_enabled)
                    self._append_log(f"[DEBUG] Enhanced MCTS set_enhancements_enabled called")
                    if enhanced_enabled:
                        self._append_log(f"[ENGINE] Enhanced MCTS features enabled")
                        # Show status of enhancements
                        status = eng.get_enhancement_status()
                        available = status.get('available_enhancements', {})
                        for feature, avail in available.items():
                            status_emoji = "✅" if avail else "❌"
                            self._append_log(f"[ENGINE]   {status_emoji} {feature.replace('_', ' ').title()}")
                    else:
                        self._append_log(f"[ENGINE] Enhanced MCTS features disabled")
                except Exception as e: 
                    self._append_log(f"[ENGINE ERROR] Failed to configure Enhanced MCTS: {e}")
        except Exception as outer_e:
            self._append_log(f"[ENGINE ERROR] Engine configuration failed: {outer_e}")

        try:
            self.player.logger.addHandler(self.log_handler)
            if self.file_log_handler:
                self.player.logger.addHandler(self.file_log_handler)
            self.player.logger.setLevel(logging.DEBUG)  # Enable DEBUG for MCTS logs
        except Exception:
            pass
        try:
            tv = logging.getLogger('ThinkVerbose')
            tv.addHandler(self.log_handler)
            if self.file_log_handler:
                tv.addHandler(self.file_log_handler)
            if tv.level > logging.DEBUG:
                tv.setLevel(logging.DEBUG)  # Enable DEBUG for MCTS logs
        except Exception:
            pass
        try:
            # Add handler to root logger to capture LLM logs
            root_logger = logging.getLogger()
            root_logger.addHandler(self.log_handler)
            if root_logger.level > logging.DEBUG:
                root_logger.setLevel(logging.DEBUG)  # Enable DEBUG for MCTS logs
        except Exception:
            pass

        await self.player.ps_client.wait_for_login()
        self._append_log("Login confirmed. Ready.")
        
        # Log that we're ready
        if self.log_file_path:
            self._append_log("🗂️ File logging enabled for battle analysis")
        
        try:
            if hasattr(self, '_refresh_neural_status') and callable(self._refresh_neural_status):
                self._call_on_main(self._refresh_neural_status)
        except Exception:
            pass
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
        try:
            m = getattr(p, "timer_all", None)
            if callable(m):
                await m(on)
                self._append_log(f"Called player.timer_all({on}).")
                return
        except Exception as e:
            self._append_log(f"player.timer_all({on}) failed: {e} — falling back to direct /timer command.")
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

    def _on_sims_changed(self):
        if self.player:
            try:
                eng = getattr(self.player, "engine", None)
                if eng and hasattr(eng, "set_simulations"):
                    eng.set_simulations(int(self.sims_var.get()))
            except Exception:
                pass
        self._save_prefs()

    def _on_cpuct_changed(self):
        if self.player:
            try:
                eng = getattr(self.player, "engine", None)
                if eng and hasattr(eng, "set_c_puct"):
                    eng.set_c_puct(float(self.cpuct_var.get()))
            except Exception:
                pass
        self._save_prefs()

    def _on_rollout_changed(self):
        if self.player:
            try:
                eng = getattr(self.player, "engine", None)
                if eng and hasattr(eng, "set_rollout_depth"):
                    eng.set_rollout_depth(int(self.rollout_var.get()))
            except Exception:
                pass
        self._save_prefs()

    def _on_time_changed(self):
        if self.player:
            try:
                eng = getattr(self.player, "engine", None)
                if eng and hasattr(eng, "set_time_limit"):
                    eng.set_time_limit(float(self.time_limit_var.get()))
            except Exception:
                pass
        self._save_prefs()

    def _on_battle_select(self):
        bid = self.battle_choice_var.get().strip()
        if not bid or not self.player: return
        self._active_battle_id = bid
        try:
            battle = self.player.battles.get(bid)
            if battle:
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
        self._append_log(f"Verbose think {'ENABLED' if v else 'DISABLED'}")
        self._save_prefs()

    def _on_llm_toggle(self):
        llm_enabled = bool(self.llm_var.get())
        try:
            if self.player and getattr(self.player, 'engine', None):
                try: 
                    self.player.engine.set_llm_enabled(llm_enabled)
                    self._append_log(f"[OK] LLM setting applied to engine")
                except Exception as e: 
                    self._append_log(f"[ERROR] Failed to apply LLM to engine: {e}")
            else:
                self._append_log(f"[WARN] No player/engine available - LLM will be applied when battle starts")
        except Exception as e: 
            self._append_log(f"[ERROR] LLM toggle error: {e}")
        self._append_log(f"LLM Enhancement {'ENABLED' if llm_enabled else 'DISABLED'}")
        self._save_prefs()
    
    def _on_enhanced_mcts_toggle(self):
        enhanced_enabled = bool(self.enhanced_mcts_var.get())
        try:
            if self.player and getattr(self.player, 'engine', None):
                try: 
                    self.player.engine.set_enhancements_enabled(enhanced_enabled)
                    self._append_log(f"[OK] Enhanced MCTS setting applied to engine")
                    if enhanced_enabled:
                        # Show available enhancements
                        status = self.player.engine.get_enhancement_status()
                        available = status.get('available_enhancements', {})
                        self._append_log(f"[ENHANCED MCTS] Available features:")
                        for feature, avail in available.items():
                            status_emoji = "✅" if avail else "❌"
                            self._append_log(f"[ENHANCED MCTS]   {status_emoji} {feature.replace('_', ' ').title()}")
                except Exception as e: 
                    self._append_log(f"[ERROR] Failed to apply Enhanced MCTS to engine: {e}")
            else:
                self._append_log(f"[WARN] No player/engine available - Enhanced MCTS will be applied when battle starts")
        except Exception as e: 
            self._append_log(f"[ERROR] Enhanced MCTS toggle error: {e}")
        self._append_log(f"Enhanced MCTS {'ENABLED' if enhanced_enabled else 'DISABLED'}")
        self._save_prefs()
    
    def _on_collect_data_toggle(self):
        collect_enabled = bool(self.collect_data_var.get())
        try:
            if self.player and getattr(self.player, 'engine', None):
                try:
                    self.player.engine.collect_training_data = collect_enabled
                    # Ensure training collector is initialized before accessing it
                    if hasattr(self.player.engine, '_ensure_training_collector'):
                        self.player.engine._ensure_training_collector()
                    
                    if hasattr(self.player.engine, '_training_collector') and self.player.engine._training_collector:
                        self.player.engine._training_collector.enable_collection(collect_enabled)
                        self._append_log(f"[OK] Training data collection setting applied to engine")
                    else:
                        self._append_log(f"[WARN] Training collector not available")
                except Exception as e:
                    self._append_log(f"[ERROR] Failed to apply training data collection: {e}")
            else:
                self._append_log(f"[WARN] No player/engine available - setting will be applied when battle starts")
        except Exception as e:
            self._append_log(f"[ERROR] Training data collection toggle error: {e}")
        self._append_log(f"Training Data Collection {'ENABLED' if collect_enabled else 'DISABLED'}")
        self._save_prefs()
    
    def _on_evaluation_mode_toggle(self):
        """Handle evaluation mode toggle"""
        evaluation_mode = bool(self.evaluation_mode_var.get())
        
        if evaluation_mode:
            self._append_log("[MODE] Evaluation Mode ENABLED - deterministic decisions, no exploration")
        else:
            self._append_log("[MODE] Self-Play Mode ENABLED - temperature-based exploration, training behavior")
        
        # Save preference
        if 'mcts_ui' not in self._prefs.data:
            self._prefs.data['mcts_ui'] = {}
        self._prefs.data['mcts_ui']['evaluation_mode'] = evaluation_mode
        self._prefs.save()
    
    def _on_parallel_mcts_toggle(self):
        """Handle parallel MCTS toggle"""
        parallel_enabled = bool(self.parallel_mcts_var.get())
        
        if parallel_enabled:
            self._append_log("[PARALLEL] Parallel MCTS ENABLED - 2-4x performance boost with multiple workers")
        else:
            self._append_log("[PARALLEL] Parallel MCTS DISABLED - sequential search only")
        
        # Save preference
        if 'mcts_ui' not in self._prefs.data:
            self._prefs.data['mcts_ui'] = {}
        self._prefs.data['mcts_ui']['parallel_mcts'] = parallel_enabled
        self._prefs.save()
    
    def _on_auto_timer_toggle(self):
        auto_timer_enabled = bool(self.auto_timer_var.get())
        self._auto_timer_enabled = auto_timer_enabled
        
        if auto_timer_enabled:
            self._append_log("[AUTO TIMER] Auto Timer ENABLED - will start timer if opponent inactive for 30 seconds")
            # Reset the turn timer when enabled
            self._last_turn_time = time.time()
        else:
            self._append_log("[AUTO TIMER] Auto Timer DISABLED")
            # Cancel any pending auto timer
            if self._auto_timer_task:
                try:
                    self.after_cancel(self._auto_timer_task)
                    self._auto_timer_task = None
                except Exception:
                    pass
        
        self._save_prefs()
    
    def _trigger_auto_timer(self):
        """Automatically start the timer when opponent is inactive"""
        try:
            self._auto_timer_task = None  # Clear the task
            self._timer_already_triggered = True  # Prevent re-triggering
            
            if not self._auto_timer_enabled:
                return
            
            # Check if we still have an active battle
            if not self.player or not self._active_battle_id:
                return
            
            # Start the timer
            self._append_log("[AUTO TIMER] 30 seconds of inactivity detected - starting timer")
            self._submit(self._timer_all(True))
            
        except Exception as e:
            self._append_log(f"[AUTO TIMER ERROR] Failed to trigger auto timer: {e}")

    def _configure_api_key(self):
        """Open dialog to configure OpenAI API key."""
        dialog = tk.Toplevel(self)
        dialog.title("OpenAI API Configuration")
        dialog.geometry("500x300")
        dialog.transient(self)
        dialog.grab_set()
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (500 // 2)
        y = (dialog.winfo_screenheight() // 2) - (300 // 2)
        dialog.geometry(f"500x300+{x}+{y}")
        
        # Instructions
        instructions = tk.Text(dialog, height=8, wrap=tk.WORD)
        instructions.pack(fill=tk.X, padx=10, pady=10)
        instructions.insert(tk.END, """OpenAI API Key Configuration

To use real GPT models for strategic analysis:

1. Get an API key from https://platform.openai.com/api-keys
2. Create a new secret key if you don't have one
3. Copy and paste it below
4. The key will be saved locally and used for LLM analysis

Note: API usage will incur costs based on OpenAI pricing.
Without an API key, the system uses a mock LLM for testing.""")
        instructions.config(state=tk.DISABLED)
        
        # API key input
        key_frame = ttk.Frame(dialog)
        key_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(key_frame, text="API Key:").pack(side=tk.LEFT)
        api_key_var = tk.StringVar(value=getattr(self, '_openai_api_key', ''))
        api_key_entry = ttk.Entry(key_frame, textvariable=api_key_var, show="*", width=50)
        api_key_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Status
        status_var = tk.StringVar()
        status_label = ttk.Label(dialog, textvariable=status_var)
        status_label.pack(pady=5)
        
        # Current status
        current_key = getattr(self, '_openai_api_key', '')
        if current_key:
            masked_key = current_key[:8] + "..." + current_key[-4:] if len(current_key) > 12 else "***"
            status_var.set(f"Current: {masked_key} (GPT-5 Enabled)")
        else:
            status_var.set("Current: None (Using Mock LLM)")
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        def save_api_key():
            new_key = api_key_var.get().strip()
            self._openai_api_key = new_key
            
            # Update the engine
            try:
                if self.player and getattr(self.player, 'engine', None):
                    self.player.engine.set_openai_api_key(new_key)
                
                if new_key:
                    masked_key = new_key[:8] + "..." + new_key[-4:] if len(new_key) > 12 else "***"
                    self._append_log(f"OpenAI API Key configured: {masked_key} - GPT-5 Enabled")
                    status_var.set(f"Updated: {masked_key} (GPT-5 Enabled)")
                else:
                    self._append_log("OpenAI API Key cleared - using Mock LLM")
                    status_var.set("Updated: None (Using Mock LLM)")
                    
                self._save_prefs()
            except Exception as e:
                self._append_log(f"Error configuring API key: {e}")
        
        def test_api_key():
            test_key = api_key_var.get().strip()
            if not test_key:
                status_var.set("Error: No API key provided")
                return
                
            status_var.set("Testing API key...")
            dialog.update()
            
            try:
                # Test the API key with GPT-5 first, fallback to GPT-4
                import openai
                client = openai.OpenAI(api_key=test_key)
                
                # Try GPT-5 first
                try:
                    response = client.chat.completions.create(
                        model="gpt-5",
                        messages=[{"role": "user", "content": "Test"}],
                        max_tokens=5
                    )
                    status_var.set("✓ API Key Valid (GPT-5 Available)")
                except Exception:
                    # Fallback to GPT-4
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": "Test"}],
                        max_tokens=5
                    )
                    status_var.set("✓ API Key Valid (GPT-4 Available)")
                    
            except Exception as e:
                status_var.set(f"✗ API Key Invalid: {str(e)[:50]}...")
        
        ttk.Button(button_frame, text="Test Key", command=test_api_key).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Save", command=save_api_key).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT)
        
        # Focus on entry
        api_key_entry.focus_set()

    def _reset_ui(self):
        try:
            self._latest_think = {}
            self._latest_snapshot = {}
            self._last_fallback_turn = None
            self._last_real_think_turn = None
            for tree in (self.cand_tree, self.switch_tree, self.team_tree, self.opp_tree):
                try: self._reload_tree(tree)
                except Exception: pass
            try:
                self.logs_text.delete('1.0', tk.END)
            except Exception: pass
            self._append_log("[reset] UI state cleared; ready for next battle.")
        except Exception as e:
            self._append_log(f"[reset] Failed: {e}")

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
        # write JSONL telemetry
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
                    f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            self._append_log(f"telemetry write failed: {e}")

        self._call_on_main(self._refresh_thinking)
        self._call_on_main(self._refresh_teams)
        # Optional: refresh neural status panel if present (used by Modern UI)
        try:
            if hasattr(self, '_refresh_neural_status') and callable(self._refresh_neural_status):
                self._call_on_main(self._refresh_neural_status)
        except Exception:
            pass

    def _refresh_thinking(self):
        if not self.winfo_exists(): return
        # Candidates
        self._reload_tree(self.cand_tree)
        for d in self._latest_think.get("candidates", []):
            try:
                move = d.get("name") or d.get("id") or d.get("move") or d.get("move_id")
                score = d.get("score"); score_s = f"{float(score):.2f}" if score is not None else ""
                exp = d.get("exp_dmg") if d.get("exp_dmg") is not None else d.get("expected")
                exp_s = f"{float(exp):.2f}" if exp is not None else ""
                acc = d.get("acc"); acc_s = f"{float(acc):.2f}" if acc is not None else ""
                eff = d.get("effectiveness") if d.get("effectiveness") is not None else d.get("eff")
                eff_s = f"{float(eff):.2f}" if eff is not None else ""
                first = d.get("first_prob"); first_s = f"{float(first):.2f}" if first is not None else ""
                opp = d.get("opp_counter_ev"); opp_s = f"{float(opp):.2f}" if opp is not None else ""
                note = d.get("why_blocked") or d.get("note") or d.get("why") or ""
                self.cand_tree.insert("", tk.END, values=(move, score_s, exp_s, acc_s, eff_s, first_s, opp_s, note))
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
                out = d.get("outgoing_frac"); out_s = f"{float(out):.2f}" if out is not None else ""
                incoming = d.get("incoming_on_switch"); in_s = f"{float(incoming):.2f}" if incoming is not None else ""
                haz = d.get("hazards_frac"); haz_s = f"{float(haz):.2f}" if haz is not None else ""
                hp = d.get("hp_fraction"); hp_s = f"{int(round(float(hp) * 100))}%" if isinstance(hp, (int, float)) else ""
                self.switch_tree.insert("", tk.END, values=(species, score, out_s, in_s, haz_s, hp_s))
            except Exception:
                try:
                    species = d.get("species") or str(d)
                    score = f"{float(d.get('score', 0)):.2f}" if d.get('score') is not None else "0.00"
                    self.switch_tree.insert("", tk.END, values=(species, score, "", "", "", ""))
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
        ids = list(battles.keys())
        try:
            self.battle_combo.configure(values=ids)
            if not self._active_battle_id or self._active_battle_id not in ids:
                old_battle_id = self._active_battle_id
                for bid, b in battles.items():
                    if not getattr(b, 'finished', False):
                        self._active_battle_id = bid; break
                else:
                    self._active_battle_id = ids[-1]
                self.battle_choice_var.set(self._active_battle_id)
                
                # Initialize auto timer for new battle
                if old_battle_id != self._active_battle_id and self._auto_timer_enabled:
                    self._last_turn_time = time.time()
                    if self._auto_timer_task:
                        try:
                            self.after_cancel(self._auto_timer_task)
                            self._auto_timer_task = None
                        except Exception:
                            pass
        except Exception: pass
        battle = battles.get(self._active_battle_id)
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
                finished = bool(getattr(b,'finished', False))
                bid = getattr(b,'battle_tag', getattr(b,'room_id', None))
                if finished and bid and bid not in self._finished_battles:
                    self._finished_battles.add(bid)
                    winner = getattr(b,'won', None)
                    self._append_log(f"[battle] Finished: {bid} (winner={winner})")
                    
                    # End training data collection for this battle
                    if (hasattr(self.player, 'engine') and hasattr(self.player.engine, '_training_collector') 
                        and self.player.engine._training_collector):
                        try:
                            # Determine game result from battle outcome
                            if winner is True:
                                game_result = 1.0  # Win
                                final_score = "1-0"
                            elif winner is False:
                                game_result = -1.0  # Loss (FIXED: was 0.0)
                                final_score = "0-1"
                            else:
                                # Handle timeout/draw cases by checking additional battle properties
                                if hasattr(b, 'winner') and hasattr(b, 'player_username'):
                                    # Check if our player won by timeout
                                    if getattr(b, 'winner', None) == getattr(b, 'player_username', None):
                                        game_result = 1.0  # Timeout win
                                        final_score = "1-0 timeout"
                                    elif getattr(b, 'winner', None) and getattr(b, 'winner', None) != getattr(b, 'player_username', None):
                                        game_result = -1.0  # Timeout loss (FIXED: was 0.0)
                                        final_score = "0-1 timeout"
                                    else:
                                        game_result = 0.5  # Draw
                                        final_score = "0.5-0.5"
                                else:
                                    # Fallback for unclear results
                                    game_result = 0.5  # Draw
                                    final_score = "0.5-0.5 unclear"
                            
                            self.player.engine._training_collector.end_game_collection(game_result, final_score)
                            self._append_log(f"[Training] Ended game collection: result={game_result:.1f}, score={final_score}")
                            
                        except Exception as e:
                            self._append_log(f"[Training] Error ending game collection: {e}")
                    
                    # Handle auto-play game completion
                    if self._auto_play_active:
                        self._auto_play_games_completed += 1
                        self._auto_play_games_remaining -= 1
                        self._append_log(f"[AUTO PLAY] Game completed ({self._auto_play_games_completed} done, {self._auto_play_games_remaining} remaining)")
                        
                        # Queue next game after a short delay
                        self.after(2000, self._queue_next_auto_play_game)
                if not finished:
                    try:
                        snap = snapshot_battle(b); self._latest_snapshot = snap
                    except Exception: snap = None
                    try: self._refresh_teams()
                    except Exception: pass
                    # Keep neural status fresh if present
                    try:
                        if hasattr(self, '_refresh_neural_status') and callable(self._refresh_neural_status):
                            self._refresh_neural_status()
                    except Exception:
                        pass
                    try:
                        turn = int(snap.get("turn")) if snap else None
                    except Exception:
                        turn = None
                    
                    # Auto Timer Logic
                    if self._auto_timer_enabled and turn is not None:
                        # Check if turn has changed
                        if turn != self._last_fallback_turn and turn != self._last_real_think_turn:
                            # Turn changed - reset timer
                            self._last_turn_time = time.time()
                            self._timer_already_triggered = False  # Reset timer trigger flag on new turn
                            if self._auto_timer_task:
                                try:
                                    self.after_cancel(self._auto_timer_task)
                                    self._auto_timer_task = None
                                except Exception:
                                    pass
                        elif self._last_turn_time is not None and not self._timer_already_triggered:
                            # Check if 30 seconds have passed without turn change
                            time_since_last_turn = time.time() - self._last_turn_time
                            if time_since_last_turn >= self._auto_timer_delay and self._auto_timer_task is None:
                                # Schedule auto timer activation (only once)
                                self._auto_timer_task = self.after(1000, self._trigger_auto_timer)
                            elif time_since_last_turn >= 25.0 and self._auto_timer_task is None:
                                # Show 5-second warning
                                remaining = int(self._auto_timer_delay - time_since_last_turn)
                                if remaining > 0:
                                    self._append_log(f"[AUTO TIMER] Auto-timer in {remaining} seconds...")
                    
                    if turn is not None and turn != self._last_fallback_turn and turn != self._last_real_think_turn:
                        self._emit_fallback_think(b, snap); self._last_fallback_turn = turn
                else:
                    pass
            else:
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
            # Try to get active species for both sides to compute type effectiveness
            try:
                my_species = getattr(getattr(battle, "active_pokemon", None), "species", None)
            except Exception:
                my_species = None
            try:
                opp_species = getattr(getattr(battle, "opponent_active_pokemon", None), "species", None) or getattr(getattr(battle, "opponent_active_pokemon", None), "base_species", None)
            except Exception:
                opp_species = None
            # Lazy import for type effectiveness calculator
            try:
                from utils.type_effectiveness import get as _get_tc
                _tc = _get_tc()
            except Exception:
                _tc = None
            for m in (getattr(battle, "available_moves", None) or []):
                try:
                    name = getattr(m, "name", None) or getattr(m, "id", None) or str(m)
                    mid = getattr(m, "id", None) or getattr(m, "move_id", None) or str(name).lower().replace(" ", "")
                    bp = getattr(m, "base_power", None) or getattr(m, "basePower", None) or 0
                    acc = getattr(m, "accuracy", None)
                    if acc is True: acc_val = 1.0
                    elif isinstance(acc, (int, float)): acc_val = float(acc) / (100.0 if acc > 1 else 1.0)
                    else: acc_val = 1.0
                    expected = float(bp or 0) * float(acc_val)
                    eff_val = 1.0
                    if _tc is not None and mid:
                        try:
                            eff_val, _ = _tc.effectiveness(str(mid), my_species, opp_species)
                        except Exception:
                            eff_val = 1.0
                    cands.append({"name": name, "score": expected, "exp_dmg": expected, "acc": acc_val, "effectiveness": float(eff_val), "note": "synthetic"})
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
            
            # Also write to file if available
            if hasattr(self, 'log_file_path') and self.log_file_path:
                try:
                    with open(self.log_file_path, 'a', encoding='utf-8') as f:
                        f.write(f"{msg}\n")
                        f.flush()
                except Exception:
                    pass
        except Exception:
            pass

    def _pump_logs(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self._append_log(msg)
        except queue.Empty:
            pass
        
        # File handler flushing is handled by the combined handler automatically
        
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
        
        # Clean up auto timer
        if self._auto_timer_task:
            try:
                self.after_cancel(self._auto_timer_task)
                self._auto_timer_task = None
            except Exception:
                pass
        
        # Clean up file logging
        if self.log_file_path:
            try:
                with open(self.log_file_path, 'a', encoding='utf-8') as f:
                    f.write(f"=== Session ended - {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                print(f"📁 Live battle log saved: {self.log_file_path}")
            except Exception as e:
                print(f"Error closing log file: {e}")
        try:
            if self.loop.is_running():
                self.loop.call_soon_threadsafe(self.loop.stop)
        except Exception:
            pass
        self.destroy()
    
    def _view_training_data(self):
        """Show training data statistics"""
        try:
            print(f"[DEBUG] _view_training_data called")
            print(f"[DEBUG] self.player exists: {self.player is not None}")
            if self.player:
                print(f"[DEBUG] engine exists: {hasattr(self.player, 'engine')}")
                if hasattr(self.player, 'engine'):
                    print(f"[DEBUG] _training_collector attr exists: {hasattr(self.player.engine, '_training_collector')}")
                    if hasattr(self.player.engine, '_training_collector'):
                        print(f"[DEBUG] _training_collector is not None: {self.player.engine._training_collector is not None}")
            
            if self.player and getattr(self.player, 'engine', None):
                # Ensure training collector is initialized before accessing it
                if hasattr(self.player.engine, '_ensure_training_collector'):
                    self.player.engine._ensure_training_collector()
                
                if hasattr(self.player.engine, '_training_collector') and self.player.engine._training_collector:
                    collector = self.player.engine._training_collector
                    stats = collector.get_collection_stats()
                
                # Create info dialog
                dialog = tk.Toplevel(self)
                dialog.title("Training Data Statistics")
                dialog.geometry("600x400")
                dialog.resizable(True, True)
                
                # Create text widget with scrollbar
                text_frame = ttk.Frame(dialog)
                text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                
                text_widget = tk.Text(text_frame, wrap=tk.WORD, font=('Consolas', 10))
                scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
                text_widget.configure(yscrollcommand=scrollbar.set)
                
                text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                
                # Display statistics
                info_text = f"""Training Data Collection Statistics

Collection Status: {'ENABLED' if stats['collection_enabled'] else 'DISABLED'}
Data Directory: {stats['data_directory']}

Game Data:
  Games on Disk: {stats['games_on_disk']}
  Total Positions Collected: {stats['positions_collected']}
  Total Data Size: {stats['total_data_size_mb']:.2f} MB

Current Session:
  Current Game Active: {'YES' if stats['current_game_active'] else 'NO'}
  Positions in Current Game: {stats['current_positions']}

Training Requirements:
  Minimum Games for Training: 10
  Ready for Training: {'YES' if stats['games_on_disk'] >= 10 else 'NO'}

Data Collection:
- Each battle automatically collects position data
- Includes MCTS scores, LLM decisions, and game outcomes
- Data is saved after each completed game
- Used to train neural networks for better position evaluation
"""
                text_widget.insert(tk.END, info_text)
                text_widget.config(state=tk.DISABLED)
                
                # Close button
                ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=5)
                
            else:
                print("[DEBUG] Collector not available through player, trying direct access")
                # Fallback: try to create collector directly
                try:
                    from Models.training_data_collector import get_training_collector
                    collector = get_training_collector()
                    stats = collector.get_collection_stats()
                    
                    # Create info dialog
                    dialog = tk.Toplevel(self)
                    dialog.title("Training Data Statistics")
                    dialog.geometry("600x400")
                    dialog.resizable(True, True)
                    
                    # Create text widget with scrollbar
                    text_frame = ttk.Frame(dialog)
                    text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                    
                    text_widget = tk.Text(text_frame, wrap=tk.WORD, font=('Consolas', 10))
                    scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
                    text_widget.configure(yscrollcommand=scrollbar.set)
                    
                    text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                    
                    # Display statistics
                    info_text = f"""Training Data Collection Statistics

Collection Status: {'ENABLED' if stats['collection_enabled'] else 'DISABLED'}
Data Directory: {stats['data_directory']}

Game Data:
  Games on Disk: {stats['games_on_disk']}
  Total Positions Collected: {stats['positions_collected']}
  Total Data Size: {stats['total_data_size_mb']:.2f} MB

Current Session:
  Current Game Active: {'YES' if stats['current_game_active'] else 'NO'}
  Positions in Current Game: {stats['current_positions']}

Training Requirements:
  Minimum Games for Training: 10
  Ready for Training: {'YES' if stats['games_on_disk'] >= 10 else 'NO'}

Data Collection:
- Each battle automatically collects position data
- Includes MCTS scores, LLM decisions, and game outcomes  
- Data is saved after each completed game
- Used to train neural networks for better position evaluation
"""
                    
                    text_widget.insert(tk.END, info_text)
                    text_widget.config(state=tk.DISABLED)
                    
                    # Close button
                    ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=5)
                    
                except Exception as fallback_error:
                    print(f"[DEBUG] Fallback also failed: {fallback_error}")
                    self._append_log(f"[ERROR] Training data collector not available: {fallback_error}")
                
        except Exception as e:
            self._append_log(f"[ERROR] Failed to view training data: {e}")
    
    def _start_neural_training(self):
        """Start neural network training"""
        try:
            # Check if training data is available
            if self.player and getattr(self.player, 'engine', None):
                # Ensure training collector is initialized before accessing it
                if hasattr(self.player.engine, '_ensure_training_collector'):
                    self.player.engine._ensure_training_collector()
                
                if hasattr(self.player.engine, '_training_collector') and self.player.engine._training_collector:
                    collector = self.player.engine._training_collector
                stats = collector.get_collection_stats()
                
                if stats['games_on_disk'] < 10:
                    self._append_log(f"[ERROR] Need at least 10 games for training, have {stats['games_on_disk']}")
                    self._append_log(f"[INFO] Play more battles to collect training data")
                    return
                
                # Start training in a simple way for now
                self._append_log("[TRAINING] Starting neural network training...")
                self._append_log("[TRAINING] This may take several minutes...")
                
                def run_training():
                    def safe_log(msg):
                        """Thread-safe logging that schedules log calls on main thread"""
                        try:
                            self.after(0, lambda: self._append_log(msg))
                        except Exception:
                            print(f"[TRAINING DEBUG] {msg}")  # Fallback to console
                    
                    try:
                        safe_log("[TRAINING] Importing training modules...")
                        from Models.neural_network_trainer import NeuralNetworkTrainer, TrainingConfig
                        safe_log("[TRAINING] Modules imported successfully")
                        
                        safe_log("[TRAINING] Creating training configuration...")
                        config = TrainingConfig(epochs=5, batch_size=16, device='cpu')
                        safe_log(f"[TRAINING] Config: epochs={config.epochs}, batch_size={config.batch_size}, device={config.device}")
                        
                        safe_log("[TRAINING] Initializing neural network trainer...")
                        trainer = NeuralNetworkTrainer(config)
                        safe_log("[TRAINING] Trainer initialized successfully")
                        
                        safe_log("[TRAINING] Starting training process...")
                        result = trainer.train()  # Use all available training data
                        safe_log("[TRAINING] Training process returned")
                        
                        if 'error' in result:
                            safe_log(f"[TRAINING ERROR] {result['error']}")
                        else:
                            safe_log(f"[TRAINING] Training completed successfully!")
                            safe_log(f"[TRAINING] Epochs: {result['epochs_completed']}")
                            safe_log(f"[TRAINING] Games used: {result.get('games_used', 'unknown')}")
                            safe_log(f"[TRAINING] Positions used: {result.get('positions_used', 'unknown')}")
                            safe_log(f"[TRAINING] Models saved to Models/ directory")
                            
                    except ImportError as e:
                        safe_log(f"[TRAINING ERROR] Import failed: {e}")
                        import traceback
                        safe_log(f"[TRAINING ERROR] Traceback: {traceback.format_exc()}")
                    except Exception as e:
                        safe_log(f"[TRAINING ERROR] Unexpected error: {e}")
                        import traceback
                        safe_log(f"[TRAINING ERROR] Traceback: {traceback.format_exc()}")
                    finally:
                        safe_log("[TRAINING] Training thread finished")
                
                # Run in thread to avoid blocking UI
                import threading
                training_thread = threading.Thread(target=run_training)
                training_thread.daemon = True
                training_thread.start()
                
            else:
                self._append_log("[ERROR] Training data collector not available")
                
        except Exception as e:
            self._append_log(f"[ERROR] Failed to start training: {e}")
    
    def _view_training_progress(self):
        """View training progress and history"""
        try:
            from pathlib import Path
            import json
            import time
            
            # First check for live progress file (training currently active)
            live_progress_file = Path("Models/training_progress.json")
            history_file = Path("Models/training_history.json")
            
            if live_progress_file.exists():
                # Training is currently active - show live progress
                try:
                    with open(live_progress_file, 'r') as f:
                        progress = json.load(f)
                    
                    if progress.get('training_active', False):
                        current_epoch = progress.get('current_epoch', 0)
                        total_epochs = progress.get('total_epochs', 0)
                        train_val_loss = progress.get('train_value_loss', 0)
                        train_pol_loss = progress.get('train_policy_loss', 0)
                        val_val_loss = progress.get('val_value_loss', 0)
                        val_pol_loss = progress.get('val_policy_loss', 0)
                        timestamp = progress.get('timestamp', 0)
                        
                        # Calculate time since last update
                        time_since_update = time.time() - timestamp
                        
                        # Create progress dialog
                        dialog = tk.Toplevel(self)
                        dialog.title("Neural Network Training Progress")
                        dialog.geometry("500x400")
                        dialog.resizable(True, True)
                        
                        # Create text widget with scrollbar
                        text_frame = ttk.Frame(dialog)
                        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                        
                        text_widget = tk.Text(text_frame, wrap=tk.WORD, font=('Consolas', 10))
                        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
                        text_widget.configure(yscrollcommand=scrollbar.set)
                        
                        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                        
                        # Progress content
                        progress_text = f"""🚀 Neural Network Training In Progress

Training Status: ACTIVE 
Epoch Progress: {current_epoch}/{total_epochs} ({(current_epoch/total_epochs*100):.1f}%)
Last Update: {time_since_update:.1f} seconds ago

Current Training Metrics:
  Value Loss (Train): {train_val_loss:.4f}
  Policy Loss (Train): {train_pol_loss:.4f}
  Value Loss (Validation): {val_val_loss:.4f}
  Policy Loss (Validation): {val_pol_loss:.4f}

Training Info:
- Training neural networks on collected battle data
- Lower loss values indicate better learning progress
- Each epoch processes all available training data
- Models are saved automatically after training

Progress updates every 10 batches during training.
This window shows the latest available progress."""

                        text_widget.insert(tk.END, progress_text)
                        text_widget.config(state=tk.DISABLED)
                        
                        # Refresh button to update progress
                        button_frame = ttk.Frame(dialog)
                        button_frame.pack(fill=tk.X, padx=10, pady=5)
                        
                        def refresh_progress():
                            self._view_training_progress()
                            dialog.destroy()
                        
                        ttk.Button(button_frame, text="Refresh", command=refresh_progress).pack(side=tk.LEFT, padx=5)
                        ttk.Button(button_frame, text="Close", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)
                        
                        return
                        
                except Exception as live_error:
                    self._append_log(f"[ERROR] Failed to read live progress: {live_error}")
            
            # No live training - check for historical training data
            if history_file.exists():
                # Load and display training history
                with open(history_file, 'r') as f:
                    history = json.load(f)
                
                # Create history dialog
                dialog = tk.Toplevel(self)
                dialog.title("Neural Network Training History")
                dialog.geometry("500x400")
                dialog.resizable(True, True)
                
                # Create text widget with scrollbar
                text_frame = ttk.Frame(dialog)
                text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                
                text_widget = tk.Text(text_frame, wrap=tk.WORD, font=('Consolas', 10))
                scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
                text_widget.configure(yscrollcommand=scrollbar.set)
                
                text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                
                # History content
                epochs_completed = history.get('epochs_completed', 0)
                value_losses = history.get('value_losses', [])
                policy_losses = history.get('policy_losses', [])
                validation_losses = history.get('validation_losses', [])
                
                history_text = f"""📊 Neural Network Training History

Training Status: COMPLETED
Epochs Completed: {epochs_completed}

Final Training Results:"""
                
                if value_losses:
                    final_val_loss = value_losses[-1]
                    history_text += f"\n  Final Value Loss: {final_val_loss:.4f}"
                
                if policy_losses:
                    final_pol_loss = policy_losses[-1]
                    history_text += f"\n  Final Policy Loss: {final_pol_loss:.4f}"
                
                if validation_losses:
                    final_validation_loss = validation_losses[-1]
                    history_text += f"\n  Final Validation Loss: {final_validation_loss:.4f}"
                
                # Show loss progression (last 5 epochs)
                if len(value_losses) > 1:
                    history_text += f"\n\nLoss Progression (Last 5 Epochs):"
                    start_idx = max(0, len(value_losses) - 5)
                    for i in range(start_idx, len(value_losses)):
                        epoch_num = i + 1
                        val_loss = value_losses[i] if i < len(value_losses) else 0
                        pol_loss = policy_losses[i] if i < len(policy_losses) else 0
                        history_text += f"\n  Epoch {epoch_num}: Value={val_loss:.4f}, Policy={pol_loss:.4f}"
                
                history_text += f"\n\nTraining completed successfully. Neural network models are ready for use."
                
                text_widget.insert(tk.END, history_text)
                text_widget.config(state=tk.DISABLED)
                
                # Close button
                ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=5)
                
            else:
                self._append_log("[INFO] No training progress found. Train the neural network first.")
                
        except Exception as e:
            self._append_log(f"[ERROR] Failed to view training progress: {e}")
    
    def _on_auto_play_games_changed(self):
        """Handle changes to auto play games count"""
        try:
            games = int(self.auto_play_games_var.get())
            self._prefs.setdefault('mcts_ui', {})['auto_play_games'] = games
            self._save_prefs()
        except Exception as e:
            self._append_log(f"[ERROR] Auto play games change error: {e}")
    
    def _toggle_auto_play(self):
        """Start or stop auto play"""
        if self._auto_play_active:
            self._stop_auto_play()
        else:
            self._start_auto_play()
    
    def _start_auto_play(self):
        """Start auto play mode"""
        try:
            games_to_play = int(self.auto_play_games_var.get())
            if games_to_play <= 0:
                self._append_log("[AUTO PLAY] Error: Number of games must be greater than 0")
                return
            
            self._auto_play_active = True
            self._auto_play_games_remaining = games_to_play
            self._auto_play_games_completed = 0
            
            self.auto_play_button.config(text="Stop Auto Play", style="Accent.TButton")
            self.auto_play_spin.config(state="disabled")
            
            self._append_log(f"[AUTO PLAY] Starting auto play for {games_to_play} games")
            self._append_log(f"[AUTO PLAY] Training data collection: {'ENABLED' if self.collect_data_var.get() else 'DISABLED'}")
            
            # Start the first game
            self._queue_next_auto_play_game()
            
        except Exception as e:
            self._append_log(f"[AUTO PLAY ERROR] Failed to start auto play: {e}")
            self._auto_play_active = False
    
    def _stop_auto_play(self):
        """Stop auto play mode"""
        self._auto_play_active = False
        self.auto_play_button.config(text="Start Auto Play", style="")
        self.auto_play_spin.config(state="normal")
        
        self._append_log(f"[AUTO PLAY] Stopped auto play. Completed {self._auto_play_games_completed} games")
    
    def _queue_next_auto_play_game(self):
        """Queue the next ladder game for auto play"""
        if not self._auto_play_active or self._auto_play_games_remaining <= 0:
            if self._auto_play_active:
                self._append_log(f"[AUTO PLAY] Completed all {self._auto_play_games_completed} games!")
                self._stop_auto_play()
            return
        
        try:
            # Queue a ladder game
            self._submit(self._ladder(1))
            self._append_log(f"[AUTO PLAY] Queued game {self._auto_play_games_completed + 1}/{self._auto_play_games_completed + self._auto_play_games_remaining}")
            
        except Exception as e:
            self._append_log(f"[AUTO PLAY ERROR] Failed to queue game: {e}")
            self._stop_auto_play()


def launch_mcts_window(root: tk.Tk, username: str, password: Optional[str],
                        server_mode: str, custom_ws: Optional[str], battle_format: str) -> MCTSWindow:
    return MCTSWindow(root, username=username, password=password,
                      server_mode=server_mode, custom_ws=custom_ws, battle_format=battle_format)
