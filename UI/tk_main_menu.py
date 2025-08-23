# UI/tk_main_menu.py
"""
Simple launcher for Stockfish and MCTS models with Tkinter.

- Lets the user enter Showdown credentials, pick server and format.
- Launches the chosen model window.
"""

from __future__ import annotations

# --- injected by fix_type_effectiveness.py (safe no-op if repeated) ---
try:
    from utils.type_effectiveness import install_logging_hook, init_typecalc
    from pathlib import Path
    import os
    install_logging_hook()
    # Dynamic showdown data dir discovery
    _root = Path(__file__).resolve().parent.parent
    _cands = [
        Path(os.getenv('POKECHAD_SHOWDOWN_DIR', '')),
        _root / 'tools' / 'Data' / 'showdown',
        _root / 'showdown',
        _root / 'tools' / 'showdown',
        _root / 'Resources' / 'showdown',
    ]
    _chosen = None
    for _c in _cands:
        if _c and ((_c / 'moves.json').exists() or (_c / 'pokedex.json').exists()):
            _chosen = _c; break
    if _chosen is None:
        _chosen = _cands[1]
    init_typecalc(_chosen)
except Exception as _typecalc_e:
    import logging
    logging.getLogger('typecalc').warning('Could not install typecalc logging: %s', _typecalc_e)
# --- end injection ---
import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import json
from pathlib import Path
import time
import threading

from UI.tk_stockfish_model_ui import launch_stockfish_window, KNOWN_FORMATS  # type: ignore
from UI.tk_mcts_model_ui import launch_mcts_window  # type: ignore
from UI.tk_mcts_model_ui_modern import ModernMCTSModelUI  # type: ignore


class MainMenu(tk.Tk):
    CONFIG_PATH = Path.home() / '.pokechad_ui_settings.json'
    def __init__(self):
        super().__init__()
        self.title("PokeCHAD — Main Menu")
        self.geometry("520x340")
        self._loaded_settings = self._load_settings()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build(self._loaded_settings)

    def _load_settings(self) -> dict:
        try:
            if self.CONFIG_PATH.exists():
                with open(self.CONFIG_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
        return {}

    def _save_settings(self):
        try:
            data = {
                'username': self.username.get().strip() if hasattr(self, 'username') else None,
                'password': self.password.get() if hasattr(self, 'password') else None,
                'server_mode': self.server_mode.get() if hasattr(self, 'server_mode') else None,
                'custom_ws': self.custom_ws.get().strip() if hasattr(self, 'custom_ws') else None,
                'format': self.format_var.get().strip() if hasattr(self, 'format_var') else None,
            }
            with open(self.CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _build(self, settings: dict):
        frame = ttk.Frame(self)
        frame.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        ttk.Label(frame, text="Username").grid(row=0, column=0, sticky="e", padx=6, pady=6)
        self.username = tk.StringVar(value=settings.get('username') or os.environ.get("PS_USERNAME", "Containedo3mini"))
        ttk.Entry(frame, textvariable=self.username, width=28).grid(row=0, column=1, sticky="w", padx=6, pady=6)

        ttk.Label(frame, text="Password (optional)").grid(row=1, column=0, sticky="e", padx=6, pady=6)
        self.password = tk.StringVar(value=settings.get('password') or os.environ.get("PS_PASSWORD", "Kodbe4-gobpot-bujmoh"))
        ttk.Entry(frame, textvariable=self.password, width=28, show="*").grid(row=1, column=1, sticky="w", padx=6, pady=6)

        ttk.Label(frame, text="Server").grid(row=2, column=0, sticky="e", padx=6, pady=6)
        self.server_mode = tk.StringVar(value=settings.get('server_mode') or "Showdown")
        ttk.Combobox(frame, textvariable=self.server_mode, values=["Showdown", "Localhost"], width=24, state="readonly").grid(row=2, column=1, sticky="w", padx=6, pady=6)

        ttk.Label(frame, text="Custom WS URL").grid(row=3, column=0, sticky="e", padx=6, pady=6)
        self.custom_ws = tk.StringVar(value=settings.get('custom_ws') or "")
        ttk.Entry(frame, textvariable=self.custom_ws, width=28).grid(row=3, column=1, sticky="w", padx=6, pady=6)

        ttk.Label(frame, text="Format").grid(row=4, column=0, sticky="e", padx=6, pady=6)
        self.format_var = tk.StringVar(value=settings.get('format') or "gen9randombattle")
        ttk.Combobox(frame, textvariable=self.format_var, values=KNOWN_FORMATS, width=24, state="readonly").grid(row=4, column=1, sticky="w", padx=6, pady=6)

        def go_stockfish():
            self._save_settings()
            launch_stockfish_window(self, username=self.username.get().strip(),
                                    password=(self.password.get().strip() or None),
                                    server_mode=self.server_mode.get(),
                                    custom_ws=(self.custom_ws.get().strip() or None),
                                    battle_format=self.format_var.get().strip())
        ttk.Button(frame, text="Launch Stockfish Model", command=go_stockfish).grid(row=5, column=0, columnspan=2, pady=(16,4))

        def go_mcts():
            self._save_settings()
            launch_mcts_window(self, username=self.username.get().strip(),
                               password=(self.password.get().strip() or None),
                               server_mode=self.server_mode.get(),
                               custom_ws=(self.custom_ws.get().strip() or None),
                               battle_format=self.format_var.get().strip())
        
        def go_mcts_modern():
            self._save_settings()
            ModernMCTSModelUI(self, username=self.username.get().strip(),
                            password=(self.password.get().strip() or None),
                            server_mode=self.server_mode.get(),
                            custom_ws=(self.custom_ws.get().strip() or None),
                            battle_format=self.format_var.get().strip())
                            
        # MCTS buttons side by side
        mcts_frame = ttk.Frame(frame)
        mcts_frame.grid(row=6, column=0, columnspan=2, pady=(6,4))
        
        ttk.Button(mcts_frame, text="Launch MCTS Model", command=go_mcts).grid(row=0, column=0, padx=(0,4))
        ttk.Button(mcts_frame, text="Launch MCTS (Modern UI)", command=go_mcts_modern).grid(row=0, column=1, padx=(4,0))

        ttk.Button(frame, text="Run Weight Tuner", command=self._open_tuner_dialog).grid(row=7, column=0, columnspan=2, pady=(10,6))
        
        # Batch Collection Button
        ttk.Button(frame, text="🎮 Batch Data Collection", command=self._launch_batch_collection).grid(row=8, column=0, columnspan=2, pady=(6,6))

    # --- Weight Tuner Dialog ---
    def _open_tuner_dialog(self):
        dlg = tk.Toplevel(self); dlg.title("Weight Tuner")
        dlg.geometry("960x560")

        # Params frame
        params = ttk.LabelFrame(dlg, text='Parameters'); params.grid(row=0, column=0, columnspan=3, sticky='ew', padx=6, pady=6)
        for i in range(16): params.columnconfigure(i, weight=0)
        ttk.Label(params, text="Format").grid(row=0, column=0, sticky='e', padx=4, pady=4)
        fmt_var = tk.StringVar(value=self.format_var.get())
        ttk.Combobox(params, textvariable=fmt_var, values=KNOWN_FORMATS, width=22, state='readonly').grid(row=0, column=1, sticky='w', padx=4, pady=4)
        def _intvar(default): return tk.IntVar(value=default)
        def _dblvar(default): return tk.DoubleVar(value=default)
        pop_var = _intvar(16); gen_var = _intvar(10); seeds_var = _intvar(16)
        sigma_var = _dblvar(1.0); promote_diff_var = _dblvar(0.02); patience_var = _intvar(10)
        online_var = tk.BooleanVar(value=False)
        ttk.Label(params, text="Population").grid(row=0, column=2, sticky='e', padx=4, pady=4)
        ttk.Entry(params, textvariable=pop_var, width=6).grid(row=0, column=3, sticky='w', padx=2, pady=4)
        ttk.Label(params, text="Generations").grid(row=0, column=4, sticky='e', padx=4, pady=4)
        ttk.Entry(params, textvariable=gen_var, width=6).grid(row=0, column=5, sticky='w', padx=2, pady=4)
        ttk.Label(params, text="Seeds").grid(row=0, column=6, sticky='e', padx=4, pady=4)
        ttk.Entry(params, textvariable=seeds_var, width=6).grid(row=0, column=7, sticky='w', padx=2, pady=4)
        ttk.Label(params, text="Sigma").grid(row=0, column=8, sticky='e', padx=4, pady=4)
        ttk.Entry(params, textvariable=sigma_var, width=6).grid(row=0, column=9, sticky='w', padx=2, pady=4)
        ttk.Label(params, text="PromoteΔ").grid(row=0, column=10, sticky='e', padx=4, pady=4)
        ttk.Entry(params, textvariable=promote_diff_var, width=6).grid(row=0, column=11, sticky='w', padx=2, pady=4)
        ttk.Label(params, text="Patience").grid(row=0, column=12, sticky='e', padx=4, pady=4)
        ttk.Entry(params, textvariable=patience_var, width=6).grid(row=0, column=13, sticky='w', padx=2, pady=4)
        ttk.Checkbutton(params, text="Online", variable=online_var).grid(row=0, column=14, sticky='w', padx=8, pady=4)
        reset_hist_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(params, text="ResetHist", variable=reset_hist_var).grid(row=0, column=15, sticky='w', padx=4, pady=4)
        prime_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(params, text="Prime baseline from latest champion", variable=prime_var).grid(row=1, column=0, columnspan=6, sticky='w', padx=8, pady=(0,4))

        # Progress bar + status
        prog_frame = ttk.Frame(dlg); prog_frame.grid(row=1, column=0, columnspan=3, sticky='ew', padx=6, pady=(0,6))
        ttk.Label(prog_frame, text='Generation Progress:').pack(side=tk.LEFT, padx=(0,6))
        gen_progress = ttk.Progressbar(prog_frame, orient='horizontal', mode='determinate', maximum=gen_var.get())
        gen_progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,6))
        status_var = tk.StringVar(value='Idle')
        ttk.Label(prog_frame, textvariable=status_var).pack(side=tk.LEFT)

        # Candidates tree
        cand_frame = ttk.LabelFrame(dlg, text='Current Generation Candidates')
        cand_frame.grid(row=2, column=0, sticky='nsew', padx=6, pady=6)
        dlg.rowconfigure(2, weight=1)
        columns = ('idx','wr','wins','games','diff','sigma')
        cand_tree = ttk.Treeview(cand_frame, columns=columns, show='headings', height=10)
        headers = {'idx':'IDX','wr':'WR','wins':'W','games':'G','diff':'ΔChampion','sigma':'σ'}
        widths = {'idx':40,'wr':60,'wins':50,'games':50,'diff':90,'sigma':60}
        for c in columns:
            cand_tree.heading(c, text=headers.get(c,c))
            cand_tree.column(c, width=widths.get(c,80), anchor=tk.CENTER)
        cand_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        cand_scroll = ttk.Scrollbar(cand_frame, orient='vertical', command=cand_tree.yview)
        cand_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        cand_tree.configure(yscrollcommand=cand_scroll.set)

        # Champion weights
        champ_frame = ttk.LabelFrame(dlg, text='Champion Weights')
        champ_frame.grid(row=2, column=1, sticky='nsew', padx=6, pady=6)
        dlg.rowconfigure(2, weight=1)
        champ_text = tk.Text(champ_frame, height=14, width=34)
        champ_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # History panel
        history_frame = ttk.LabelFrame(dlg, text='Champion History')
        history_frame.grid(row=2, column=2, rowspan=2, sticky='nsew', padx=6, pady=6)
        dlg.columnconfigure(2, weight=1)
        history_list = tk.Listbox(history_frame, height=16)
        history_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4,0), pady=4)
        hist_scroll = ttk.Scrollbar(history_frame, orient='vertical', command=history_list.yview)
        hist_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        history_list.configure(yscrollcommand=hist_scroll.set)
        hist_btn_frame = ttk.Frame(history_frame); hist_btn_frame.pack(fill=tk.X, padx=4, pady=(0,4))
        refresh_hist_btn = ttk.Button(hist_btn_frame, text='Refresh History')
        refresh_hist_btn.pack(side=tk.LEFT, padx=(0,4))
        export_hist_btn = ttk.Button(hist_btn_frame, text='Export Selected')
        export_hist_btn.pack(side=tk.LEFT, padx=(0,4))
        load_hist_btn = ttk.Button(hist_btn_frame, text='Load Selected')
        load_hist_btn.pack(side=tk.LEFT, padx=(0,4))
        use_latest_btn = ttk.Button(hist_btn_frame, text='Use Latest Champion')
        use_latest_btn.pack(side=tk.RIGHT)

        # Log box
        log_frame = ttk.LabelFrame(dlg, text='Raw Log')
        log_frame.grid(row=3, column=0, columnspan=3, sticky='nsew', padx=6, pady=6)
        dlg.rowconfigure(3, weight=1)
        log_box = tk.Text(log_frame, height=8)
        log_box.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Control buttons
        btn_frame = ttk.Frame(dlg); btn_frame.grid(row=5, column=0, columnspan=3, sticky='ew', padx=6, pady=(0,6))
        start_btn = ttk.Button(btn_frame, text='Start')
        start_btn.pack(side=tk.LEFT, padx=(0,6))
        stop_btn = ttk.Button(btn_frame, text='Stop')
        stop_btn.pack(side=tk.LEFT, padx=(0,6))
        close_btn = ttk.Button(btn_frame, text='Close', command=dlg.destroy)
        close_btn.pack(side=tk.LEFT)
        reload_btn = ttk.Button(btn_frame, text='Reload champion from file')
        reload_btn.pack(side=tk.RIGHT)

        # Shared state
        state = {
            'current_gen': 0,
            'champion_wr': 0.5,
            'champion': {},
            'sigma': None,
            'candidates': [],
            'champion_wr_at_gen_start': 0.5,
            'stop': False,
            'log_pos': 0,
            'log_path': Path('logs')/ 'weight_tuning.jsonl',
            'run_proc': None,
            'expected_generations': lambda: gen_var.get(),
            'history': [],
            'selected_history': None,
            'monitor_active': False,
            'param_refs': {
                'population': pop_var,
                'generations': gen_var,
                'seeds': seeds_var,
                'sigma': sigma_var,
                'promote_diff': promote_diff_var,
                'patience': patience_var,
                'online': online_var,
                'reset_hist': reset_hist_var,
            }
        }
        lock = threading.Lock()

        def append_log(msg: str):
            try:
                log_box.insert(tk.END, msg + '\n'); log_box.see(tk.END)
            except Exception:
                pass

        def refresh_ui():
            if state['stop'] or not dlg.winfo_exists():
                return
            with lock:
                try:
                    gen_progress.configure(maximum=state['expected_generations']())
                    gen_progress['value'] = state['current_gen']
                except Exception:
                    pass
                try:
                    sigma_val = state.get('sigma')
                    sigma_str = '?' if sigma_val is None else f"{float(sigma_val):.3f}"
                    status_var.set(f"Gen {state['current_gen']} σ={sigma_str}")
                except Exception:
                    status_var.set(f"Gen {state['current_gen']} σ=?")
                try:
                    for iid in cand_tree.get_children():
                        cand_tree.delete(iid)
                    base_wr = state.get('champion_wr_at_gen_start', 0.5) or 0.0
                    for c in sorted(state['candidates'], key=lambda d: d.get('win_rate',0) or 0, reverse=True):
                        diff = (c.get('win_rate',0) or 0) - base_wr
                        cand_tree.insert('', tk.END, values=(c.get('idx'), f"{(c.get('win_rate',0) or 0):.3f}", c.get('wins'), c.get('games'), f"{diff:.3f}", f"{(c.get('sigma',0) or 0):.2f}"))
                except Exception:
                    pass
                try:
                    champ_text.delete('1.0', tk.END)
                    src_weights = state['selected_history']['weights'] if state.get('selected_history') else state['champion']
                    for k, v in sorted((src_weights or {}).items()):
                        champ_text.insert(tk.END, f"{k}: {float(v):.3f}\n")
                except Exception:
                    pass
            dlg.after(1000, refresh_ui)

        def build_history_from_log():
            path = state['log_path']
            hist = []
            init_seen = False
            if path.exists():
                try:
                    with open(path,'r',encoding='utf-8') as f:
                        for line in f:
                            if not line.strip():
                                continue
                            try:
                                evt = json.loads(line)
                            except Exception:
                                continue
                            et = evt.get('event')
                            if et == 'init' and not init_seen:
                                hist.append({'gen': 0, 'wr': 0.5, 'sigma': evt.get('sigma'), 'weights': evt.get('champion') or {}, 'time': None})
                                init_seen = True
                            elif et == 'promote':
                                hist.append({'gen': evt.get('generation'), 'wr': evt.get('champion_wr'), 'sigma': evt.get('sigma'), 'weights': evt.get('new_champion') or {}, 'time': evt.get('gen_seconds')})
                            elif et == 'done':
                                hist.append({'gen': evt.get('generations'), 'wr': evt.get('champion_wr'), 'sigma': None, 'weights': evt.get('final_champion') or {}, 'time': None})
                except Exception:
                    pass
            dedup = {}
            for h in hist:
                dedup[h['gen']] = h
            out = list(sorted(dedup.values(), key=lambda d: d['gen']))
            state['history'] = out
            history_list.delete(0, tk.END)
            for h in out:
                try:
                    history_list.insert(tk.END, f"G{h['gen']} WR={float(h['wr'] or 0):.3f} σ={h['sigma'] if h['sigma'] is not None else '-'}")
                except Exception:
                    pass
            if out and not state.get('champion') and out[0].get('gen') == 0:
                state['champion'] = out[0].get('weights') or {}

        def on_history_select(event=None):
            try:
                sel = history_list.curselection()
                if not sel:
                    state['selected_history'] = None; return
                idx = sel[0]
                if 0 <= idx < len(state['history']):
                    state['selected_history'] = state['history'][idx]
            except Exception:
                state['selected_history'] = None
        history_list.bind('<<ListboxSelect>>', on_history_select)

        def export_selected():
            h = state.get('selected_history')
            if not h:
                append_log('[warn] no history selection to export'); return
            weights = h.get('weights') or {}
            if not weights:
                append_log('[warn] selected history has no weights'); return
            try:
                models_dir = Path('Models'); models_dir.mkdir(exist_ok=True)
                target = models_dir / 'weights.json'
                if target.exists():
                    backup = target.with_name(f"weights.json.bak.{int(time.time())}")
                    try: target.replace(backup)
                    except Exception: pass
                with open(target,'w',encoding='utf-8') as f:
                    json.dump(weights, f, indent=2)
                append_log('[export] wrote selected champion to Models/weights.json')
            except Exception as e:
                append_log(f'[error] export failed: {e}')
        refresh_hist_btn.configure(command=build_history_from_log)
        export_hist_btn.configure(command=export_selected)
        def load_selected():
            h = state.get('selected_history')
            if not h:
                append_log('[warn] no history selection to load'); return
            state['champion'] = h.get('weights') or state['champion']
            append_log(f"[load] loaded history G{h.get('gen')} into champion view (not saved yet)")
        load_hist_btn.configure(command=load_selected)

        def monitor_log():
            last_size = 0
            last_error_ts = 0.0
            state['monitor_active'] = True
            while not state['stop']:
                path = state['log_path']
                if path.exists():
                    try:
                        cur_size = path.stat().st_size
                        if cur_size < last_size:
                            last_size = 0
                        with open(path, 'r', encoding='utf-8') as f:
                            if last_size:
                                try:
                                    f.seek(last_size)
                                except Exception:
                                    last_size = 0; f.seek(0)
                            while True:
                                line = f.readline()
                                if not line:
                                    break
                                line = line.strip()
                                try:
                                    last_size = f.tell()
                                except Exception:
                                    pass
                                if not line:
                                    continue
                                try:
                                    evt = json.loads(line)
                                except Exception:
                                    append_log(line); continue
                                et = evt.get('event')
                                with lock:
                                    if et == 'gen_start':
                                        state['current_gen'] = int(evt.get('generation', state['current_gen']))
                                        state['champion_wr'] = float(evt.get('champion_wr', state['champion_wr']))
                                        state['champion_wr_at_gen_start'] = state['champion_wr']
                                        state['champion'] = evt.get('champion') or state['champion']
                                        state['sigma'] = evt.get('sigma', state.get('sigma'))
                                        state['candidates'] = []
                                    elif et == 'candidate':
                                        state['candidates'].append({'idx': evt.get('idx'), 'win_rate': evt.get('win_rate'), 'wins': evt.get('wins'), 'games': evt.get('games'), 'sigma': evt.get('sigma')})
                                    elif et == 'promote':
                                        state['champion'] = evt.get('new_champion') or state['champion']
                                        state['champion_wr'] = float(evt.get('champion_wr', state['champion_wr']))
                                        state['sigma'] = evt.get('sigma', state.get('sigma'))
                                        state['history'].append({'gen': evt.get('generation'), 'wr': evt.get('champion_wr'), 'sigma': evt.get('sigma'), 'weights': evt.get('new_champion') or {}, 'time': evt.get('gen_seconds')})
                                    elif et == 'no_improve':
                                        state['sigma'] = evt.get('sigma', state.get('sigma'))
                                    elif et == 'init':
                                        state['champion'] = evt.get('champion') or state['champion']
                                        need_init = True
                                        for h in state['history']:
                                            if h.get('gen') == 0:
                                                need_init = False; break
                                        if need_init:
                                            state['history'].append({'gen': 0, 'wr': 0.5, 'sigma': evt.get('sigma'), 'weights': evt.get('champion') or {}, 'time': None})
                                    elif et == 'done':
                                        state['current_gen'] = int(evt.get('generations', state['current_gen']))
                                        final_gen = evt.get('generations')
                                        final_entry = {
                                            'gen': final_gen,
                                            'wr': evt.get('champion_wr'),
                                            'sigma': None,
                                            'weights': evt.get('final_champion') or state.get('champion', {}),
                                            'time': None,
                                        }
                                        already = any(h.get('gen') == final_gen for h in state['history'])
                                        if not already:
                                            state['history'].append(final_entry)
                                    if et in ('promote','done','init'):
                                        seen = {h['gen']: h for h in state['history']}
                                        state['history'] = list(sorted(seen.values(), key=lambda d: d['gen']))
                                        history_list.delete(0, tk.END)
                                        for h in state['history']:
                                            try:
                                                history_list.insert(tk.END, f"G{h['gen']} WR={h['wr']:.3f} σ={h['sigma'] if h['sigma'] is not None else '-'}")
                                            except Exception:
                                                pass
                                append_log(line)
                    except Exception as e:
                        import time as _time
                        now = _time.time()
                        if now - last_error_ts > 2.0:
                            append_log(f"[monitor-error] {e}")
                            last_error_ts = now
                import time as _time
                _time.sleep(0.5)

        def start_monitor():
            if state.get('monitor_active'):
                return
            state['stop'] = False
            threading.Thread(target=monitor_log, daemon=True).start()

        def reload_champion():
            try:
                weights_path = Path('Models') / 'weights.json'
                if weights_path.exists():
                    with open(weights_path,'r',encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        with lock:
                            state['champion'] = data
                        append_log('[reload] loaded champion weights from Models/weights.json')
                else:
                    append_log('[reload] Models/weights.json not found')
            except Exception as e:
                append_log(f'[reload-error] {e}')
        reload_btn.configure(command=reload_champion)

        try:
            build_history_from_log()
        except Exception:
            pass

        def _on_gen_change(*_):
            try: gen_progress.configure(maximum=int(gen_var.get()))
            except Exception: pass
        gen_var.trace_add('write', _on_gen_change)

        def start_run():
            if state.get('run_proc') is not None:
                append_log('[warn] tuner already running')
                return
            start_monitor()
            if reset_hist_var.get():
                try:
                    log_path = state['log_path']
                    if log_path.exists():
                        bak = log_path.with_name(f"weight_tuning.jsonl.bak.{int(time.time())}")
                        log_path.replace(bak)
                        append_log(f"[reset] previous log rotated to {bak.name}")
                except Exception as e:
                    append_log(f"[reset] log rotate failed: {e}")
                with lock:
                    state['history'].clear(); state['selected_history']=None
                try:
                    history_list.delete(0, tk.END)
                except Exception:
                    pass
            with lock:
                state['current_gen']=0; state['candidates'].clear(); state['champion_wr']=0.5; state['champion_wr_at_gen_start']=0.5; state['sigma']=None; state['selected_history']=None
            try:
                log_box.delete('1.0', tk.END)
            except Exception:
                pass
            try:
                if prime_var.get():
                    hist = state.get('history') or []
                    if hist:
                        latest = hist[-1]
                        weights = latest.get('weights') or {}
                        if weights:
                            models_dir = Path('Models'); models_dir.mkdir(exist_ok=True)
                            target = models_dir / 'weights.json'
                            if target.exists():
                                backup = target.with_name(f"weights.json.bak.{int(time.time())}")
                                try: target.replace(backup)
                                except Exception: pass
                            with open(target,'w',encoding='utf-8') as f:
                                json.dump(weights, f, indent=2)
                            append_log(f"[prime] baseline set to latest champion (G{latest.get('gen')}) in Models/weights.json")
                            with lock:
                                state['champion'] = weights
            except Exception as e:
                append_log(f"[prime-error] {e}")
            fmt = fmt_var.get().strip() or 'gen9randombattle'
            pop = max(2, int(pop_var.get()))
            gens = max(1, int(gen_var.get()))
            seeds = max(2, int(seeds_var.get()))
            sigma = float(sigma_var.get())
            promote_delta = float(promote_diff_var.get())
            patience = max(1, int(patience_var.get()))
            args = [sys.executable, str(Path(__file__).resolve().parent.parent / 'tools' / 'weight_tuner.py'), '--format', fmt, '--population', str(pop), '--generations', str(gens), '--seeds', str(seeds), '--sigma', f"{sigma}", '--min-promote-diff', f"{promote_delta}", '--patience', str(patience)]
            if online_var.get():
                args.append('--online')
            append_log('[run] ' + ' '.join(args))
            status_var.set('Running')
            def worker():
                import subprocess
                try:
                    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    state['run_proc'] = proc
                    for line in proc.stdout:  # type: ignore
                        append_log(line.rstrip())
                    proc.wait()
                    code = proc.returncode
                    if code == 0:
                        append_log('[done] tuner finished.')
                        status_var.set('Finished')
                    else:
                        append_log(f'[error] exit code {code}')
                        status_var.set(f'Error {code}')
                except Exception as e:
                    append_log(f'[error] {e}')
                    status_var.set('Error')
                finally:
                    state['run_proc'] = None
            threading.Thread(target=worker, daemon=True).start()
        start_btn.configure(command=start_run)

        def stop_run():
            proc = state.get('run_proc')
            if not proc:
                append_log('[info] no active run to stop')
                return
            try:
                proc.terminate()
                append_log('[stop] sent terminate to tuner process')
                status_var.set('Stopping...')
            except Exception as e:
                append_log(f'[error] stop failed: {e}')
        stop_btn.configure(command=stop_run)

        def on_close_dialog():
            state['stop'] = True
            dlg.destroy()
        dlg.protocol('WM_DELETE_WINDOW', on_close_dialog)

        refresh_ui()

    # --- Batch Collection Launcher ---
    def _launch_batch_collection(self):
        """Launch batch collection with authentication from main menu"""
        username = self.username.get().strip()
        password = self.password.get().strip()
        
        if not username or not password:
            messagebox.showerror(
                "Authentication Required", 
                "Please enter your Pokemon Showdown username and password before launching batch collection."
            )
            return
        
        # Launch batch collection in a new thread
        def run_batch_collection():
            try:
                import asyncio
                import sys
                from pathlib import Path
                
                # Add project root to path if needed
                project_root = Path(__file__).parent.parent
                if str(project_root) not in sys.path:
                    sys.path.insert(0, str(project_root))
                
                # Import and run batch collection with auth
                from launch_batch_collection import main as batch_main
                asyncio.run(batch_main(username=username, password=password))
                
            except Exception as e:
                print(f"❌ Batch collection failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Show info message and launch
        messagebox.showinfo(
            "Batch Collection Starting", 
            f"Launching batch collection with account: {username}\n\n"
            f"The collection will start in a new terminal window.\n"
            f"Check the terminal for progress and collection statistics."
        )
        
        threading.Thread(target=run_batch_collection, daemon=True).start()

    def _on_close(self):
        self._save_settings()
        try: self.destroy()
        except Exception: pass

    def run(self):
        self.mainloop()


def main():
    MainMenu().run()


if __name__ == "__main__":
    main()
