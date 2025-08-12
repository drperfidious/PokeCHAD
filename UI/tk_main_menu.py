# UI/tk_main_menu.py
"""
Simple launcher for Stockfish model with Tkinter.

- Lets the user enter Showdown credentials, pick server and format.
- Launches the Stockfish window.
"""

from __future__ import annotations

import os
import sys
import tkinter as tk
from tkinter import ttk

_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from UI.tk_stockfish_model_ui import launch_stockfish_window, KNOWN_FORMATS  # type: ignore

class MainMenu(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PokeCHAD — Main Menu")
        self.geometry("520x320")
        self._build()

    def _build(self):
        frame = ttk.Frame(self)
        frame.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        ttk.Label(frame, text="Username").grid(row=0, column=0, sticky="e", padx=6, pady=6)
        self.username = tk.StringVar(value=os.environ.get("PS_USERNAME", "Containedo3mini"))
        ttk.Entry(frame, textvariable=self.username, width=28).grid(row=0, column=1, sticky="w", padx=6, pady=6)

        ttk.Label(frame, text="Password (optional)").grid(row=1, column=0, sticky="e", padx=6, pady=6)
        self.password = tk.StringVar(value=os.environ.get("PS_PASSWORD", "Kodbe4-gobpot-bujmoh"))
        ttk.Entry(frame, textvariable=self.password, width=28, show="*").grid(row=1, column=1, sticky="w", padx=6, pady=6)

        ttk.Label(frame, text="Server").grid(row=2, column=0, sticky="e", padx=6, pady=6)
        self.server_mode = tk.StringVar(value="Showdown")
        ttk.Combobox(frame, textvariable=self.server_mode, values=["Showdown", "Localhost"], width=24, state="readonly").grid(row=2, column=1, sticky="w", padx=6, pady=6)

        ttk.Label(frame, text="Custom WS URL").grid(row=3, column=0, sticky="e", padx=6, pady=6)
        self.custom_ws = tk.StringVar(value="")
        ttk.Entry(frame, textvariable=self.custom_ws, width=28).grid(row=3, column=1, sticky="w", padx=6, pady=6)

        ttk.Label(frame, text="Format").grid(row=4, column=0, sticky="e", padx=6, pady=6)
        self.format_var = tk.StringVar(value="gen9randombattle")
        ttk.Combobox(frame, textvariable=self.format_var, values=KNOWN_FORMATS, width=24, state="readonly").grid(row=4, column=1, sticky="w", padx=6, pady=6)

        def go():
            launch_stockfish_window(self, username=self.username.get().strip(),
                                    password=(self.password.get().strip() or None),
                                    server_mode=self.server_mode.get(),
                                    custom_ws=(self.custom_ws.get().strip() or None),
                                    battle_format=self.format_var.get().strip())

        ttk.Button(frame, text="Launch Stockfish Model", command=go).grid(row=5, column=0, columnspan=2, pady=(18,8))

    def run(self):
        self.mainloop()


if __name__ == "__main__":
    MainMenu().run()
