#-------------------------------------------------------------------------------
# NFoil: a Numba JIT-accelerated version of mfoil for subsonic airfoil analysis.
#
# Copyright (C) 2026 Cayetano Martínez-Muriel
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#-------------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk

import threading
import copy

from nfoil import nfoil

try:
    from taichi_fields import TaichiFlowField
    TAICHI_AVAILABLE = True
except ImportError:
    TAICHI_AVAILABLE = False

# Setup appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams['axes.labelsize'] = 20   # Tamaño de las etiquetas de los ejes
plt.rcParams['xtick.labelsize'] = 18  # Tamaño de los ticks del eje X
plt.rcParams['ytick.labelsize'] = 18  # Tamaño de los ticks del eje Y
plt.rcParams['axes.titlesize'] = 20   # Tamaño del título
emerald   = "#16c172"

plt.close('all')

class NFoilApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("NFoil - Advanced Airfoil Solver")
        self.geometry("1400x1200")
        self.minsize(1000, 700)
        
        # State Data for Export
        self.state_data = {
            'cp': None,
            'bl': None,
            'geom': None,
            'polars': None
        }

        # Current nfoil active state to be passed sideways
        self.last_M = None
        self.last_alpha = 0.0
        self.sweep_results = {}
        self.polars_list = []
        self.init_bl_var = None
        
        # Start Numba warmup thread
        threading.Thread(target=self.warmup_numba, daemon=True).start()

        # Grid Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # -- LEFT SIDEBAR --
        self.sidebar_frame = ctk.CTkScrollableFrame(self, width=320, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text=r"NFoil", font=ctk.CTkFont(size=22, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(10, 5), sticky="w")

        # Geometry Settings
        self.geom_frame = ctk.CTkFrame(self.sidebar_frame)
        self.geom_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        
        ctk.CTkLabel(self.geom_frame, text="Geometry", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.naca_var = ctk.StringVar(value="2412")
        self.n_panels_var = ctk.IntVar(value=100)
        self.flap_x_var = ctk.StringVar(value="0.0")
        self.flap_z_var = ctk.StringVar(value="0.0")
        self.flap_eta_var = ctk.StringVar(value="0")
        self.xref_var = ctk.StringVar(value="0.25")
        
        ctk.CTkLabel(self.geom_frame, text="NACA:").grid(row=1, column=0, padx=10, pady=2, sticky="w")
        ctk.CTkEntry(self.geom_frame, textvariable=self.naca_var, width=120).grid(row=1, column=1, padx=10, pady=2, sticky="w")
        
        ctk.CTkLabel(self.geom_frame, text="Panels:").grid(row=2, column=0, padx=10, pady=2, sticky="w")
        ctk.CTkEntry(self.geom_frame, textvariable=self.n_panels_var, width=120).grid(row=2, column=1, padx=10, pady=2, sticky="w")

        ctk.CTkLabel(self.geom_frame, text="Flap (X, Z):").grid(row=3, column=0, padx=10, pady=2, sticky="w")
        flap_xz_frame = ctk.CTkFrame(self.geom_frame, fg_color="transparent")
        flap_xz_frame.grid(row=3, column=1, padx=10, pady=2, sticky="w")
        ctk.CTkEntry(flap_xz_frame, textvariable=self.flap_x_var, width=55, placeholder_text="0.8").pack(side="left", padx=(0, 5))
        ctk.CTkEntry(flap_xz_frame, textvariable=self.flap_z_var, width=55, placeholder_text="0.0").pack(side="left")

        ctk.CTkLabel(self.geom_frame, text="Flap Angle (deg):").grid(row=4, column=0, padx=10, pady=2, sticky="w")
        ctk.CTkEntry(self.geom_frame, textvariable=self.flap_eta_var, width=120, placeholder_text="eta").grid(row=4, column=1, padx=10, pady=2, sticky="w")
        
        ctk.CTkLabel(self.geom_frame, text="Moment Ref X:").grid(row=5, column=0, padx=10, pady=2, sticky="w")
        ctk.CTkEntry(self.geom_frame, textvariable=self.xref_var, width=120).grid(row=5, column=1, padx=10, pady=2, sticky="w")

        self.loaded_coords = None  # custom airfoil coordinates
        airfoil_btn_frame = ctk.CTkFrame(self.geom_frame, fg_color="transparent")
        airfoil_btn_frame.grid(row=6, column=0, columnspan=2, padx=10, pady=(5, 0))
        ctk.CTkButton(airfoil_btn_frame, text="Load Airfoil", height=24, width=100,
                      command=self._load_airfoil_file).pack(side="left", padx=(0, 5))
        ctk.CTkButton(airfoil_btn_frame, text="Unload", height=24, width=70,
                      fg_color="#6b7280", hover_color="#4b5563",
                      command=self._unload_airfoil).pack(side="left")
        self.loaded_label = ctk.CTkLabel(self.geom_frame, text="", font=ctk.CTkFont(size=11))
        self.loaded_label.grid(row=7, column=0, columnspan=2, padx=10, pady=0)
        
        # Flow State
        self.flow_frame = ctk.CTkFrame(self.sidebar_frame)
        self.flow_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

        ctk.CTkLabel(self.flow_frame, text="Flow State", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.re_var = ctk.DoubleVar(value=1e6)
        self.alpha_var = ctk.DoubleVar(value=2.0)
        self.mach_var = ctk.DoubleVar(value=0.0)
        self.viscous_var = ctk.BooleanVar(value=True)
        self.reuse_bl_var = ctk.BooleanVar(value=False)
        self.givencl_var = ctk.BooleanVar(value=False)
        self.cltgt_var = ctk.DoubleVar(value=0.5)

        ctk.CTkLabel(self.flow_frame, text="Reynolds:").grid(row=1, column=0, padx=10, pady=2, sticky="w")
        ctk.CTkEntry(self.flow_frame, textvariable=self.re_var, width=120).grid(row=1, column=1, padx=10, pady=2, sticky="w")
        
        ctk.CTkLabel(self.flow_frame, text="Alpha (deg):").grid(row=2, column=0, padx=10, pady=2, sticky="w")
        ctk.CTkEntry(self.flow_frame, textvariable=self.alpha_var, width=120).grid(row=2, column=1, padx=10, pady=2, sticky="w")
        
        ctk.CTkLabel(self.flow_frame, text="Mach:").grid(row=3, column=0, padx=10, pady=2, sticky="w")
        ctk.CTkEntry(self.flow_frame, textvariable=self.mach_var, width=120).grid(row=3, column=1, padx=10, pady=2, sticky="w")

        ctk.CTkSwitch(self.flow_frame, text="Target Cl", variable=self.givencl_var).grid(row=4, column=0, padx=10, pady=(10, 2), sticky="w")
        ctk.CTkEntry(self.flow_frame, textvariable=self.cltgt_var, width=60).grid(row=4, column=1, padx=10, pady=(10, 2), sticky="w")

        ctk.CTkSwitch(self.flow_frame, text="Viscous Solver", variable=self.viscous_var).grid(row=5, column=0, columnspan=2, padx=10, pady=(2, 2), sticky="w")
        ctk.CTkSwitch(self.flow_frame, text="Reuse BL", variable=self.reuse_bl_var).grid(row=6, column=0, columnspan=2, padx=10, pady=(2, 2), sticky="w")

        self.xftu_var = ctk.StringVar(value="")
        self.xftl_var = ctk.StringVar(value="")
        ctk.CTkLabel(self.flow_frame, text="xft (U, L):").grid(row=7, column=0, padx=10, pady=2, sticky="w")
        xft_frame = ctk.CTkFrame(self.flow_frame, fg_color="transparent")
        xft_frame.grid(row=7, column=1, padx=10, pady=2, sticky="w")
        ctk.CTkEntry(xft_frame, textvariable=self.xftu_var, width=55, placeholder_text="1.0").pack(side="left", padx=(0, 5))
        ctk.CTkEntry(xft_frame, textvariable=self.xftl_var, width=55, placeholder_text="1.0").pack(side="left")

        # Solver Settings
        self.solver_frame = ctk.CTkFrame(self.sidebar_frame)
        self.solver_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        
        ctk.CTkLabel(self.solver_frame, text="Solver Settings", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.max_iter_var = ctk.IntVar(value=50)
        self.tol_var = ctk.StringVar(value="1e-10")
        self.ncrit_var = ctk.DoubleVar(value=9.0)
        self.verb_var = ctk.IntVar(value=1)
        
        ctk.CTkLabel(self.solver_frame, text="Max Iters:").grid(row=1, column=0, padx=10, pady=2, sticky="w")
        ctk.CTkEntry(self.solver_frame, textvariable=self.max_iter_var, width=120).grid(row=1, column=1, padx=10, pady=2, sticky="w")
        
        ctk.CTkLabel(self.solver_frame, text="Tol (1e-X):").grid(row=2, column=0, padx=10, pady=2, sticky="w")
        ctk.CTkEntry(self.solver_frame, textvariable=self.tol_var, width=120).grid(row=2, column=1, padx=10, pady=2, sticky="w")

        ctk.CTkLabel(self.solver_frame, text="Ncrit:").grid(row=3, column=0, padx=10, pady=2, sticky="w")
        ctk.CTkEntry(self.solver_frame, textvariable=self.ncrit_var, width=120).grid(row=3, column=1, padx=10, pady=2, sticky="w")

        ctk.CTkLabel(self.solver_frame, text="Verbosity:").grid(row=4, column=0, padx=10, pady=2, sticky="w")
        ctk.CTkEntry(self.solver_frame, textvariable=self.verb_var, width=120).grid(row=4, column=1, padx=10, pady=2, sticky="w")

        self.btn_run = ctk.CTkButton(self.sidebar_frame, text="Run Single Point", command=self.run_single, fg_color="#38bdf8", text_color="#0f172a", hover_color="#0284c7")
        self.btn_run.grid(row=4, column=0, padx=20, pady=10, sticky="ew")

        # Alpha Sweep
        self.sweep_frame = ctk.CTkFrame(self.sidebar_frame)
        self.sweep_frame.grid(row=5, column=0, padx=20, pady=10, sticky="ew")

        ctk.CTkLabel(self.sweep_frame, text="Polars Computation", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=5, sticky="w", columnspan=3)
        
        self.sw_min_var = ctk.DoubleVar(value=-5.0)
        self.sw_max_var = ctk.DoubleVar(value=15.0)
        self.sw_n_var = ctk.IntVar(value=21)

        ctk.CTkLabel(self.sweep_frame, text="Min").grid(row=1, column=0, padx=(10, 2), pady=0)
        ctk.CTkLabel(self.sweep_frame, text="Max").grid(row=1, column=1, padx=2, pady=0)
        ctk.CTkLabel(self.sweep_frame, text="Pts").grid(row=1, column=2, padx=(2, 10), pady=0)

        ctk.CTkEntry(self.sweep_frame, textvariable=self.sw_min_var, width=60).grid(row=2, column=0, padx=(10, 2), pady=(0, 10))
        ctk.CTkEntry(self.sweep_frame, textvariable=self.sw_max_var, width=60).grid(row=2, column=1, padx=2, pady=(0, 10))
        ctk.CTkEntry(self.sweep_frame, textvariable=self.sw_n_var, width=60).grid(row=2, column=2, padx=(2, 10), pady=(0, 10))

        self.btn_sweep = ctk.CTkButton(self.sidebar_frame, text="Compute Polars", command=self.run_sweep, fg_color="#34d399", text_color="#0f172a", hover_color="#059669")
        self.btn_sweep.grid(row=6, column=0, padx=20, pady=10, sticky="ew")

        # Sweeps selection dropdown
        self.sweep_dropdown = ctk.CTkOptionMenu(self.sidebar_frame, values=["---"], command=self.load_sweep_point)
        self.sweep_dropdown.grid(row=7, column=0, padx=20, pady=0, sticky="ew")
        self.sweep_dropdown.set("---")
        
        # Output Log / Status
        self.status_label = ctk.CTkLabel(self.sidebar_frame, text="Ready.", text_color="#94a3b8", anchor="w")
        self.status_label.grid(row=8, column=0, padx=20, pady=10, sticky="ew")
        
        self.results_textbox = ctk.CTkTextbox(self.sidebar_frame, height=120, font=ctk.CTkFont(family="monospace", size=13))
        self.results_textbox.grid(row=9, column=0, padx=20, pady=10, sticky="ew")
        self.results_textbox.insert("0.0", "--- Run to see forces ---\n")
        
        # Export Buttons
        self.export_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.export_frame.grid(row=10, column=0, padx=20, pady=10, sticky="sew")
        
        ctk.CTkButton(self.export_frame, text="Export Cp", height=24, width=120, command=lambda: self.export_data('cp')).grid(row=0, column=0, padx=2, pady=2)
        ctk.CTkButton(self.export_frame, text="Export Geom", height=24, width=120, command=lambda: self.export_data('geom')).grid(row=0, column=1, padx=2, pady=2)
        ctk.CTkButton(self.export_frame, text="Export BL", height=24, width=120, command=lambda: self.export_data('bl')).grid(row=1, column=0, padx=2, pady=2)
        ctk.CTkButton(self.export_frame, text="Export Polars", height=24, width=120, command=lambda: self.export_data('polars')).grid(row=1, column=1, padx=2, pady=2)
        
        # Taichi Integration Controls
        self.taichi_frame = ctk.CTkFrame(self.sidebar_frame)
        self.taichi_frame.grid(row=11, column=0, padx=20, pady=10, sticky="ew")
        
        ctk.CTkLabel(self.taichi_frame, text="Flow Field Parameters", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=5, sticky="w", columnspan=2)
        
        self.streams_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(self.taichi_frame, text="Draw Streamlines", variable=self.streams_var).grid(row=1, column=0, padx=10, pady=5, sticky="w")
        
        self.bl_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(self.taichi_frame, text="Draw BL", variable=self.bl_var).grid(row=1, column=1, padx=10, pady=5, sticky="w")
        
        self.bl_color_var = ctk.StringVar(value="blue")
        ctk.CTkLabel(self.taichi_frame, text="BL Color:").grid(row=2, column=0, padx=10, pady=2, sticky="w")
        ctk.CTkEntry(self.taichi_frame, textvariable=self.bl_color_var, width=120).grid(row=2, column=1, padx=10, pady=2, sticky="w")
        
        self.tf_bounds_var = ctk.StringVar(value="-0.5, 1.5, -0.5, 0.5")
        ctk.CTkLabel(self.taichi_frame, text="Bounds (L,R,B,T):").grid(row=3, column=0, padx=10, pady=2, sticky="w")
        ctk.CTkEntry(self.taichi_frame, textvariable=self.tf_bounds_var, width=120).grid(row=3, column=1, padx=10, pady=2, sticky="w")

        self.btn_taichi = ctk.CTkButton(self.taichi_frame, text="Generate Flow Fields", command=self.open_flowfield_window, fg_color="#6366f1", text_color="#f8fafc", hover_color="#4f46e5", state="normal" if TAICHI_AVAILABLE else "disabled")
        self.btn_taichi.grid(row=4, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        # -- RIGHT MAIN CONTENT --
        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        self.tabview = ctk.CTkTabview(self.main_frame, corner_radius=0)
        self.tabview.grid(row=0, column=0, sticky="nsew")
        
        self.tab_main = self.tabview.add(r"Aerodynamic Analysis")
        self.tab_bl = self.tabview.add("Boundary Layer Properties")
        
        self.tab_main.configure(fg_color="#1e293b")
        self.tab_bl.configure(fg_color="#1e293b")

        self.tab_main.grid_columnconfigure(0, weight=1)
        self.tab_main.grid_rowconfigure(0, weight=1)
        self.tab_bl.grid_columnconfigure(0, weight=1)
        self.tab_bl.grid_rowconfigure(0, weight=1)

        # Map Matplotlib styling to dark theme
        plt.style.use('dark_background')
        plt.rcParams.update({
            "figure.facecolor": "#1e293b",
            "axes.facecolor": "#0f172a",
            "axes.edgecolor": "#334155",
            "axes.labelcolor": "#f8fafc",
            "xtick.color": "#f8fafc",
            "ytick.color": "#f8fafc",
            "text.color": "#f8fafc",
            "grid.color": "#334155",
            "lines.linewidth": 1.5
        })

        # Tab 1: Main Grid
        self.fig = Figure(figsize=(8, 6), dpi=100, layout="constrained")
        gs = self.fig.add_gridspec(2, 2)
        
        self.ax_cp = self.fig.add_subplot(gs[0, 0])
        self.ax_cl_alpha = self.fig.add_subplot(gs[0, 1])
        self.ax_geom = self.fig.add_subplot(gs[1, 0])
        self.ax_polars = self.fig.add_subplot(gs[1, 1])


        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab_main)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", pady=(15, 0))
        
        self.toolbar_frame_main = ctk.CTkFrame(master=self.tab_main, fg_color="transparent")
        self.toolbar_frame_main.grid(row=1, column=0, sticky="ew")
        self.toolbar_main = NavigationToolbar2Tk(self.canvas, self.toolbar_frame_main)
        self.toolbar_main.update()

        # Tab 2: BL Grid with dropdown-selectable quantities
        self.bl_quantities = ['Cf', 'δ*', 'θ', 'Hk', 'ue', 'Amp/Ctau', 'Reθ']
        self.bl_selections = [ctk.StringVar(value=q) for q in ['Cf', 'δ*', 'θ', 'Hk']]

        # Top row: 4 dropdowns for each subplot
        bl_controls_frame = ctk.CTkFrame(self.tab_bl, fg_color="transparent")
        bl_controls_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=(5, 0))
        for i in range(4):
            bl_controls_frame.grid_columnconfigure(i, weight=1)
            menu = ctk.CTkOptionMenu(bl_controls_frame, values=self.bl_quantities,
                                     variable=self.bl_selections[i],
                                     command=lambda v: self._refresh_bl_plots(),
                                     width=120, height=24)
            menu.grid(row=0, column=i, padx=5, pady=2)

        self.fig_bl = Figure(figsize=(10, 8), dpi=100, layout="constrained")
        self.axs_bl = self.fig_bl.subplots(4, 1)

        self.ax_cf = self.axs_bl[0]
        self.ax_ds = self.axs_bl[1]
        self.ax_th = self.axs_bl[2]
        self.ax_hk = self.axs_bl[3]

        self.canvas_bl = FigureCanvasTkAgg(self.fig_bl, master=self.tab_bl)
        self.canvas_bl.draw()
        self.canvas_bl.get_tk_widget().grid(row=1, column=0, sticky="nsew", pady=(5, 0))
        self.tab_bl.grid_rowconfigure(1, weight=1)

        self.toolbar_frame_bl = ctk.CTkFrame(master=self.tab_bl, fg_color="transparent")
        self.toolbar_frame_bl.grid(row=2, column=0, sticky="ew")
        self.toolbar_bl = NavigationToolbar2Tk(self.canvas_bl, self.toolbar_frame_bl)
        self.toolbar_bl.update()

        self.init_plots()
        
        # Auto run on startup
        # self.after(500, self.run_single)  # Removed auto-run on startup

    def init_plots(self):
        self.ax_cp.set_title(r"$C_p$ Distribution")
        self.ax_cp.set_xlabel(r"$x/c$")
        self.ax_cp.set_ylabel(r"$-Cp$")
        # if not self.ax_cp.yaxis_inverted():
            # self.ax_cp.invert_yaxis()
        self.ax_cp.grid(True, linestyle='--', alpha=0.5)

        self.ax_geom.set_title(r"Geometry \& Wake")
        self.ax_geom.set_xlabel(r"$x/c$")
        self.ax_geom.set_ylabel(r"$z/c$")
        self.ax_geom.set_aspect('equal', 'datalim')
        self.ax_geom.grid(True, linestyle='--', alpha=0.5)

        self.ax_cf.set_title(r"Skin Friction $C_f$")
        self.ax_cf.set_xlabel(r"$s$")
        self.ax_cf.set_ylabel(r"$C_f$")
        self.ax_cf.set_yscale('log')
        self.ax_cf.grid(True, linestyle='--', alpha=0.5)

        self.ax_cl_alpha.set_title(r"$C_l$ vs $\alpha$")
        self.ax_cl_alpha.set_xlabel(r"$\alpha$ (deg)")
        self.ax_cl_alpha.set_ylabel(r"$C_l$")
        self.ax_cl_alpha.grid(True, linestyle='--', alpha=0.5)

        self.ax_polars.set_title(r"Aerodynamic polar")
        self.ax_polars.set_xlabel(r"$C_d$")
        self.ax_polars.set_ylabel(r"$C_l$")
        self.ax_polars.grid(True, linestyle='--', alpha=0.5)

        # BL tab: configure labels from dropdown selections
        bl_labels = {
            'Cf':        (r'Skin Friction $C_f$', r'$C_f$', True),
            'δ*':        (r'Displacement Thickness ($\delta^*$)', r'$\delta^*$', False),
            'θ':         (r'Momentum Thickness ($\theta$)', r'$\theta$', False),
            'Hk':        (r'Shape Parameter ($H_k$)', r'$H_k$', False),
            'ue':        (r'Edge Velocity ($u_e$)', r'$u_e$', False),
            'Amp/Ctau':  (r'Amplification / $\sqrt{C_\tau}$', r'$n$ or $\sqrt{C_\tau}$', False),
            'Reθ':       (r'Reynolds number $Re_\theta$', r'$Re_\theta$', False),
        }
        for i, ax in enumerate(self.axs_bl):
            sel = self.bl_selections[i].get()
            title, ylabel, logscale = bl_labels.get(sel, (sel, sel, False))
            ax.set_title(title)
            ax.set_xlabel(r'$s$')
            ax.set_ylabel(ylabel)
            if logscale:
                ax.set_yscale('log')
            else:
                ax.set_yscale('linear')
            ax.grid(True, linestyle='--', alpha=0.5)

    def warmup_numba(self):
        try:
            self.print_status("Numba warmup in progress...")
            # Extremely fast dummy run just to compile decorators behind the scenes
            dummy = nfoil(None, '0012', 20)
            dummy.param.doplot = False
            dummy.setoper(alpha=0.0, visc=True)
            dummy.solve()
            self.print_status("Ready.")
        except Exception:
            pass

    def print_status(self, text):
        self.status_label.configure(text=text)
        self.update_idletasks()

    def print_results(self, text):
        self.results_textbox.delete("0.0", "end")
        self.results_textbox.insert("0.0", text)

    def _load_airfoil_file(self):
        """Load airfoil coordinates from a file (x z columns, starting from TE)."""
        from tkinter import filedialog
        filepath = filedialog.askopenfilename(
            title="Load Airfoil Coordinates",
            filetypes=[("DAT files", "*.dat"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not filepath:
            return
        try:
            # Read coordinates, skipping comment/header lines
            lines = []
            with open(filepath) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or line.startswith('%'):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            x, z = float(parts[0]), float(parts[1])
                            lines.append([x, z])
                        except ValueError:
                            continue  # skip header lines like "NACA 2412"
            if len(lines) < 5:
                messagebox.showerror("Load Error", f"Only {len(lines)} valid coordinate points found.")
                return
            coords = np.array(lines).T  # shape (2, N)
            self.loaded_coords = coords
            import os
            fname = os.path.basename(filepath)
            self.loaded_label.configure(text=f"✓ {fname} ({coords.shape[1]} pts)")
            self.print_status(f"Loaded airfoil: {fname} ({coords.shape[1]} points)")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def _unload_airfoil(self):
        """Clear loaded airfoil coordinates, reverting to NACA generation."""
        self.loaded_coords = None
        self.loaded_label.configure(text="")
        self.print_status("Airfoil unloaded. Using NACA generator.")

    def run_single(self):
        try:
            self.print_status("Running solver... (First run takes ~3-5s to compile Numba JIT)")
            self.status_label.update()
            naca = self.naca_var.get()
            n_panels = self.n_panels_var.get()
            reuse_bl = self.reuse_bl_var.get()
            givencl = self.givencl_var.get()
            cltgt = self.cltgt_var.get()
            
            # Reuse previous setup if geometry is identical and Reuse BL is checked
            if reuse_bl and self.last_M and getattr(self.last_M, "naca_id", None) == naca and self.last_M.foil.N == n_panels:
                M = self.last_M
                M.param.doplot = False
            else:
                if self.loaded_coords is not None:
                    M = nfoil(self.loaded_coords, naca, n_panels)
                else:
                    M = nfoil(None, naca, n_panels)
                M.naca_id = naca
                M.param.doplot = False
                
                flap_x_str = self.flap_x_var.get().strip()
                flap_z_str = self.flap_z_var.get().strip()
                flap_eta_str = self.flap_eta_var.get().strip()
                if flap_x_str and flap_z_str and flap_eta_str:
                    try:
                        M.geom_flap(np.array([float(flap_x_str), float(flap_z_str)]), float(flap_eta_str))
                    except ValueError:
                        pass
                        
            xref_str = self.xref_var.get().strip()
            if xref_str:
                try:
                    M.geom.xref = np.array([float(xref_str), 0.0])
                except ValueError:
                    pass
                    
            M.param.niglob = self.max_iter_var.get()
            try:
                M.param.rtol = float(self.tol_var.get().strip())
            except ValueError:
                pass
            M.param.ncrit = self.ncrit_var.get()
            M.param.verb = self.verb_var.get()

            # Forced transition x/c
            xftu_str = self.xftu_var.get().strip()
            xftl_str = self.xftl_var.get().strip()
            if xftu_str:
                try: M.param.xftu = float(xftu_str)
                except ValueError: pass
            if xftl_str:
                try: M.param.xftl = float(xftl_str)
                except ValueError: pass
            
            viscous = self.viscous_var.get()
            alpha = self.alpha_var.get()
            mach = self.mach_var.get()
            re = self.re_var.get()

            if givencl:
                M.setoper(cl=cltgt, alpha=alpha, Re=re, Ma=mach, visc=viscous)
            elif viscous:
                M.setoper(alpha=alpha, Re=re, Ma=mach, visc=True)
            else:
                M.setoper(alpha=alpha, Ma=mach, visc=False)
                
            M.oper.initbl = not reuse_bl
                
            M.solve()

            alpha_result = M.oper.alpha
            self.print_results(f"Alpha: {alpha_result:.2f} deg\nCl: {M.post.cl:.4f}\nCd: {M.post.cd:.5f}\nCm: {M.post.cm:.4f}")
            self.last_M = M
            self.last_alpha = alpha_result
            self.update_plots(M, alpha_result)
            self.print_status("Ready.")
        except Exception as e:
            self.status_label.configure(text=f"Error: {str(e)}")
            messagebox.showerror("Solver Error", str(e))

    def load_sweep_point(self, choice):
        if choice in getattr(self, 'sweep_results', {}):
            M_choice = self.sweep_results[choice]
            # Extract alpha from the label (format: "NACA... α=X.XX ...")
            try:
                alpha_choice = float(choice.split('α=')[1].split('°')[0])
            except (IndexError, ValueError):
                alpha_choice = 0.0
            self.last_M = M_choice
            self.last_alpha = alpha_choice
            
            self.print_results(f"Loaded: {choice}\nCl: {M_choice.post.cl:.4f}\nCd: {M_choice.post.cd:.5f}\nCm: {M_choice.post.cm:.4f}")
            self.update_plots(M_choice, alpha_choice)

    def update_plots(self, M, alpha):
        # Update Cp
        self.ax_cp.clear()
        self.ax_geom.clear()
        self.ax_cf.clear()
        self.ax_ds.clear()
        self.ax_th.clear()
        self.ax_hk.clear()
        
        self.init_plots()
        
        x_foil = M.foil.x[0, :]
        z_foil = M.foil.x[1, :]
        
        x_cp = x_foil.copy()
        if self.viscous_var.get() and M.wake.N > 0:
            x_cp = np.concatenate((x_cp, M.wake.x[0, :]))

        cp = M.post.cp if hasattr(M.post, 'cp') else []
        cpi = M.post.cpi if hasattr(M.post, 'cpi') else []
        
        if len(cp) > 0:
            self.ax_cp.plot(x_cp, -cp, '-', color='#38bdf8', label='Viscous Cp')
        if len(cpi) > 0:
            self.ax_cp.plot(x_cp, -cpi, '--', color='#f472b6', label='Inviscid Cp')
        if len(cp) > 0 or len(cpi) > 0:
            self.ax_cp.legend(fontsize=12)
            
            if self.viscous_var.get() and hasattr(M, 'glob') and not getattr(M.glob, 'conv', False):
                self.ax_cp.text(0.5, 0.5, "NOT CONVERGED", color='#ef4444', fontsize=18, fontweight='bold',
                                horizontalalignment='center', verticalalignment='center', transform=self.ax_cp.transAxes)
            
            self.state_data['cp'] = {
                'title': f'Cp Distribution NACA {self.naca_var.get()} Alpha={alpha}',
                'headers': ['x/c', 'Cp_viscous', 'Cp_inviscid'],
                'data': np.column_stack((x_cp, cp, cpi))
            }

        # Update Geom
        self.ax_geom.fill(x_foil, z_foil, color='#94a3b8', alpha=0.3)
        self.ax_geom.plot(x_foil, z_foil, color='#94a3b8')
        
        if self.viscous_var.get() and M.wake.N > 0:
            self.ax_geom.plot(M.wake.x[0, :], M.wake.x[1, :], ':', color='#34d399')
            
        self.state_data['geom'] = {
            'title': f'Geometry NACA {self.naca_var.get()}',
            'headers': ['x', 'z'],
            'data': np.column_stack((x_foil, z_foil))
        }

        # Update Cf
        self.state_data['bl'] = None
        if self.viscous_var.get() and hasattr(M.vsol, 'Is') and len(M.vsol.Is) >= 2:
            Is_l = M.vsol.Is[0]
            Is_u = M.vsol.Is[1]
            if hasattr(M.post, 'ds'):
                s = M.isol.xi
                
                # Plot physical BL thickness (delta* approx) on top of the geometry
                t_u = M.foil.t[:, Is_u]
                n_u = np.array([-t_u[1, :], t_u[0, :]])
                delta_u = M.post.ds[Is_u]
                x_bl_u = x_foil[Is_u] + n_u[0, :] * delta_u
                z_bl_u = z_foil[Is_u] + n_u[1, :] * delta_u
                
                t_l = M.foil.t[:, Is_l]
                n_l = np.array([-t_l[1, :], t_l[0, :]])
                delta_l = M.post.ds[Is_l]
                x_bl_l = x_foil[Is_l] + n_l[0, :] * delta_l
                z_bl_l = z_foil[Is_l] + n_l[1, :] * delta_l
                
                self.ax_geom.plot(x_bl_u, z_bl_u, '-', color='c', alpha=0.8, label="Upper BL")
                self.ax_geom.plot(x_bl_l, z_bl_l, '-', color='m', alpha=0.8, label="Lower BL")
                
                # Don't plot wake BL in the main geom figure as requested
                
                self.ax_geom.legend(fontsize=12, loc="upper right")
            
            if hasattr(M.post, 'cf'):
                s = M.isol.xi
                self.ax_cf.plot(s[Is_u], M.post.cf[Is_u], '-', color='c', label='Upper Cf')
                self.ax_cf.plot(s[Is_l], M.post.cf[Is_l], '-', color='m', label='Lower Cf')
                self.ax_cf.legend(fontsize=12)
                
                # Zero padding for mismatched lengths
                max_len = max(len(s[Is_u]), len(s[Is_l]))
                arr = np.zeros((max_len, 4))
                arr[:len(s[Is_u]), 0] = s[Is_u]
                arr[:len(s[Is_u]), 1] = M.post.cf[Is_u]
                arr[:len(s[Is_l]), 2] = s[Is_l]
                arr[:len(s[Is_l]), 3] = M.post.cf[Is_l]
                
                self.state_data['bl'] = self._build_bl_export(M, alpha, Is_u, Is_l)
                
            # BL tab: plot selected quantities via dropdowns
            self._plot_bl_tab(M, Is_u, Is_l)

        self.canvas.draw()
        self.canvas_bl.draw()

    def run_sweep(self):
        self.btn_sweep.configure(state="disabled")
        # Read ALL tkinter variables on the main thread to avoid thread-safety issues
        config = {
            'naca': self.naca_var.get(),
            'n_panels': self.n_panels_var.get(),
            'reuse_bl': self.reuse_bl_var.get(),
            'sw_min': self.sw_min_var.get(),
            'sw_max': self.sw_max_var.get(),
            'sw_n': self.sw_n_var.get(),
            'viscous': self.viscous_var.get(),
            'mach': self.mach_var.get(),
            're': self.re_var.get(),
            'flap_x': self.flap_x_var.get().strip(),
            'flap_z': self.flap_z_var.get().strip(),
            'flap_eta': self.flap_eta_var.get().strip(),
            'xref': self.xref_var.get().strip(),
            'max_iter': self.max_iter_var.get(),
            'tol': self.tol_var.get().strip(),
            'ncrit': self.ncrit_var.get(),
            'verb': self.verb_var.get(),
            'last_M': self.last_M,
            'last_alpha': getattr(self, 'last_alpha', 0.0),
            'loaded_coords': self.loaded_coords,
        }
        threading.Thread(target=self._run_sweep_thread, args=(config,), daemon=True).start()
        
    def _run_sweep_thread(self, cfg):
        try:
            self.print_status("Running alpha sweep (First run takes ~3-5s to compile Numba JIT)...")
            naca = cfg['naca']
            n_panels = cfg['n_panels']
            
            reuse_bl = cfg['reuse_bl']
            alphas = np.linspace(cfg['sw_min'], cfg['sw_max'], cfg['sw_n'])
            
            valid_last_m = False
            if reuse_bl and cfg['last_M'] and getattr(cfg['last_M'], "naca_id", None) == naca and cfg['last_M'].foil.N == n_panels:
                valid_last_m = True
                
            viscous = cfg['viscous']
            mach = cfg['mach']
            re = cfg['re']

            # Initialize base sweep point
            if valid_last_m:
                start_alpha = cfg['last_alpha']
                M_converged = copy.deepcopy(cfg['last_M'])
                M_converged.param.doplot = False
            else:
                start_alpha = 0.0
                if cfg['loaded_coords'] is not None:
                    M_converged = nfoil(cfg['loaded_coords'], naca, n_panels)
                else:
                    M_converged = nfoil(None, naca, n_panels)
                M_converged.naca_id = naca
                M_converged.param.doplot = False
                
                flap_x_str = cfg['flap_x']
                flap_z_str = cfg['flap_z']
                flap_eta_str = cfg['flap_eta']
                if flap_x_str and flap_z_str and flap_eta_str:
                    try:
                        M_converged.geom_flap(np.array([float(flap_x_str), float(flap_z_str)]), float(flap_eta_str))
                    except ValueError:
                        pass
                xref_str = cfg['xref']
                if xref_str:
                    try:
                        M_converged.geom.xref = np.array([float(xref_str), 0.0])
                    except ValueError:
                        pass
                M_converged.param.niglob = cfg['max_iter']
                try:
                    M_converged.param.rtol = float(cfg['tol'])
                except ValueError:
                    pass
                M_converged.param.ncrit = cfg['ncrit']
                M_converged.param.verb = cfg['verb']
                
                self.print_status(f"Initializing sweep at geometric 0.0 deg...")
                if viscous:
                    M_converged.setoper(alpha=start_alpha, Re=re, Ma=mach, visc=True)
                else:
                    M_converged.setoper(alpha=start_alpha, Ma=mach, visc=False)
                    
                M_converged.oper.initbl = True
                M_converged.solve()
                
                if viscous and (not hasattr(M_converged, 'glob') or not getattr(M_converged.glob, 'conv', False)):
                    self.print_status(f"Alpha {start_alpha:.2f} did not converge. Sweep aborted.")
                    self.after(0, self.btn_sweep.configure, {"state": "normal"})
                    return
                if np.isnan(M_converged.post.cl) or np.isnan(M_converged.post.cd):
                    self.print_status(f"Alpha {start_alpha:.2f} diverged (NaNs). Sweep aborted.")
                    self.after(0, self.btn_sweep.configure, {"state": "normal"})
                    return
                    
            alphas_up = [a for a in alphas if a >= start_alpha]
            alphas_up.sort()
            alphas_down = [a for a in alphas if a < start_alpha]
            alphas_down.sort(reverse=True)
            
            results = [] 
            
            M_up = copy.deepcopy(M_converged)
            M_last_good_up = copy.deepcopy(M_converged)
            for i, a in enumerate(alphas_up):
                self.print_status(f"Computing Polar Alpha {a:.2f} (Upward)")
                if viscous:
                    M_up.setoper(alpha=a, Re=re, Ma=mach, visc=True)
                else:
                    M_up.setoper(alpha=a, Ma=mach, visc=False)
                M_up.oper.initbl = False
                M_up.solve()
                
                converged = True
                if viscous and (not hasattr(M_up, 'glob') or not getattr(M_up.glob, 'conv', False)):
                    converged = False
                if np.isnan(M_up.post.cl) or np.isnan(M_up.post.cd):
                    converged = False
                
                if not converged:
                    # Retry with cold start from last good state
                    self.print_status(f"Alpha {a:.2f} did not converge. Retrying with cold start...")
                    M_up = copy.deepcopy(M_last_good_up)
                    if viscous:
                        M_up.setoper(alpha=a, Re=re, Ma=mach, visc=True)
                    else:
                        M_up.setoper(alpha=a, Ma=mach, visc=False)
                    M_up.oper.initbl = True
                    M_up.solve()
                    
                    converged = True
                    if viscous and (not hasattr(M_up, 'glob') or not getattr(M_up.glob, 'conv', False)):
                        converged = False
                    if np.isnan(M_up.post.cl) or np.isnan(M_up.post.cd):
                        converged = False
                
                if converged:
                    cdp = getattr(M_up.post, 'cdp', 0.0)
                    cdf = getattr(M_up.post, 'cdf', 0.0)
                    results.append((a, M_up.post.cl, M_up.post.cd, M_up.post.cm, cdp, cdf, copy.deepcopy(M_up)))
                    M_last_good_up = copy.deepcopy(M_up)
                else:
                    self.print_status(f"Alpha {a:.2f} skipped (did not converge after retry). Continuing...")
                    M_up = copy.deepcopy(M_last_good_up)  # reset to last good state
                
            M_down = copy.deepcopy(M_converged)
            M_last_good_down = copy.deepcopy(M_converged)
            for i, a in enumerate(alphas_down):
                self.print_status(f"Computing Polar Alpha {a:.2f} (Downward)")
                if viscous:
                    M_down.setoper(alpha=a, Re=re, Ma=mach, visc=True)
                else:
                    M_down.setoper(alpha=a, Ma=mach, visc=False)
                M_down.oper.initbl = False
                M_down.solve()
                
                converged = True
                if viscous and (not hasattr(M_down, 'glob') or not getattr(M_down.glob, 'conv', False)):
                    converged = False
                if np.isnan(M_down.post.cl) or np.isnan(M_down.post.cd):
                    converged = False
                
                if not converged:
                    # Retry with cold start from last good state
                    self.print_status(f"Alpha {a:.2f} did not converge. Retrying with cold start...")
                    M_down = copy.deepcopy(M_last_good_down)
                    if viscous:
                        M_down.setoper(alpha=a, Re=re, Ma=mach, visc=True)
                    else:
                        M_down.setoper(alpha=a, Ma=mach, visc=False)
                    M_down.oper.initbl = True
                    M_down.solve()
                    
                    converged = True
                    if viscous and (not hasattr(M_down, 'glob') or not getattr(M_down.glob, 'conv', False)):
                        converged = False
                    if np.isnan(M_down.post.cl) or np.isnan(M_down.post.cd):
                        converged = False
                
                if converged:
                    cdp = getattr(M_down.post, 'cdp', 0.0)
                    cdf = getattr(M_down.post, 'cdf', 0.0)
                    results.append((a, M_down.post.cl, M_down.post.cd, M_down.post.cm, cdp, cdf, copy.deepcopy(M_down)))
                    M_last_good_down = copy.deepcopy(M_down)
                else:
                    self.print_status(f"Alpha {a:.2f} skipped (did not converge after retry). Continuing...")
                    M_down = copy.deepcopy(M_last_good_down)  # reset to last good state
                
            results.sort(key=lambda x: x[0])
            
            alphas_conv = [r[0] for r in results]
            cls = [r[1] for r in results]
            cds = [r[2] for r in results]
            cms = [r[3] for r in results]
            cdps = [r[4] for r in results]
            cdfs = [r[5] for r in results]
            
            # Accumulate cases with descriptive labels (don't overwrite)
            for r in results:
                label = f"{naca} Re={re:.0e} α={r[0]:.2f}° M={mach}"
                self.sweep_results[label] = r[6]
            
            # Build dropdown from ALL accumulated cases, newest first
            dropdown_values = list(reversed(list(self.sweep_results.keys())))
            
            final_M = results[-1][6] if len(results) > 0 else M_converged

            self.after(0, self._finish_sweep_ui, naca, re, alphas_conv, cls, cds, cms, cdps, cdfs, dropdown_values, final_M)

        except Exception as e:
            self.after(0, self._handle_sweep_error, str(e))
            
    def _handle_sweep_error(self, err_msg):
        self.status_label.configure(text=f"Error: {err_msg}")
        messagebox.showerror("Sweep Error", err_msg)
        self.btn_sweep.configure(state="normal")
            
    def _finish_sweep_ui(self, naca, re, alphas_conv, cls, cds, cms, cdps, cdfs, dropdown_values, M):
        if len(alphas_conv) == 0:
            self.print_status("Sweep Failed: No points converged.")
            self.btn_sweep.configure(state="normal")
            return
            
        self.sweep_dropdown.configure(values=dropdown_values)
        self.sweep_dropdown.set(dropdown_values[-1])

        self.polars_list.append({
            'label': f'Re={re:.1e}',
            'alphas_conv': alphas_conv,
            'cds': cds,
            'cls': cls
        })

        # Update Polars Plot
        self.ax_polars.clear()
        self.ax_cl_alpha.clear()
        self.init_plots()
        for p in self.polars_list:
            self.ax_polars.plot(p['cds'], p['cls'], '-o', markersize=4, label=p['label'])
            self.ax_cl_alpha.plot(p['alphas_conv'], p['cls'], '-o', markersize=4, label=p['label'])
        self.ax_polars.legend(fontsize=12)
        self.ax_cl_alpha.legend(fontsize=12)
        self.canvas.draw()
        
        self.state_data['polars'] = {
            'title': f'Polars NACA {naca} Re={re}',
            'headers': ['Alpha', 'Cl', 'Cd', 'Cdp', 'Cdf', 'Cm'],
            'data': np.column_stack((alphas_conv, cls, cds, cdps, cdfs, cms))
        }
        
        self.last_M = M
        self.last_alpha = alphas_conv[-1]

        self.print_results(f"Polars Completed.\nConverged: {len(alphas_conv)}/{len(cls)}\nMax Cl: {max(cls):.4f}")
        self.print_status("Ready.")
        self.btn_sweep.configure(state="normal")

    def _get_bl_data(self, M, quantity, indices):
        """Return data for a given BL quantity at the specified indices."""
        q = quantity
        if q == 'Cf':
            return M.post.cf[indices] if hasattr(M.post, 'cf') else None
        elif q == 'δ*':
            return M.post.ds[indices] if hasattr(M.post, 'ds') else None
        elif q == 'θ':
            return M.post.th[indices] if hasattr(M.post, 'th') else None
        elif q == 'Hk':
            return M.post.Hk[indices] if hasattr(M.post, 'Hk') else None
        elif q == 'ue':
            return M.post.ue[indices] if hasattr(M.post, 'ue') else None
        elif q == 'Amp/Ctau':
            return M.post.sa[indices] if hasattr(M.post, 'sa') else None
        elif q == 'Reθ':
            return M.post.Ret[indices] if hasattr(M.post, 'Ret') else None
        return None

    def _plot_bl_tab(self, M, Is_u, Is_l):
        """Plot the 4 dropdown-selected BL quantities on the BL tab."""
        s_u = M.isol.xi[Is_u]
        s_l = M.isol.xi[Is_l]
        Is_w = M.vsol.Is[2] if len(M.vsol.Is) >= 3 else []
        s_w = M.isol.xi[Is_w] if len(Is_w) > 0 else []

        for i, ax in enumerate(self.axs_bl):
            sel = self.bl_selections[i].get()
            d_u = self._get_bl_data(M, sel, Is_u)
            d_l = self._get_bl_data(M, sel, Is_l)
            if d_u is not None:
                ax.plot(s_u, d_u, '-', color='c', label='Upper')
            if d_l is not None:
                ax.plot(s_l, d_l, '-', color='m', label='Lower')
            if len(Is_w) > 0:
                d_w = self._get_bl_data(M, sel, Is_w)
                if d_w is not None:
                    ax.plot(s_w, d_w, '-', color=emerald, label='Wake')
            ax.legend(fontsize=12)

    def _refresh_bl_plots(self):
        """Re-draw BL tab when a dropdown selection changes."""
        if not self.last_M or not self.viscous_var.get():
            return
        M = self.last_M
        if not hasattr(M.vsol, 'Is') or len(M.vsol.Is) < 2:
            return
        Is_u = M.vsol.Is[1]
        Is_l = M.vsol.Is[0]
        for ax in self.axs_bl:
            ax.clear()
        self.init_plots()
        self._plot_bl_tab(M, Is_u, Is_l)
        self.canvas_bl.draw()

    def _build_bl_export(self, M, alpha, Is_u, Is_l):
        """Build comprehensive BL export data dict."""
        s_u = M.isol.xi[Is_u]
        s_l = M.isol.xi[Is_l]
        naca = self.naca_var.get()

        # Collect all quantities for upper and lower
        fields_u = {'s': s_u}
        fields_l = {'s': s_l}
        for name, attr in [('cf', 'cf'), ('ds', 'ds'), ('th', 'th'), ('Hk', 'Hk'),
                           ('ue', 'ue'), ('sa', 'sa'), ('Ret', 'Ret')]:
            if hasattr(M.post, attr):
                fields_u[name] = getattr(M.post, attr)[Is_u]
                fields_l[name] = getattr(M.post, attr)[Is_l]

        # Stagnation point
        sstag = getattr(M.isol, 'sstag', 0.0)
        xstag = getattr(M.isol, 'xstag', np.array([0.0, 0.0]))

        # Transition locations
        xt_u = M.vsol.Xt[1, 1] if hasattr(M.vsol, 'Xt') else 0.0
        xt_l = M.vsol.Xt[0, 1] if hasattr(M.vsol, 'Xt') else 0.0

        return {
            'title': f'BL Properties NACA {naca} Alpha={alpha}',
            'fields_u': fields_u,
            'fields_l': fields_l,
            'sstag': sstag,
            'xstag': xstag,
            'xt_u': xt_u,
            'xt_l': xt_l,
            'naca': naca,
            'alpha': alpha,
        }

    def export_data(self, key):
        data = self.state_data.get(key)
        if not data:
            messagebox.showwarning("Export", "No data available. Run the solver first.")
            return

        if key == 'bl':
            self._export_bl_comprehensive(data)
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            initialfile=f"{data['title'].replace(' ', '_').lower()}.txt",
            title=f"Save {key} Data"
        )
        
        if filename:
            with open(filename, 'w') as f:
                f.write(f"# {data['title']}\n")
                f.write("# " + "\t".join(data['headers']) + "\n")
                np.savetxt(f, data['data'], fmt="%.6e", delimiter="\t")
            self.print_status(f"Saved {filename}")

    def _export_bl_comprehensive(self, data):
        """Export all BL data into a folder with upper.txt, lower.txt, info.txt, and plot_bl.py."""
        import os
        folder = filedialog.askdirectory(title="Select folder for BL export")
        if not folder:
            return

        naca = data['naca']
        alpha = data['alpha']
        prefix = f"bl_naca{naca}_alpha{alpha:.1f}"

        # Write upper surface
        fu = data['fields_u']
        cols_u = list(fu.keys())
        arr_u = np.column_stack([fu[c] for c in cols_u])
        upper_file = os.path.join(folder, f"{prefix}_upper.txt")
        with open(upper_file, 'w') as f:
            f.write(f"# Upper surface BL - NACA {naca}, alpha={alpha}\n")
            f.write("# " + "\t".join(cols_u) + "\n")
            np.savetxt(f, arr_u, fmt="%.6e", delimiter="\t")

        # Write lower surface
        fl = data['fields_l']
        cols_l = list(fl.keys())
        arr_l = np.column_stack([fl[c] for c in cols_l])
        lower_file = os.path.join(folder, f"{prefix}_lower.txt")
        with open(lower_file, 'w') as f:
            f.write(f"# Lower surface BL - NACA {naca}, alpha={alpha}\n")
            f.write("# " + "\t".join(cols_l) + "\n")
            np.savetxt(f, arr_l, fmt="%.6e", delimiter="\t")

        # Write info file
        info_file = os.path.join(folder, f"{prefix}_info.txt")
        with open(info_file, 'w') as f:
            f.write(f"NACA: {naca}\n")
            f.write(f"Alpha: {alpha}\n")
            f.write(f"Stagnation s: {data['sstag']:.6e}\n")
            xstag = data['xstag']
            if hasattr(xstag, '__len__') and len(xstag) >= 2:
                f.write(f"Stagnation x: {xstag[0]:.6e}\n")
                f.write(f"Stagnation z: {xstag[1]:.6e}\n")
            f.write(f"Transition x/c upper: {data['xt_u']:.6f}\n")
            f.write(f"Transition x/c lower: {data['xt_l']:.6f}\n")

        # Write Python loader script
        script_file = os.path.join(folder, f"plot_{prefix}.py")
        with open(script_file, 'w') as f:
            f.write(f'''#!/usr/bin/env python3
"""Auto-generated script to load and plot BL data for NACA {naca}, alpha={alpha}."""
import numpy as np
import matplotlib.pyplot as plt

# Load data
upper = np.loadtxt("{prefix}_upper.txt")
lower = np.loadtxt("{prefix}_lower.txt")
cols = {cols_u}

# Column mapping
idx = {{c: i for i, c in enumerate(cols)}}

fig, axes = plt.subplots(3, 2, figsize=(12, 10))
fig.suptitle("BL Properties NACA {naca}, $\\\\alpha$={alpha}°", fontsize=14)

quantities = [
    ('cf',  '$C_f$',                   True),
    ('ds',  '$\\\\delta^*$',           False),
    ('th',  '$\\\\theta$',             False),
    ('Hk',  '$H_k$',                   False),
    ('ue',  '$u_e$',                    False),
    ('Ret', '$Re_\\\\theta$',          False),
]

for ax, (col, label, logscale) in zip(axes.flat, quantities):
    if col in idx:
        ax.plot(upper[:, idx['s']], upper[:, idx[col]], '-', color='c', label='Upper')
        ax.plot(lower[:, idx['s']], lower[:, idx[col]], '-', color='m', label='Lower')
        ax.set_ylabel(label)
        ax.set_xlabel('$s$')
        if logscale:
            ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("{prefix}_plots.png", dpi=150)
plt.show()
print("Saved {prefix}_plots.png")
''')

        self.print_status(f"BL exported to {folder}/ ({len(cols_u)} quantities per surface)")


    def open_flowfield_window(self):
        if not self.last_M:
            messagebox.showwarning("Flow Field", "Please run a single point flow solution first.")
            return
            
        top = ctk.CTkToplevel(self)
        top.title(f"Flow Fields - NACA {self.naca_var.get()}")
        top.geometry("1400x400")
        
        lbl = ctk.CTkLabel(top, text="Computing massive flow-field array dynamically using GPU...", font=ctk.CTkFont(slant="italic"))
        lbl.pack(pady=1)
        top.update()
        
        try:
            bounds = [float(x.strip()) for x in self.tf_bounds_var.get().split(',')]
            if len(bounds) != 4: raise ValueError("Bounds must be 4 comma-separated numbers")
            
            solver = TaichiFlowField(res_x=700, res_z=700)
            X, Z, u, w = solver.solve(self.last_M, self.last_alpha, grid_bounds=tuple(bounds))
            mag = np.sqrt(u**2 + w**2)
            cp_f = 1.0 - mag**2
            
            tf_fig = Figure(figsize=(12, 5), dpi=100, layout="constrained")
            ax1 = tf_fig.add_subplot(121)
            ax2 = tf_fig.add_subplot(122)
            
            vmax_val = np.max(mag)
            cpmin_val = np.min(cp_f)
            
            c1 = ax1.contourf(X, Z, mag, levels=np.linspace(0, vmax_val, 50), cmap='viridis')
            cb1 = tf_fig.colorbar(c1, ax=ax1, label=r"Velocity Magnitude, $|U|/U_\infty$")
            cb1.locator = ticker.MaxNLocator(nbins=10, steps=[1, 2.5, 5, 10])
            cb1.formatter = ticker.FormatStrFormatter('%.2f')
            cb1.update_ticks()
            
            c2 = ax2.contourf(X, Z, -cp_f, levels=np.linspace(-1.0, -cpmin_val, 50), cmap='inferno')
            cb2 = tf_fig.colorbar(c2, ax=ax2, label=r"Pressure Coefficient, $-C_p$")
            cb2.locator = ticker.MaxNLocator(nbins=10, steps=[1, 2.5, 5, 10])
            cb2.formatter = ticker.FormatStrFormatter('%.2f')
            cb2.update_ticks()
            
            import matplotlib.path as mpath
            
            # Draw airfoil
            xf = self.last_M.foil.x[0,:]
            zf = self.last_M.foil.x[1,:]
            
            x_mask = np.copy(xf)
            z_mask = np.copy(zf)
            has_bl = False
            
            if self.bl_var.get() and self.last_M.oper.viscous and hasattr(self.last_M.vsol, 'Is') and len(self.last_M.vsol.Is) >= 2:
                if hasattr(self.last_M.post, 'ds'):
                    has_bl = True
                    Is_l = self.last_M.vsol.Is[0]
                    Is_u = self.last_M.vsol.Is[1]
                    
                    t_u = self.last_M.foil.t[:, Is_u]
                    n_u = np.array([-t_u[1, :], t_u[0, :]])
                    delta_u = self.last_M.post.ds[Is_u]
                    x_bl_u = xf[Is_u] + n_u[0, :] * delta_u
                    z_bl_u = zf[Is_u] + n_u[1, :] * delta_u
                    
                    t_l = self.last_M.foil.t[:, Is_l]
                    n_l = np.array([-t_l[1, :], t_l[0, :]])
                    delta_l = self.last_M.post.ds[Is_l]
                    x_bl_l = xf[Is_l] + n_l[0, :] * delta_l
                    z_bl_l = zf[Is_l] + n_l[1, :] * delta_l
                    
                    t_foil = self.last_M.foil.t
                    n_foil = np.array([-t_foil[1, :], t_foil[0, :]])
                    delta_foil = self.last_M.post.ds[:len(xf)]
                    x_mask = xf + n_foil[0, :] * delta_foil
                    z_mask = zf + n_foil[1, :] * delta_foil
            
            if self.streams_var.get():
                # Streamlines masking interior of effective aerodynamic body
                airfoil_path = mpath.Path(np.column_stack([x_mask, z_mask]))
                points = np.column_stack([X.flatten(), Z.flatten()])
                mask = airfoil_path.contains_points(points).reshape(X.shape)
                
                u_masked = np.ma.array(u, mask=mask)
                w_masked = np.ma.array(w, mask=mask)
                
                ax1.streamplot(X, Z, u_masked, w_masked, color='white', linewidth=0.6, density=1, arrowsize=1.0)
                ax2.streamplot(X, Z, u_masked, w_masked, color='white', linewidth=0.6, density=1, arrowsize=1.0)
                
            if has_bl:
                bl_color = self.bl_color_var.get().strip() or "blue"
                for ax in [ax1, ax2]:
                    ax.plot(x_bl_u, z_bl_u, '-', color=bl_color, linewidth=1.5, label="Boundary Layer")
                    ax.plot(x_bl_l, z_bl_l, '-', color=bl_color, linewidth=1.5)
            
            for ax in [ax1, ax2]:
                ax.fill(xf, zf, color='black', alpha=1.0)
                ax.plot(xf, zf, color='white', linewidth=0.5)
                ax.set_aspect('equal', 'box')
                ax.set_xlim([bounds[0], bounds[1]])
                ax.set_ylim([bounds[2], bounds[3]])
                ax.set_xlabel(r"$x/c$")
                ax.set_ylabel(r"$z/c$")
                
            ax1.set_title(f"Velocity Magnitude NACA {self.naca_var.get()}")
            ax2.set_title(f"Pressure Coefficient NACA {self.naca_var.get()}")
            
            canvas = FigureCanvasTkAgg(tf_fig, master=top)
            canvas.draw()
            
            toolbar = NavigationToolbar2Tk(canvas, top)
            toolbar.update()
            
            canvas.get_tk_widget().pack(fill="both", expand=True)
            lbl.destroy()
        except Exception as e:
            lbl.configure(text=f"Error evaluating field: {e}", text_color="red")

if __name__ == "__main__":
    app = NFoilApp()
    app.mainloop()
