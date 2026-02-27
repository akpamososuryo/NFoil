# NFoil

**High-performance subsonic airfoil analysis with Numba JIT acceleration.**

NFoil is a fork of [mfoil](https://websites.umich.edu/~kfid/codes.html#mfoil) by Krzysztof J. Fidkowski, rewritten with Numba JIT compilation for **~15x faster** subsonic airfoil analysis. It retains full numerical compatibility with the original while dramatically reducing solve times with the addition of a Graphical User Interface.

---

##  Performance

Complete alpha sweep benchmark (NACA 2412, 16 angles, $Re=10^6$):

| Panels | Original (s) | NFoil (s) | Speedup |
|--------|-------------|--------------|---------|
| 100    | 28.2        | **1.8**      | **15.6x** |
| 200    | 70.4        | **4.7**      | **15.0x** |
| 300    | 117.2       | **7.6**      | **15.5x** |
| 400    | 182.2       | **12.3**     | **14.8x** |

Both versions share the same algorithmic complexity of $\approx O(N^1.35)$. The speedup comes entirely from eliminating Python/SciPy overhead via JIT compilation and dense array operations.

![Scalability Benchmark](scalability_benchmark.png)

---

##  Changes from Original `mfoil`

### 1. Numba JIT-Compiled Physics Kernels

All core boundary-layer functions have been rewritten as `@njit(cache=True, fastmath=True)` kernels:

| Function | Description |
|----------|-------------|
| `get_Hk`, `get_Hs`, `get_Hss` | Shape parameter correlations |
| `get_Ret`, `get_cf`, `get_cd` | Skin friction & dissipation |
| `get_Us`, `get_Mach2` | Boundary layer edge quantities |
| `get_cttr`, `get_de` | Transition & entrainment |
| `residual_station_jit` | Station-level BL residual + Jacobian |
| `residual_transition` | Transition interval residual |
| `_calc_force_inviscid_jit` | Pressure force integration |
| `build_B_bulk`, `build_Csig_bulk` | Inviscid influence matrix assembly |
| `march_amplification` | eⁿ transition amplification |
| `TE_info_jit` | Trailing-edge geometry |
| `cosd`, `sind`, `norm2` | Utility functions |

**Key design choice:** BL parameters are passed as a `namedtuple` (`BL_Param`) rather than mutable class attributes, allowing Numba to compile the full residual evaluation chain without falling back to Python object mode.

### 2. Dense Array Assembly (Eliminating SciPy Sparse Overhead)

The #1 bottleneck in the original code was SciPy sparse matrix overhead inside the coupled Newton solver:

- **`build_glob_sys`**: Replaced `scipy.sparse.dok_matrix` allocation for `R_U` and `R_x` with dense `np.zeros` arrays. This eliminated ~70% of total runtime that was spent in DOK indexing, `_validate_indices`, `__setitem__`, and COO conversion.

- **`solve_glob`**: Replaced `scipy.sparse.lil_matrix` assembly + `scipy.sparse.linalg.spsolve` with dense `np.zeros` array + `np.linalg.solve`. Includes a `LinAlgError` fallback to `np.linalg.lstsq` for near-singular Jacobians at extreme operating conditions.

### 3. Vectorized Sensitivity Assembly

`calc_ue_m` (inviscid edge velocity sensitivity matrix) was rebuilt with parallelised Numba kernels:

- `build_B_bulk`: Vectorized source-panel influence coefficient assembly over all panels simultaneously, replacing an O(N²) Python loop.
- `build_Csig_bulk`: Vectorized wake source-influence assembly.

### 4. Robust Solver Improvements

- **Singular matrix fallback**: `solve_glob` catches `np.linalg.LinAlgError` and falls back to `np.linalg.lstsq` for near-singular Jacobians (common at very low Re or high α).
- **Bug fix**: Fixed missing `.shape` in sparse matrix comparison at line 2057 of the original code (`R_x == (3*Nsys, Nsys)` → `R_x.shape == (3*Nsys, Nsys)`).

### 5. Code Cleanup

- Consolidated duplicate imports (3× `from scipy import sparse`, 2× `import numpy as np`, etc.) into a single clean import block.
- Removed dead `build_glob_sys_jit` stub (34 lines of placeholder code).
- Restored complete copyright header that was fragmented by earlier edits.

---

##  Interactive GUI

NFoil includes a full-featured GUI (`gui.py`) built with CustomTkinter:

- **Real-time analysis**: Single-point solve with live Cp distribution, airfoil geometry (with physical BL thickness δ), and aerodynamic coefficients.
- **Robust polar sweeps**: Multi-angle α sweeps with automatic continuation past non-converged points. If a point fails, the solver retries with a cold BL restart; if it still fails, it skips that point and continues the sweep. All computed cases from multiple runs are stored with descriptive labels (NACA/Re/α/M) in a persistent dropdown.
- **Target Cl (Inverse Mode)**: A dedicated toggle and input field to find the angle of attack for a specific lift coefficient ($C_L$), integrated into both single-point solves and polar sweeps.
- **Load/Unload Airfoil**: Directly load custom coordinates from `.dat` or `.txt` files (TE-to-TE, CCW/CW). Includes an "Unload" button to instantly revert to the parameterized NACA generator.
- **Real-time analysis**: Single-point solve with live Cp distribution, airfoil geometry (with physical BL thickness δ), and aerodynamic coefficients.
- **Lift & drag polars**: Overlay polars from multiple Reynolds numbers for comparison.
- **BL properties tab**: Skin friction (Cf), displacement thickness (δ*), momentum thickness (θ), and shape parameter (Hk) across upper, lower, and wake surfaces.
- **Flap deflection**: Geometric trailing-edge flap with configurable hinge point and deflection angle.
- **GPU flow fields** (optional): Real-time velocity/pressure contour maps via [Taichi](https://www.taichi-lang.org/) (`taichi_fields.py`), leveraging Metal/CUDA GPUs.
- **ASCII export**: Cp curves, geometry, skin friction arrays, and polar data.

```bash
python gui.py
```

---

##  Dependencies

```
numpy
scipy
matplotlib
numba
customtkinter   # for GUI only
taichi           # optional, for GPU flow fields
```

##  Quick Start

```python
import nfoil as nf

# Create and solve
N = nf.nfoil(naca='2412', npanel=200)
N.setoper(alpha=5, Re=1e6, Ma=0.3)
N.solve() 

# Results
print(f"Cl = {N.post.cl:.4f}")
print(f"Cd = {N.post.cd:.6f}")
print(f"Cm = {N.post.cm:.4f}")
```

##  License

MIT License — Copyright (C) 2026 Cayetano Martínez-Muriel. 

##  Acknowledgments

Based on [mfoil](https://websites.umich.edu/~kfid/codes.html#mfoil) by Krzysztof J. Fidkowski.
