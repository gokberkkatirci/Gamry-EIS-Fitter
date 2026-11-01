"""EIS fitting, validation, DRT, plotting, and interactive workflow.

Circuit: L + Rs + sum(R || CPE). See docs/methods.md for conventions.
"""

from . import __version__

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from scipy.optimize import differential_evolution, minimize
import os
import sys
import csv
from datetime import datetime

# ---- Global font: Arial for all plots ----
plt.rcParams["font.family"] = "DejaVu Sans"

# ============================================================
# SECTION 1: Terminal Colors & Helpers
# ============================================================

class C:
    """ANSI color codes for terminal output."""
    HEADER = "\033[95m"
    BLUE   = "\033[94m"
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    UNDERLINE = "\033[4m"
    RESET  = "\033[0m"


def banner(text, color=C.CYAN, width=70):
    """Print a colored banner."""
    print(f"\n{color}{C.BOLD}{'=' * width}")
    print(f"  {text}")
    print(f"{'=' * width}{C.RESET}")


def section(text, color=C.YELLOW):
    """Print a colored section divider."""
    print(f"\n{color}{C.BOLD}{'-' * 60}")
    print(f"  {text}")
    print(f"{'-' * 60}{C.RESET}")


def info(text):
    print(f"  {C.CYAN}{text}{C.RESET}")


def success(text):
    print(f"  {C.GREEN}{C.BOLD}{text}{C.RESET}")


def warn(text):
    print(f"  {C.YELLOW}{text}{C.RESET}")


def error(text):
    print(f"  {C.RED}{C.BOLD}{text}{C.RESET}")


# ============================================================
# SECTION 2: Engineering Notation Helpers
# ============================================================

_ENG_PREFIXES = {
    -15: "f", -12: "p", -9: "n", -6: "\u03bc", -3: "m",
    0: "", 3: "k", 6: "M", 9: "G", 12: "T",
}


def eng_str(value, unit="", sigfigs=3):
    """
    Format a number in engineering notation.
    Examples: 4700  -> '4.70 k',  0.001 -> '1.00 m',  1e6 -> '1.00 M'
    """
    if value == 0:
        return f"0 {unit}".strip()
    sign = ""
    if value < 0:
        sign = "-"
        value = abs(value)
    exp = int(np.floor(np.log10(value) / 3) * 3)
    exp = max(-15, min(12, exp))  # clamp to known prefixes
    mantissa = value / 10**exp
    prefix = _ENG_PREFIXES.get(exp, f"e{exp}")
    fmt = f"{{:.{max(0, sigfigs - 1)}f}}"
    return f"{sign}{fmt.format(mantissa)} {prefix}{unit}".strip()


def apply_eng_format(ax, which="both"):
    """Apply engineering-notation tick formatter to an axis."""
    fmt = mticker.EngFormatter(places=1)
    if which in ("x", "both"):
        ax.xaxis.set_major_formatter(fmt)
    if which in ("y", "both"):
        ax.yaxis.set_major_formatter(fmt)


# ============================================================
# SECTION 3: File Discovery & Selection
# ============================================================

def find_all_dta_files(root_dir="."):
    """Recursively find all .DTA files under root_dir."""
    dta_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.upper().endswith(".DTA"):
                dta_files.append(os.path.join(dirpath, f))
    dta_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
    return dta_files


def display_file_list(dta_files):
    """Display a numbered list of DTA files."""
    banner("AVAILABLE .DTA FILES", C.BLUE)
    for i, fpath in enumerate(dta_files, start=1):
        num_color = C.GREEN if i % 2 == 0 else C.CYAN
        print(f"  {num_color}[{i:3d}]{C.RESET}  {os.path.relpath(fpath, '.')}")
    print(f"{C.BLUE}{C.BOLD}{'=' * 70}{C.RESET}")


def select_dta_file(dta_files):
    """Prompt user to pick a file by number."""
    while True:
        try:
            choice = int(input(
                f"\n  {C.BOLD}Enter file number (1-{len(dta_files)}): {C.RESET}"
            ))
            if 1 <= choice <= len(dta_files):
                return dta_files[choice - 1]
            warn("Invalid number, try again.")
        except ValueError:
            warn("Please enter a valid integer.")
        except EOFError:
            error("stdin not available.")
            print("  Run from terminal:  eis-fit")
            sys.exit(1)


# ============================================================
# SECTION 4: DTA File Reader
# ============================================================

def read_dta_file(filepath):
    """
    Read EIS data from a Gamry .DTA file.
    Returns: freq (Hz), Zreal (Ohm), Zimag (Ohm) as numpy arrays.
    """
    with open(filepath, "r", encoding="latin-1") as handle:
        lines = handle.readlines()
    table = next((i for i, line in enumerate(lines)
                  if "ZCURVE" in line and "TABLE" in line), None)
    if table is None:
        raise ValueError("ZCURVE TABLE not found.")
    if len(lines) <= table + 3:
        raise ValueError("Incomplete ZCURVE table: header, units, and rows required.")
    headers = lines[table + 1].strip().split("\t")
    required = ("Freq", "Zreal", "Zimag")
    if any(headers.count(name) != 1 for name in required):
        raise ValueError("Required columns Freq, Zreal, Zimag must occur once.")
    columns = [headers.index(name) for name in required]
    rows = []
    for line_number, line in enumerate(lines[table + 3:], table + 4):
        if not line.strip():
            break
        fields = line.strip().split("\t")
        try:
            row = [float(fields[index]) for index in columns]
        except (ValueError, IndexError) as exc:
            raise ValueError(
                f"Malformed ZCURVE row at line {line_number}."
            ) from exc
        rows.append(row)
    values = np.asarray(rows, dtype=float)
    if not rows:
        raise ValueError("ZCURVE table has no data rows.")
    freq, zreal, zimag = validate_spectrum(*values.T)
    info(f"Read {len(freq)} data points from file.")
    info(f"Frequency range: {eng_str(freq.min(), 'Hz')} - "
         f"{eng_str(freq.max(), 'Hz')}")
    return freq, zreal, zimag


def validate_spectrum(freq, zreal, zimag, min_points=5):
    """Validate spectra without changing order, units, or sign convention."""
    arrays = tuple(np.asarray(value, dtype=float)
                   for value in (freq, zreal, zimag))
    if any(value.ndim != 1 for value in arrays):
        raise ValueError("Spectrum arrays must be one-dimensional.")
    if len({len(value) for value in arrays}) != 1:
        raise ValueError("Frequency and impedance arrays must have equal lengths.")
    if len(arrays[0]) < min_points:
        raise ValueError(f"At least {min_points} usable points are required.")
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("Frequency and impedance values must be finite.")
    if np.any(arrays[0] <= 0):
        raise ValueError("Frequencies must be positive (Hz).")
    return arrays


# ============================================================
# SECTION 5: Data Filtering (remove inductive region)
# ============================================================

def filter_capacitive_region(freq, zreal, zimag):
    """
    Keep only the capacitive region of the Nyquist plot.
    Gamry convention: Zimag < 0 is capacitive.
    """
    freq, zreal, zimag = validate_spectrum(freq, zreal, zimag)
    n = len(freq)
    low_freq_half = slice(n // 2, n)

    if np.mean(zimag[low_freq_half]) < 0:
        mask = zimag < 0
        sign_convention = "negative"
    else:
        mask = zimag > 0
        sign_convention = "positive"

    freq_f = freq[mask]
    zreal_f = zreal[mask]
    zimag_f = zimag[mask]

    info(f"Sign convention: capacitive Zimag is {sign_convention}.")
    info(f"Filtered to {len(freq_f)} capacitive points "
         f"(removed {n - len(freq_f)} inductive).")
    validate_spectrum(freq_f, zreal_f, zimag_f)
    return freq_f, zreal_f, zimag_f, sign_convention


# ============================================================
# SECTION 6: Impedance Model  L + Rs + N x (R || CPE)
# ============================================================

def Z_inductor(omega, L):
    return 1j * omega * L


def Z_cpe(omega, Q, alpha):
    return 1.0 / (Q * (1j * omega) ** alpha)


def Z_voigt(omega, R, Q, alpha):
    Zcpe = Z_cpe(omega, Q, alpha)
    return (R * Zcpe) / (R + Zcpe)


def Z_model(omega, params, n_voigt):
    """
    Full model: L + Rs + Sum_i(R_i || CPE_i)
    params = [L, Rs, R1, Q1, a1, ..., Rn, Qn, an]
    """
    L = params[0]
    Rs = params[1]
    Z = Z_inductor(omega, L) + Rs
    for i in range(n_voigt):
        idx = 2 + 3 * i
        Z += Z_voigt(omega, params[idx], params[idx + 1], params[idx + 2])
    return Z


def Z_model_components(omega, params, n_voigt):
    """Return total Z and each Voigt element's own impedance."""
    L = params[0]
    Rs = params[1]
    Z_base = Z_inductor(omega, L) + Rs
    Z_total = Z_base.copy()
    voigt_list = []
    for i in range(n_voigt):
        idx = 2 + 3 * i
        Zv = Z_voigt(omega, params[idx], params[idx + 1], params[idx + 2])
        voigt_list.append(Zv)
        Z_total += Zv
    return Z_total, Z_base, voigt_list


# ============================================================
# SECTION 7: Fitting Engine  (tau-parameterisation, multi-restart)
# ============================================================
#
# Internal parameterisation uses time constant (tau) during
# Differential Evolution.  Unordered time constants are heavily
# penalised to break permutation symmetry.  Standard Q parameters
# are returned at the end so the rest of the program is unchanged.
#
# Overlapping arcs are fully supported: the ordering constraint
# tau_1 <= tau_2 <= ... only removes duplicate permutations; it
# does NOT prevent time constants from being arbitrarily close.
# ============================================================

def build_bounds_tau(n_voigt, zreal, zimag, freq):
    """Build parameter bounds for L + Rs + N x (R || CPE) using tau.

    CPE alpha bounds are 0.7 to 1.0. The tau search extends three
    decades beyond the reciprocal measured angular-frequency range.
    """
    Rs_est = np.min(np.abs(zreal))
    R_total = np.ptp(zreal)
    if R_total <= 0:
        R_total = 1.0

    # Estimate reasonable time constant limits from frequency range
    # Extended by 1000x (instead of 100x) to capture convoluted arcs
    # whose merged peak may shift outside the obvious frequency band.
    tau_min = 1.0 / (2.0 * np.pi * freq.max() * 1000.0)
    tau_max = 1.0 / (2.0 * np.pi * freq.min() / 1000.0)

    bounds = [
        (0, 1e-3),                                         # L
        (max(Rs_est * 0.3, 0), max(Rs_est * 2.0, 0.01)),   # Rs
    ]
    for _ in range(n_voigt):
        bounds.append((1e-8, R_total * 3.0))               # Ri
        bounds.append((tau_min, tau_max))                  # tau_i
        bounds.append((0.7, 1.0))                          # alpha_i
    return bounds


def Z_model_tau(omega, params, n_voigt):
    """ZARC model parameterised directly by time constant tau."""
    L = params[0]
    Rs = params[1]
    Z = 1j * omega * L + Rs
    for i in range(n_voigt):
        idx = 2 + 3 * i
        R = params[idx]
        tau = params[idx + 1]
        alpha = params[idx + 2]
        Z += R / (1.0 + (1j * omega * tau)**alpha)
    return Z


def objective_tau(params, omega, zreal_exp, zimag_exp, n_voigt,
                  penalty_mode="none"):
    """Modulus-weighted least-squares objective with configurable ordering.

    penalty_mode controls how out-of-order time constants are handled:
      "none"  – no ordering constraint at all (best for convoluted arcs,
                lets the optimizer freely explore overlapping tau regions).
      "soft"  – adds a smooth quadratic penalty that gently nudges taus
                toward ascending order without creating a cliff edge.
                This preserves gradients so L-BFGS-B polish works well.
      "hard"  – legacy behaviour: returns 1e12 for out-of-order taus.
                Breaks DE exploration near equal taus; NOT recommended
                for convoluted/overlapping arcs.

    The default applies no ordering penalty.
    The soft penalty uses a scaling factor relative to the geometric
    mean of all taus so it adapts to any order of magnitude.
    """
    if penalty_mode == "hard" and n_voigt > 1:
        taus = [params[3 + 3*i] for i in range(n_voigt)]
        if not all(taus[i] <= taus[i+1] for i in range(n_voigt - 1)):
            return 1e12

    Z_calc = Z_model_tau(omega, params, n_voigt)
    dr = Z_calc.real - zreal_exp
    di = Z_calc.imag - zimag_exp
    w = 1.0 / (np.sqrt(zreal_exp**2 + zimag_exp**2) + 1e-12)
    cost = np.sum(w**2 * (dr**2 + di**2))

    if penalty_mode == "soft" and n_voigt > 1:
        taus = [params[3 + 3*i] for i in range(n_voigt)]
        # Geometric mean of taus as a natural scale factor
        tau_geomean = np.exp(np.mean(np.log(np.clip(taus, 1e-30, None))))
        lam = 1.0  # soft penalty weight (unit-less after normalisation)
        for i in range(n_voigt - 1):
            diff = taus[i] - taus[i + 1]       # positive when disordered
            if diff > 0:
                cost += lam * (diff / tau_geomean) ** 2

    return cost


def _convert_tau_to_Q(params_tau, n_voigt):
    """Convert tau-parameterized params to standard Q-parameterization.

    In-place converts tau → Q for each Voigt element:
      Q = tau^alpha / R  (exact inverse of Z_ZARC tau-parameterisation)
    """
    params = params_tau.copy()
    for i in range(n_voigt):
        idx = 2 + 3 * i
        R = params_tau[idx]
        tau = params_tau[idx + 1]
        alpha = params_tau[idx + 2]
        Q = (tau**alpha) / R if R > 1e-15 else 1e-12
        params[idx + 1] = Q
    return params


def _compute_fit_metrics(zreal, zimag, Z_fit, best_cost, n_params):
    """Compute RMSE, R², ss_res from fit residuals."""
    ss_res = np.sum((zreal - Z_fit.real)**2 + (zimag - Z_fit.imag)**2)
    ss_tot = np.sum((zreal - zreal.mean())**2 + (zimag - zimag.mean())**2)
    rmse = np.sqrt(ss_res / len(zreal))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"cost": best_cost, "rmse": rmse, "r2": r2,
            "n_params": n_params, "ss_res": ss_res}


def fit_eis(freq, zreal, zimag, n_voigt,
            maxiter=400, popsize=25, n_restarts=5):
    """Fit EIS data with L + Rs + n_voigt x (R || CPE).

    Multi-restart differential evolution with L-BFGS-B polish.
    Uses tau-parameterization internally to isolate overlapping arcs.

    Three-strategy restart schedule for overlapping arcs:
      Phase A (~ 40 % restarts): "soft" ordering penalty during DE.
          Gently guides taus toward ascending order while keeping the
          landscape smooth so DE can still find near-overlapping taus.
      Phase B (~ 40 % restarts): "none" — completely free exploration.
          No ordering at all; the optimizer can place taus anywhere,
          which is essential when arcs are strongly convoluted.
      Phase C (~ 20 % restarts): "hard" ordering during DE.
          Retained for well-separated spectra where the strict
          constraint breaks permutation degeneracy quickly.

    All phases polish with L-BFGS-B using "soft" penalty (smooth
    gradients, no cliff edge).  The overall best across all phases
    and restarts is returned.
    """
    freq, zreal, zimag = validate_spectrum(freq, zreal, zimag)
    omega = 2.0 * np.pi * freq
    bounds = build_bounds_tau(n_voigt, zreal, zimag, freq)
    n_params = 2 + 3 * n_voigt

    # Allocate restarts across the three penalty strategies
    n_soft = max(1, int(round(n_restarts * 0.4)))
    n_free = max(1, int(round(n_restarts * 0.4)))
    n_hard = max(1, n_restarts - n_soft - n_free)
    total_restarts = n_soft + n_free + n_hard

    info(f"Fitting {n_voigt} Voigt element(s) "
         f"({n_params} params, {total_restarts} restarts: "
         f"{n_soft} soft + {n_free} free + {n_hard} hard) ...")

    # Build schedule: list of (penalty_mode_for_DE, label)
    schedule = (
        [("soft",  "soft")]  * n_soft +
        [("none",  "free")]  * n_free +
        [("hard",  "hard")]  * n_hard
    )

    best_params_tau = None
    best_cost = np.inf

    for restart, (de_mode, label) in enumerate(schedule):
        seed = 42 + restart * 7  # deterministic but different seeds

        try:
            result = differential_evolution(
                objective_tau, bounds,
                args=(omega, zreal, zimag, n_voigt, de_mode),
                maxiter=maxiter, popsize=popsize, tol=1e-8,
                seed=seed, disp=False,
                workers=-1, updating="deferred",
            )
        except Exception:
            result = differential_evolution(
                objective_tau, bounds,
                args=(omega, zreal, zimag, n_voigt, de_mode),
                maxiter=maxiter, popsize=popsize, tol=1e-8,
                seed=seed, disp=False, workers=1,
            )

        # Polish with L-BFGS-B using "soft" penalty (smooth gradients)
        result2 = minimize(
            objective_tau, result.x,
            args=(omega, zreal, zimag, n_voigt, "soft"),
            method="L-BFGS-B", bounds=bounds,
        )

        # Also try polish with NO penalty — important for convoluted arcs
        result3 = minimize(
            objective_tau, result.x,
            args=(omega, zreal, zimag, n_voigt, "none"),
            method="L-BFGS-B", bounds=bounds,
        )

        # Pick the best among DE result and both polishes
        candidates = [
            (result.fun, result.x),
            (result2.fun, result2.x),
            (result3.fun, result3.x),
        ]
        cost, params = min(candidates, key=lambda c: c[0])

        is_new_best = cost < best_cost
        if is_new_best:
            best_cost = cost
            best_params_tau = params.copy()

        print(f"    restart {restart + 1}/{total_restarts} [{label}]: "
              f"cost = {cost:.4e}"
              + (" *best*" if is_new_best else ""))

    # Convert tau -> Q, sort by tau, compute metrics
    best_params = _convert_tau_to_Q(best_params_tau, n_voigt)
    best_params = sort_voigt_params(best_params, n_voigt)
    Z_fit, Z_base, voigt_list = Z_model_components(
        omega, best_params, n_voigt)
    metrics = _compute_fit_metrics(zreal, zimag, Z_fit, best_cost, n_params)
    info(f"Best: cost = {best_cost:.4e},  "
         f"RMSE = {metrics['rmse']:.6f} Ohm,  R^2 = {metrics['r2']:.6f}")
    return best_params, Z_fit, Z_base, voigt_list, metrics


# ============================================================
# SECTION 8: Auto-scan best number of Voigt elements
# ============================================================

def auto_scan_voigt(freq, zreal, zimag, n_min=1, n_max=10):
    """Fit n_min..n_max Voigt elements.  Pick best by BIC."""
    N = len(freq)
    results = {}
    banner(f"AUTO-SCAN (BIC): fitting {n_min} to {n_max} Voigt elements",
           C.HEADER)

    for n in range(n_min, n_max + 1):
        params, Z_fit, Z_base, voigt_list, metrics = fit_eis(
            freq, zreal, zimag, n,
            maxiter=300, popsize=20, n_restarts=3)
        k = metrics["n_params"]
        rss = metrics["ss_res"]
        bic = N * np.log(rss / N + 1e-30) + k * np.log(N)
        metrics["bic"] = bic
        results[n] = {"params": params, "Z_fit": Z_fit, "Z_base": Z_base,
                       "voigt_list": voigt_list, "metrics": metrics}
        print(f"    n={n:2d}  |  BIC = {bic:+.2f}  |  "
              f"R^2 = {metrics['r2']:.6f}  |  RMSE = {metrics['rmse']:.6f}")

    best_n = min(results, key=lambda k: results[k]["metrics"]["bic"])
    success(f">>> Best model by BIC: {best_n} Voigt elements <<<")
    return best_n, results[best_n], results


def auto_best_r2(freq, zreal, zimag, n_min=1, n_max=10):
    """Fit n_min..n_max Voigt elements.  Pick best by highest R^2."""
    results = {}
    banner(f"AUTO (best R^2): fitting {n_min} to {n_max} Voigt elements",
           C.HEADER)

    for n in range(n_min, n_max + 1):
        params, Z_fit, Z_base, voigt_list, metrics = fit_eis(
            freq, zreal, zimag, n,
            maxiter=400, popsize=25, n_restarts=3)
        results[n] = {"params": params, "Z_fit": Z_fit, "Z_base": Z_base,
                       "voigt_list": voigt_list, "metrics": metrics}
        print(f"    n={n:2d}  |  R^2 = {metrics['r2']:.8f}  |  "
              f"RMSE = {metrics['rmse']:.6f}")

    best_n = max(results, key=lambda k: results[k]["metrics"]["r2"])
    best_r2 = results[best_n]["metrics"]["r2"]
    success(f">>> Best model by R^2: {best_n} Voigt elements "
            f"(R^2 = {best_r2:.8f}) <<<")
    return best_n, results[best_n], results


# ============================================================
# SECTION 9: Pretty-Print & Sort Parameters
# ============================================================

def sort_voigt_params(params, n_voigt):
    """Sort Voigt elements by ZARC time constant tau (ascending).

    The ZARC element Z = R / (1 + (jωτ)^α) with
    Q = τ^α / R  gives the correct inverse  τ = (R·Q)^(1/α).
    The simpler relation τ = R·Q holds only for α = 1.
    """
    voigt_blocks = []
    for i in range(n_voigt):
        idx = 2 + 3 * i
        R, Q, a = params[idx], params[idx + 1], params[idx + 2]
        # Correct ZARC time constant: τ = (R·Q)^(1/α)
        rq = R * Q
        tau = (rq ** (1.0 / a)) if (rq > 0 and a > 0) else rq
        voigt_blocks.append((tau, R, Q, a))
    voigt_blocks.sort(key=lambda x: x[0])
    new_params = list(params[:2])
    for _, R, Q, a in voigt_blocks:
        new_params.extend([R, Q, a])
    return np.array(new_params)


def format_params(params, n_voigt):
    """Format parameters for display, including characteristic freq.

    Tau is computed as (R·Q)^(1/α) — the correct
    ZARC time constant — instead of the incorrect R·Q.
    """
    lines = [
        f"  L   = {params[0]:.4e} H",
        f"  Rs  = {params[1]:.6f} Ohm",
    ]
    for i in range(n_voigt):
        idx = 2 + 3 * i
        R, Q, a = params[idx], params[idx + 1], params[idx + 2]
        # Correct ZARC time constant: τ = (R·Q)^(1/α)
        rq = R * Q
        tau = (rq ** (1.0 / a)) if (rq > 0 and a > 0) else rq
        # Characteristic frequency where the arc peaks
        f_char = 1.0 / (2.0 * np.pi * tau) if tau > 0 else np.inf
        lines.append(
            f"  R{i+1}  = {R:.6f} Ohm,  Q{i+1} = {Q:.4e},  "
            f"a{i+1} = {a:.4f}  "
            f"(tau ~ {tau:.4e} s, f_c ~ {eng_str(f_char, 'Hz')})")
    return "\n".join(lines)


# ============================================================
# SECTION 10: Kramers-Kronig Validation (Extended Lin-KK)
# ============================================================
#
# Extended Lin-KK method fits the data with a linear model:
#   Z_KK(w) = R_inf + Sum_{k=1}^{M} R_k / (1 + jw*tau_k) + jw*L
#
# where tau_k are FIXED time constants log-spaced across an EXTENDED
# range (1 decade beyond measurement frequencies on each side).
# The inductance term L captures high-frequency inductive behavior.
# Since the model is linear in R_inf, R_k and L, all coefficients
# are found via ordinary least-squares.
#
# Tau-range extension is used to reduce frequency-edge effects.
#
# The number of RC elements M is auto-selected by scanning from
# small M upward until the mean relative residual drops below a
# threshold, with an overfitting guard (M <= N_data).
#
# Inspect residual structure as well as the numerical threshold.
# A residual test alone does not establish measurement validity.
# ============================================================

def _kk_build_matrix(omega, taus):
    """
    Build the 2N x (M+2) design matrix for the extended Lin-KK fit.

    Columns: [R_inf, R_1, R_2, ..., R_M, L]
    Rows:    [Zreal(w_1)..Zreal(w_N), Zimag(w_1)..Zimag(w_N)]

    The inductance column captures high-frequency inductive behavior
    that pure RC elements cannot represent.
    """
    N = len(omega)
    M = len(taus)
    A = np.zeros((2 * N, M + 2))

    # R_inf: contributes +1 to real part, 0 to imaginary
    A[:N, 0] = 1.0

    for k in range(M):
        wt2 = (omega * taus[k]) ** 2
        denom = 1.0 + wt2
        A[:N, k + 1] = 1.0 / denom                       # real
        A[N:, k + 1] = -omega * taus[k] / denom           # imag

    # Inductance: Z_L = j*omega*L  =>  real = 0, imag = omega*L
    A[N:, M + 1] = omega

    return A


def _kk_fit_single(omega, zreal, zimag, n_rc):
    """
    Perform a single extended Lin-KK fit with n_rc RC elements.

    Improvements over basic Lin-KK:
      - Tau range extended by 1 decade each side to eliminate boundary
        artefacts. This implementation uses its own residual stop rule;
        see docs/methods.md.
      - Inductance term included so the linear model can capture the
        inductive rise at high frequencies without distorting RC residuals.

    Returns: coeffs, taus, Z_kk, res_real_pct, res_imag_pct, mean_res_pct
             coeffs = [R_inf, R_1..R_M, L]
    """
    N = len(omega)
    f = omega / (2.0 * np.pi)
    tau_min = 1.0 / (2.0 * np.pi * f.max())
    tau_max = 1.0 / (2.0 * np.pi * f.min())

    # Extend tau range by 1 decade each side to reduce boundary errors
    tau_min_ext = tau_min / 10.0
    tau_max_ext = tau_max * 10.0
    taus = np.logspace(np.log10(tau_min_ext), np.log10(tau_max_ext), n_rc)

    A = _kk_build_matrix(omega, taus)
    b = np.concatenate([zreal, zimag])

    # Solve with least-squares (no regularization needed for M <= N)
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # Reconstruct KK impedance: R_inf + sum(R_k/(1+jw*tau_k)) + jw*L
    Z_kk = np.full(N, x[0], dtype=complex)
    for k in range(n_rc):
        Z_kk += x[k + 1] / (1.0 + 1j * omega * taus[k])
    Z_kk += 1j * omega * x[n_rc + 1]   # inductance term

    # Relative residuals (% of |Z| at each frequency)
    Z_mag = np.sqrt(zreal**2 + zimag**2)
    Z_mag_safe = np.where(Z_mag > 1e-15, Z_mag, 1e-15)
    res_real_pct = 100.0 * (zreal - Z_kk.real) / Z_mag_safe
    res_imag_pct = 100.0 * (zimag - Z_kk.imag) / Z_mag_safe

    mean_res_pct = np.mean(np.sqrt(res_real_pct**2 + res_imag_pct**2))

    return x, taus, Z_kk, res_real_pct, res_imag_pct, mean_res_pct


def kramers_kronig_test(freq, zreal, zimag):
    """
    Auto-fit Kramers-Kronig test using the Lin-KK method.

    Scans M (number of RC elements) from 2 upward.  Picks the smallest M
    where the mean relative residual < 1 %, with an upper bound of
    M = N_data to avoid overfitting.

    Returns a dict with:
      n_rc, R_coeffs, taus, Z_kk, res_real_pct, res_imag_pct,
      mean_res_pct, passed  (True if mean residual < 1 %)
    """
    freq, zreal, zimag = validate_spectrum(freq, zreal, zimag)
    omega = 2.0 * np.pi * freq
    N = len(freq)
    M_max = N  # upper bound

    banner("KRAMERS-KRONIG VALIDATION (Lin-KK)", C.HEADER)
    info(f"Scanning M = 2 .. {M_max} RC elements ...")

    best = None
    # threshold aligned with pass criterion (1.0%) — stop as soon as KK is satisfied
    threshold = 1.0
    PLATEAU_WINDOW = 5
    PLATEAU_DELTA = 0.05   # stop if improvement < 0.05% over last 5 steps
    scan_results = []
    recent_res = []

    for m in range(2, M_max + 1):
        x, taus, Z_kk, rr, ri, mean_r = _kk_fit_single(
            omega, zreal, zimag, m)
        scan_results.append((m, mean_r))

        if best is None or mean_r < best["mean_res_pct"]:
            best = {
                "n_rc": m, "R_coeffs": x, "taus": taus, "Z_kk": Z_kk,
                "res_real_pct": rr, "res_imag_pct": ri,
                "mean_res_pct": mean_r,
            }

        # Print progress every few steps
        if m <= 10 or m % 5 == 0 or m == M_max:
            print(f"    M={m:3d}  |  mean |res| = {mean_r:.4f} %")

        # Early exit: threshold met
        if mean_r < threshold and m >= 3:
            break
        # Early exit: residual has plateaued and best already passes
        recent_res.append(mean_r)
        if len(recent_res) >= PLATEAU_WINDOW:
            improvement = recent_res[-PLATEAU_WINDOW] - min(recent_res[-PLATEAU_WINDOW:])
            if best["mean_res_pct"] < threshold and improvement < PLATEAU_DELTA:
                break

    # Determine pass/fail
    passed = best["mean_res_pct"] < 1.0
    best["passed"] = passed
    status = "PASS" if passed else "FAIL"
    status_color = C.GREEN if passed else C.RED

    print()
    print(f"  {status_color}{C.BOLD}KK Result: {status}  "
          f"(M = {best['n_rc']} RC elements, "
          f"mean |res| = {best['mean_res_pct']:.4f} %){C.RESET}")

    if passed:
        success("KK residual criterion met; inspect residuals and measurement assumptions.")
    else:
        warn("Data may NOT be fully KK-compliant. Check residual plot.")

    return best


def format_kk_results(kk):
    """Format KK results for text output."""
    L_kk = kk['R_coeffs'][kk['n_rc'] + 1]
    lines = [
        f"  KK Method   : Lin-KK (extended tau + inductance)",
        f"  RC elements : {kk['n_rc']}",
        f"  R_inf       : {kk['R_coeffs'][0]:.6f} Ohm",
        f"  L_kk        : {L_kk:.6e} H",
        f"  Mean |res|  : {kk['mean_res_pct']:.4f} %",
        f"  Status      : {'PASS' if kk['passed'] else 'FAIL'}",
    ]
    return "\n".join(lines)


# ============================================================
# SECTION 11: Interactive Frequency Range Selection
# ============================================================

def _nyquist_sign(zimag, sign_conv):
    """Return y-values for Nyquist: -Zimag for negative convention."""
    return -zimag if sign_conv == "negative" else zimag


def select_frequency_range(freq, zreal, zimag, sign_conv, title_extra=""):
    """
    Show Nyquist plot; user clicks 2 points to define the frequency range.
    Frequency annotations use engineering notation.
    Returns boolean mask of selected indices.
    """
    y_nq = _nyquist_sign(zimag, sign_conv)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(zreal, y_nq, "o", ms=6, color="royalblue", picker=5)
    ax.set_xlabel("Z' (Ohm)", fontsize=12)
    ylabel = "-Z'' (Ohm)" if sign_conv == "negative" else "Z'' (Ohm)"
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(
        f"Click 2 points to select frequency range\n{title_extra}",
        fontsize=13, fontweight="bold")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3, ls="--")

    # Engineering-format axis ticks
    apply_eng_format(ax, "both")

    # Annotate every Nth point with freq in engineering notation
    step = max(1, len(freq) // 15)
    for i in range(0, len(freq), step):
        ax.annotate(
            eng_str(freq[i], "Hz"), (zreal[i], y_nq[i]),
            fontsize=7.5, color="dimgray", alpha=0.85,
            textcoords="offset points", xytext=(5, 5))

    instruction_text = ax.text(
        0.5, 0.02,
        "Click on the START point of your region of interest",
        transform=ax.transAxes, ha="center", fontsize=11,
        fontweight="bold", color="red",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                  alpha=0.9))

    fig.tight_layout()

    clicked_points = []

    def find_nearest(cx, cy):
        x_range = max(zreal.max() - zreal.min(), 1e-12)
        y_range = max(y_nq.max() - y_nq.min(), 1e-12)
        dist = ((zreal - cx) / x_range)**2 + ((y_nq - cy) / y_range)**2
        return np.argmin(dist)

    def on_click(event):
        if event.inaxes != ax or event.button != 1:
            return
        if len(clicked_points) >= 2:
            return

        idx = find_nearest(event.xdata, event.ydata)
        clicked_points.append(idx)

        color = "red" if len(clicked_points) == 1 else "green"
        label_txt = "START" if len(clicked_points) == 1 else "END"
        ax.plot(zreal[idx], y_nq[idx], "X", ms=15, color=color,
                markeredgecolor="black", markeredgewidth=1.5,
                label=f"{label_txt}: {eng_str(freq[idx], 'Hz')}",
                zorder=10)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.8)

        if len(clicked_points) == 1:
            instruction_text.set_text(
                "Now click on the END point of your region")
            instruction_text.set_color("green")
            fig.canvas.draw_idle()
        elif len(clicked_points) == 2:
            idx1, idx2 = sorted(clicked_points)
            ax.plot(zreal[idx1:idx2+1], y_nq[idx1:idx2+1],
                    "o", ms=8, mfc="none", mec="lime", mew=2,
                    label="Selected region", zorder=6)
            instruction_text.set_text(
                "Region selected! Close this window to continue.")
            instruction_text.set_color("darkgreen")
            ax.legend(fontsize=8, loc="upper right", framealpha=0.8)
            fig.canvas.draw_idle()

    cid = fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show(block=True)
    fig.canvas.mpl_disconnect(cid)
    plt.close(fig)

    if len(clicked_points) < 2:
        warn("Less than 2 points clicked -- using all data.")
        return np.ones(len(freq), dtype=bool)

    idx1, idx2 = sorted(clicked_points)
    mask = np.zeros(len(freq), dtype=bool)
    mask[idx1:idx2+1] = True

    f_lo, f_hi = freq[mask].min(), freq[mask].max()
    success(f"Selected {mask.sum()} points: "
            f"{eng_str(f_lo, 'Hz')} - {eng_str(f_hi, 'Hz')}")
    return mask


# ============================================================
# SECTION 12: Plotting  (overlapping arcs)
# ============================================================
#
# Individual arcs are drawn using a *characteristic-
# frequency offset* so that convoluted (overlapping) arcs appear
# correctly on the Nyquist plot.
#
# For each Voigt element k (sorted by tau, smallest first):
#   offset_k = Rs + SUM_{j<k} Re( Z_j(omega_char_k) )
#
# where omega_char_k = 1/tau_k is the characteristic angular
# frequency of element k.  This evaluates how much each preceding
# arc has relaxed AT THE FREQUENCY WHERE ARC k PEAKS.
#
# Behaviour:
#   - Well-separated arcs (tau_k >> tau_{k-1}):
#       preceding arcs are fully relaxed at omega_char_k, so
#       Re(Z_j) ~ R_j  =>  offset reduces to Rs + SUM R_j
#       (the sequential DC offset).
#
#   - Overlapping arcs (tau_k ~ tau_{k-1}):
#       preceding arcs are only partially relaxed, so
#       Re(Z_j) < R_j  =>  offset is SMALLER, and the arcs
#       overlap on the Z' axis, matching the physical reality.
#
# Component offsets are a plotting convention; only the summed model
# is fitted to the impedance data.
# ============================================================

def _annotate_freq_on_nyquist(ax, freq, xvals, yvals, step=None):
    """Add engineering-format frequency annotations to a Nyquist plot."""
    if step is None:
        step = max(1, len(freq) // 12)
    for i in range(0, len(freq), step):
        ax.annotate(
            eng_str(freq[i], "Hz"), (xvals[i], yvals[i]),
            fontsize=7.5, color="dimgray", alpha=0.85,
            textcoords="offset points", xytext=(4, 4))


# ---- ECM Circuit Diagram Drawing Helpers ----

def _draw_inductor_coil(ax, x0, y0, width, amp, n_bumps, lw, color):
    """Draw inductor symbol as a series of bumps above the wire."""
    t = np.linspace(0, n_bumps * np.pi, n_bumps * 30 + 1)
    x_pts = x0 + (t / (n_bumps * np.pi)) * width
    y_pts = y0 + amp * np.abs(np.sin(t))
    ax.plot(x_pts, y_pts, '-', lw=lw, color=color, solid_capstyle='round')


def _draw_zigzag(ax, x0, y0, width, amp, n_teeth, lw, color):
    """Draw a zigzag (resistor) symbol."""
    flat = width * 0.12
    body = width - 2 * flat
    seg = body / (2 * n_teeth)

    pts_x = [x0, x0 + flat]
    pts_y = [y0, y0]
    for i in range(2 * n_teeth):
        pts_x.append(x0 + flat + (i + 0.5) * seg)
        pts_y.append(y0 + amp * (1 if i % 2 == 0 else -1))
    pts_x.extend([x0 + width - flat, x0 + width])
    pts_y.extend([y0, y0])
    ax.plot(pts_x, pts_y, '-', lw=lw, color=color, solid_capstyle='round')


def _draw_cpe_plates(ax, x_center, y0, gap_half, plate_h, lw, color):
    """Draw CPE symbol: two parallel plates, one tilted."""
    # Left plate (straight vertical)
    ax.plot([x_center - gap_half, x_center - gap_half],
            [y0 - plate_h, y0 + plate_h],
            '-', lw=lw + 0.8, color=color, solid_capstyle='butt')
    # Right plate (tilted for CPE)
    tilt = plate_h * 0.35
    ax.plot([x_center + gap_half, x_center + gap_half],
            [y0 - plate_h + tilt, y0 + plate_h - tilt],
            '-', lw=lw + 0.8, color=color, solid_capstyle='butt')


def _draw_ecm_circuit(ax, n_voigt, params):
    """Draw the equivalent circuit diagram: L + Rs + N×(R‖CPE).

    Draws on the given axes with proper scaling for n_voigt elements.
    """
    ax.set_axis_off()

    # Layout constants
    wire_y = 0.50
    lead = 0.20
    L_w = 0.55
    Rs_w = 0.55
    V_w = 0.80          # total width per Voigt element
    gap = 0.18
    branch_dy = 0.24    # vertical offset for parallel branches
    lw = 1.6
    clr = '#333333'

    total_w = (lead + L_w + gap + Rs_w
               + n_voigt * (gap + V_w) + gap + lead)

    ax.set_xlim(-0.08, total_w + 0.08)
    ax.set_ylim(0.05, 0.95)

    x = 0.0

    # ---- Start terminal ----
    ax.plot(x, wire_y, 'o', ms=5, color=clr, zorder=5)
    ax.plot([x, x + lead], [wire_y, wire_y], '-', lw=lw, color=clr)
    x += lead

    # ---- Inductor L ----
    _draw_inductor_coil(ax, x, wire_y, L_w, 0.065, 4, lw, clr)
    ax.text(x + L_w / 2, wire_y + 0.17, 'L', ha='center', va='bottom',
            fontsize=10, fontweight='bold', color='#1a5276')
    ax.text(x + L_w / 2, wire_y - 0.13, eng_str(params[0], 'H'),
            ha='center', va='top', fontsize=8, color='#1a5276',
            fontfamily='monospace')
    x += L_w

    # ---- Wire gap ----
    ax.plot([x, x + gap], [wire_y, wire_y], '-', lw=lw, color=clr)
    x += gap

    # ---- Rs resistor ----
    _draw_zigzag(ax, x, wire_y, Rs_w, 0.055, 5, lw, clr)
    ax.text(x + Rs_w / 2, wire_y + 0.16, 'Rs', ha='center', va='bottom',
            fontsize=10, fontweight='bold', color='#922B21')
    ax.text(x + Rs_w / 2, wire_y - 0.13,
            eng_str(params[1], '\u03a9'),
            ha='center', va='top', fontsize=8, color='#922B21',
            fontfamily='monospace')
    x += Rs_w

    # ---- Voigt elements ----
    for i in range(n_voigt):
        idx = 2 + 3 * i
        Ri = params[idx]
        Qi = params[idx + 1]
        ai = params[idx + 2]

        # Wire gap to junction
        ax.plot([x, x + gap], [wire_y, wire_y], '-', lw=lw, color=clr)
        x += gap

        jx_start = x
        jx_end = x + V_w
        y_top = wire_y + branch_dy
        y_bot = wire_y - branch_dy

        # Vertical junction lines (left & right)
        ax.plot([jx_start, jx_start], [y_bot, y_top],
                '-', lw=lw, color=clr)
        ax.plot([jx_end, jx_end], [y_bot, y_top],
                '-', lw=lw, color=clr)

        # ---- Top branch: Resistor Ri ----
        r_margin = 0.10
        r_w = V_w - 2 * r_margin
        ax.plot([jx_start, jx_start + r_margin], [y_top, y_top],
                '-', lw=lw, color=clr)
        _draw_zigzag(ax, jx_start + r_margin, y_top, r_w, 0.045, 4,
                     lw, '#1a5276')
        ax.plot([jx_end - r_margin, jx_end], [y_top, y_top],
                '-', lw=lw, color=clr)
        ax.text(jx_start + V_w / 2, y_top + 0.24,
                f'R{i+1}', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='#1a5276')
        ax.text(jx_start + V_w / 2, y_top + 0.14,
                eng_str(Ri, '\u03a9'), ha='center', va='bottom',
                fontsize=7, color='#1a5276', fontfamily='monospace')

        # ---- Bottom branch: CPE ----
        c_wire_l = V_w * 0.35
        c_wire_r = V_w * 0.35
        ax.plot([jx_start, jx_start + c_wire_l], [y_bot, y_bot],
                '-', lw=lw, color=clr)
        _draw_cpe_plates(ax, jx_start + V_w / 2, y_bot,
                         0.045, 0.065, lw, '#196F3D')
        ax.plot([jx_end - c_wire_r, jx_end], [y_bot, y_bot],
                '-', lw=lw, color=clr)
        ax.text(jx_start + V_w / 2, y_bot - 0.18,
                f'CPE{i+1}', ha='center', va='top',
                fontsize=9, fontweight='bold', color='#196F3D')
        ax.text(jx_start + V_w / 2, y_bot - 0.28,
                f'Q={Qi:.1e}  \u03b1={ai:.3f}',
                ha='center', va='top', fontsize=6.5, color='#196F3D',
                fontfamily='monospace')

        x = jx_end

    # ---- End terminal ----
    ax.plot([x, x + gap + lead], [wire_y, wire_y], '-', lw=lw, color=clr)
    ax.plot(x + gap + lead, wire_y, 'o', ms=5, color=clr, zorder=5)


def plot_fit_with_components(freq, zreal, zimag, Z_fit, Z_base, voigt_list,
                             params, n_voigt, metrics, sign_conv,
                             kk_result=None, save_path=None,
                             data_file=None):
    """
    Extended figure with:
      Top row:      Settings table (left/center) + ECM circuit diagram (right)
      Middle row:   Nyquist (left) + Bode (right)
      Bottom row:   Circuit-model residuals (left) + KK residuals (right)
    Left-aligned suptitle.
    """
    omega = 2.0 * np.pi * freq
    colors = plt.cm.tab10(np.linspace(0, 0.7, max(n_voigt, 1)))
    has_kk = kk_result is not None

    # ---- Adaptive figure size (fits 2K 24" screen) ----
    fig_w = max(19, 17 + n_voigt * 1.0)
    fig_h = 10.5
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = GridSpec(3, 2, figure=fig,
                  height_ratios=[0.45, 1.3, 0.7],
                  hspace=0.35, wspace=0.28)

    # ---- Centered suptitle (includes data file name if provided) ----
    _title_file = ''
    if data_file:
        _title_file = f'\n{os.path.basename(data_file)}'
    fig.suptitle(
        f"ECM Fit Results \u2014 L + Rs + {n_voigt}\u00d7(R\u2016CPE)"
        f"{_title_file}",
        fontsize=14, fontweight='bold',
        x=0.50, ha='center', va='top')

    # ---- Top-left: Settings table ----
    ax_tbl = fig.add_subplot(gs[0, 0])
    ax_tbl.set_axis_off()

    # Build table data
    tbl_data = [
        ['Model',          f'L + Rs + {n_voigt}\u00d7(R\u2016CPE)'],
        ['Voigt elements', f'{n_voigt}'],
        ['Parameters',     f'{metrics["n_params"]}'],
        ['Data points',    f'{len(freq)}'],
        ['Freq. range',    f'{eng_str(freq.min(), "Hz")} \u2013 '
                           f'{eng_str(freq.max(), "Hz")}'],
        ['RMSE',           f'{metrics["rmse"]:.6f} \u03a9'],
        ['R\u00b2',        f'{metrics["r2"]:.6f}'],
    ]
    if 'bic' in metrics:
        tbl_data.append(['BIC', f'{metrics["bic"]:.2f}'])
    if has_kk:
        kk_status = 'PASS' if kk_result['passed'] else 'FAIL'
        tbl_data.append(['KK status',
                         f'{kk_status}  (M={kk_result["n_rc"]}, '
                         f'res={kk_result["mean_res_pct"]:.3f}%)'])

    tbl = ax_tbl.table(
        cellText=tbl_data,
        colLabels=['Setting', 'Value'],
        cellLoc='center', colLoc='center',
        loc='center',
        colWidths=[0.38, 0.58])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.35)

    # Style the table
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor('#cccccc')
        if row == 0:
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(color='white', fontweight='bold',
                                fontsize=9)
        elif row % 2 == 0:
            cell.set_facecolor('#f2f3f4')
        else:
            cell.set_facecolor('#ffffff')
        if col == 0 and row > 0:
            cell.set_text_props(fontweight='bold', fontsize=8)

    # (no separate title — column headers "Setting"/"Value" are enough)

    # ---- Top-right: ECM circuit diagram ----
    ax_circ = fig.add_subplot(gs[0, 1])
    _draw_ecm_circuit(ax_circ, n_voigt, params)
    # (no subtitle — the circuit diagram is self-explanatory)

    # ---- Existing 4 subplots (unchanged content) ----
    ax1 = fig.add_subplot(gs[1, 0])   # Nyquist
    ax2 = fig.add_subplot(gs[1, 1])   # Bode
    ax3 = fig.add_subplot(gs[2, 0])   # Circuit residuals
    ax4 = fig.add_subplot(gs[2, 1])   # KK residuals

    # ---- 1) Nyquist ----
    y_meas = _nyquist_sign(zimag, sign_conv)
    y_fit = _nyquist_sign(Z_fit.imag, sign_conv)

    ax1.plot(zreal, y_meas, "o", ms=6, mfc="none", mec="dimgray",
             mew=1.5, label="Measured", zorder=2, alpha=0.85)
    ax1.plot(Z_fit.real, y_fit, "-", lw=3, color="black",
             label="Circuit Fit", zorder=5)

    # KK fit on Nyquist
    if has_kk:
        y_kk = _nyquist_sign(kk_result["Z_kk"].imag, sign_conv)
        ax1.plot(kk_result["Z_kk"].real, y_kk, "--", lw=2,
                 color="magenta",
                 label=f"KK Fit (M={kk_result['n_rc']})",
                 zorder=4, alpha=0.8)

    # ---- Individual arcs with characteristic-frequency offset ----
    # Dense frequency sweep for smooth arc curves
    omega_fine = np.logspace(
        np.log10(omega.min()), np.log10(omega.max()), 500)
    zreal_lo = zreal.min()
    zreal_hi = zreal.max()

    # Pre-compute each Voigt element's dense impedance and tau
    arc_data = []
    for i in range(n_voigt):
        idx = 2 + 3 * i
        Ri = params[idx]
        Qi = params[idx + 1]
        ai = params[idx + 2]
        # Correct ZARC time constant: τ = (R·Q)^(1/α)
        rq_i = Ri * Qi
        tau_i = (rq_i ** (1.0 / ai)) if (rq_i > 0 and ai > 0) else rq_i
        Zv_fine = Z_voigt(omega_fine, Ri, Qi, ai)
        arc_data.append((Ri, Qi, ai, tau_i, Zv_fine))

    # Draw each arc with its characteristic-frequency offset
    Rs = params[1]
    for i in range(n_voigt):
        Ri, Qi, ai, tau_i, Zv_fine_i = arc_data[i]

        # Characteristic angular frequency for this arc
        omega_char_i = (1.0 / tau_i) if tau_i > 1e-30 else omega_fine[0]

        # Offset = Rs + cumulative real impedance of all preceding
        # arcs evaluated at THIS arc's characteristic frequency.
        real_offset = Rs
        for j in range(i):
            Rj, Qj, aj, _, _ = arc_data[j]
            Zj_at_char = Z_voigt(np.array([omega_char_i]), Rj, Qj, aj)
            real_offset += Zj_at_char.real[0]

        # Arc curve: individual semicircle at the computed offset
        x_arc = real_offset + Zv_fine_i.real
        y_arc = _nyquist_sign(Zv_fine_i.imag, sign_conv)

        # Clip arcs so they don't extend beyond the measured data range
        arc_mask = (x_arc >= zreal_lo) & (x_arc <= zreal_hi)
        xm = x_arc[arc_mask]
        ym = y_arc[arc_mask]

        # Semi-transparent fill to visualise overlap zones
        if len(xm) > 2:
            ax1.fill_between(xm, 0, ym, color=colors[i], alpha=0.10,
                             zorder=1)
        ax1.plot(xm, ym, "--", lw=1.5, color=colors[i],
                 label=f"RC{i+1} (R={Ri:.4f}, a={ai:.3f})",
                 zorder=3, alpha=0.85)

    # Frequency annotations on Nyquist
    _annotate_freq_on_nyquist(ax1, freq, zreal, y_meas)

    ylabel_nq = "-Z'' (Ohm)" if sign_conv == "negative" else "Z'' (Ohm)"
    ax1.set_xlabel("Z' (Ohm)", fontsize=11)
    ax1.set_ylabel(ylabel_nq, fontsize=11)
    ax1.set_title(f"Nyquist - L+Rs+{n_voigt}x(R||CPE)",
                  fontsize=11, fontweight="bold")
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.grid(True, alpha=0.3, ls="--")
    ax1.legend(fontsize=8, loc="upper right", framealpha=0.8)
    apply_eng_format(ax1, "both")

    # Lock axes to measured data range
    x_pad = (zreal.max() - zreal.min()) * 0.05 + 1e-9
    y_pad = (y_meas.max() - y_meas.min()) * 0.05 + 1e-9
    ax1.set_xlim(zreal.min() - x_pad, zreal.max() + x_pad)
    ax1.set_ylim(y_meas.min() - y_pad, y_meas.max() + y_pad)

    # Parameter text box
    txt = format_params(params, n_voigt)
    txt += f"\n\n  RMSE = {metrics['rmse']:.6f} Ohm"
    txt += f"\n  R^2  = {metrics['r2']:.6f}"
    if "bic" in metrics:
        txt += f"\n  BIC  = {metrics['bic']:.2f}"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.85)
    ax1.text(0.02, 0.98, txt, transform=ax1.transAxes, fontsize=6,
             verticalalignment="top", fontfamily="monospace", bbox=props)

    # ---- 2) Bode ----
    Zmod_m = np.sqrt(zreal**2 + zimag**2)
    Zph_m = np.degrees(np.arctan2(zimag, zreal))
    Zmod_f = np.abs(Z_fit)
    Zph_f = np.degrees(np.arctan2(Z_fit.imag, Z_fit.real))

    ax2.set_xlabel("Frequency (Hz)", fontsize=11)
    ax2.set_xscale("log")
    ax2.set_ylabel("|Z| (Ohm)", fontsize=11, color="tab:blue")
    ax2.loglog(freq, Zmod_m, "o", ms=4, color="tab:blue",
               label="|Z| meas", alpha=0.6)
    ax2.loglog(freq, Zmod_f, "-", lw=2.5, color="navy",
               label="|Z| circuit fit")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    if has_kk:
        Zmod_kk = np.abs(kk_result["Z_kk"])
        ax2.loglog(freq, Zmod_kk, "--", lw=1.5, color="magenta",
                   label=f"|Z| KK (M={kk_result['n_rc']})", alpha=0.8)

    ax2b = ax2.twinx()
    ax2b.set_ylabel("Phase (deg)", fontsize=11, color="tab:red")
    ax2b.semilogx(freq, Zph_m, "s", ms=3, color="tab:red",
                  label="Phase meas", alpha=0.6)
    ax2b.semilogx(freq, Zph_f, "-", lw=2.5, color="darkred",
                  label="Phase circuit")
    if has_kk:
        Zph_kk = np.degrees(np.arctan2(kk_result["Z_kk"].imag,
                                       kk_result["Z_kk"].real))
        ax2b.semilogx(freq, Zph_kk, "--", lw=1.5, color="orchid",
                       label="Phase KK", alpha=0.8)
    ax2b.tick_params(axis="y", labelcolor="tab:red")

    ax2.set_title("Bode Plot", fontsize=11, fontweight="bold")
    ax2.grid(True, alpha=0.3, ls="--")
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8, framealpha=0.8)
    apply_eng_format(ax2, "x")
    apply_eng_format(ax2, "y")

    # ---- 3) Circuit-model residuals ----
    res_real = zreal - Z_fit.real
    res_imag = zimag - Z_fit.imag
    ax3.semilogx(freq, res_real, "o-", ms=3, color="steelblue",
                 label="Zreal res.")
    ax3.semilogx(freq, res_imag, "s-", ms=3, color="tomato",
                 label="Zimag res.")
    ax3.axhline(0, color="black", lw=0.8, ls="--")
    ax3.set_xlabel("Frequency (Hz)", fontsize=11)
    ax3.set_ylabel("Residual (Ohm)", fontsize=11)
    ax3.set_title("Circuit Fit Residuals", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=8, loc="upper right", framealpha=0.8)
    ax3.grid(True, alpha=0.3, ls="--")
    apply_eng_format(ax3, "x")
    apply_eng_format(ax3, "y")

    # ---- 4) KK residuals (%) ----
    if has_kk:
        ax4.semilogx(freq, kk_result["res_real_pct"], "o-", ms=3,
                      color="steelblue", label="KK Zreal res. %")
        ax4.semilogx(freq, kk_result["res_imag_pct"], "s-", ms=3,
                      color="tomato", label="KK Zimag res. %")
        ax4.axhline(0, color="black", lw=0.8, ls="--")
        ax4.axhline(1.0, color="gray", lw=0.6, ls=":", alpha=0.5)
        ax4.axhline(-1.0, color="gray", lw=0.6, ls=":", alpha=0.5)
        status_txt = ("PASS" if kk_result["passed"] else "FAIL")
        status_clr = "green" if kk_result["passed"] else "red"
        ax4.text(0.98, 0.98,
                 f"KK: {status_txt}\nM={kk_result['n_rc']}\n"
                 f"mean |res|={kk_result['mean_res_pct']:.3f}%",
                 transform=ax4.transAxes, fontsize=7,
                 ha="right", va="top",
                 color=status_clr, fontweight="bold",
                 bbox=dict(boxstyle="round", facecolor="white",
                           alpha=0.8))
    else:
        ax4.text(0.5, 0.5, "No KK data", transform=ax4.transAxes,
                 ha="center", va="center", fontsize=12, color="gray")
    ax4.set_xlabel("Frequency (Hz)", fontsize=11)
    ax4.set_ylabel("Relative Residual (%)", fontsize=11)
    ax4.set_title("KK Residuals", fontsize=11, fontweight="bold")
    ax4.legend(fontsize=8, loc="upper right", framealpha=0.8)
    ax4.grid(True, alpha=0.3, ls="--")
    apply_eng_format(ax4, "x")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        success(f"Plot saved to: {save_path}")

    plt.show(block=True)
    plt.close(fig)


# ============================================================
# SECTION 13: CSV Export
# ============================================================

def export_csv(filepath, freq, zreal, zimag, Z_fit, params, n_voigt,
               metrics, kk_result=None):
    """Export measured, fitted, and KK data to CSV."""
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([f"# EIS Fitter {__version__} - Fit Results"])
        w.writerow([f"# Model: L + Rs + {n_voigt}x(R||CPE)"])
        w.writerow(["# " + line.strip()
                     for line in format_params(params, n_voigt).split("\n")])
        w.writerow([f"# RMSE = {metrics['rmse']:.6f} Ohm"])
        w.writerow([f"# R^2  = {metrics['r2']:.6f}"])
        if "bic" in metrics:
            w.writerow([f"# BIC  = {metrics['bic']:.2f}"])
        if kk_result:
            w.writerow([f"# KK: M={kk_result['n_rc']}, "
                         f"mean_res={kk_result['mean_res_pct']:.4f}%, "
                         f"{'PASS' if kk_result['passed'] else 'FAIL'}"])
        w.writerow([])

        header = ["Freq_Hz", "Zreal_meas", "Zimag_meas",
                   "Zreal_fit", "Zimag_fit",
                   "Residual_real", "Residual_imag"]
        if kk_result:
            header += ["Zreal_KK", "Zimag_KK",
                       "KK_res_real_%", "KK_res_imag_%"]
        w.writerow(header)

        for i in range(len(freq)):
            row = [
                f"{freq[i]:.6e}", f"{zreal[i]:.6e}", f"{zimag[i]:.6e}",
                f"{Z_fit.real[i]:.6e}", f"{Z_fit.imag[i]:.6e}",
                f"{zreal[i] - Z_fit.real[i]:.6e}",
                f"{zimag[i] - Z_fit.imag[i]:.6e}",
            ]
            if kk_result:
                row += [
                    f"{kk_result['Z_kk'].real[i]:.6e}",
                    f"{kk_result['Z_kk'].imag[i]:.6e}",
                    f"{kk_result['res_real_pct'][i]:.4f}",
                    f"{kk_result['res_imag_pct'][i]:.4f}",
                ]
            w.writerow(row)
    success(f"CSV exported to: {filepath}")


# ============================================================
# SECTION 14: Save text results
# ============================================================

def save_results_txt(filepath, chosen_file, freq, params, n_voigt,
                     metrics, kk_result=None):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"EIS Fitter {__version__} - FIT RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Data file  : {os.path.relpath(chosen_file, '.')}\n")
        f.write(f"Model      : L + Rs + {n_voigt}x(R||CPE)\n")
        f.write(f"Data points: {len(freq)}\n")
        f.write(f"Freq range : {eng_str(freq.min(), 'Hz')} - "
                f"{eng_str(freq.max(), 'Hz')}\n\n")
        f.write("PARAMETERS (sorted by time constant):\n")
        f.write(format_params(params, n_voigt) + "\n\n")
        f.write(f"RMSE = {metrics['rmse']:.6f} Ohm\n")
        f.write(f"R^2  = {metrics['r2']:.6f}\n")
        if "bic" in metrics:
            f.write(f"BIC  = {metrics['bic']:.2f}\n")
        if kk_result:
            f.write("\n" + "-" * 40 + "\n")
            f.write("KRAMERS-KRONIG VALIDATION:\n")
            f.write(format_kk_results(kk_result) + "\n")
        f.write("=" * 60 + "\n")
    success(f"Results saved to: {filepath}")


# ============================================================
# SECTION 15: Persistent Fit Log (includes KK)
# ============================================================

def log_fit_action(log_path, chosen_file, method_label, params, n_voigt,
                   metrics, kk_result=None):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("-" * 70 + "\n")
        f.write(f"[{now}]\n")
        f.write(f"File   : {os.path.relpath(chosen_file, '.')}\n")
        f.write(f"Method : {method_label}\n")
        f.write(f"Model  : L + Rs + {n_voigt}x(R||CPE)\n")
        f.write(format_params(params, n_voigt) + "\n")
        f.write(f"  RMSE = {metrics['rmse']:.6f} Ohm\n")
        f.write(f"  R^2  = {metrics['r2']:.6f}\n")
        if "bic" in metrics:
            f.write(f"  BIC  = {metrics['bic']:.2f}\n")
        if kk_result:
            f.write("  --- Kramers-Kronig ---\n")
            f.write(format_kk_results(kk_result) + "\n")
        f.write("\n")
    info(f"Logged to: {log_path}")


def log_kk_standalone(log_path, chosen_file, kk_result, freq):
    """Log a standalone KK validation entry (before circuit fitting)."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("-" * 70 + "\n")
        f.write(f"[{now}]\n")
        f.write(f"File   : {os.path.relpath(chosen_file, '.')}\n")
        f.write(f"Method : Kramers-Kronig validation (Lin-KK, extended)\n")
        f.write(f"Freq   : {eng_str(freq.min(), 'Hz')} - "
                f"{eng_str(freq.max(), 'Hz')}  ({len(freq)} pts)\n")
        f.write(format_kk_results(kk_result) + "\n")
        f.write("\n")
    info(f"KK result logged to: {log_path}")


# ============================================================
# SECTION 15.5: DRT Analysis (Distribution of Relaxation Times)
# ============================================================
#
# Implements DRT deconvolution following the methodology of:
#   Wang et al., "DRTtools: Freely Accessible Distribution of
#   Relaxation Times Analysis for Electrochemical Impedance
#   Spectroscopy", ACS Electrochem. 2025, 1, 2680-2689.
#
# The DRT represents the total impedance as:
#   Z(f) = R_inf + j*omega*L_0 + integral gamma(log tau) /
#          (1 + j*2*pi*f*tau) d(log tau)
#
# where gamma(log tau) is the distribution function whose peaks
# may indicate relaxation features and their time
# constants.  The integral equation is discretised with Gaussian
# radial basis functions (RBFs) and solved via Tikhonov
# regularisation with an optional non-negativity constraint.
#
# Regularisation parameter selection:
#   - Custom (user-specified lambda)
#   - GCV  (generalised cross-validation)
#   - mGCV (modified GCV, recommended)
#   - LC   (L-curve)
# ============================================================

from scipy.optimize import minimize_scalar
from scipy.optimize import nnls as _scipy_nnls
from scipy.linalg import cho_factor, cho_solve


def _gaussian_rbf(x, mu, epsilon):
    """Gaussian radial basis function centred at mu."""
    return np.exp(-(epsilon * (x - mu)) ** 2)


def _build_drt_A_matrix(omega, log_tau_grid, epsilon):
    """
    Build the discretisation matrix A for the DRT integral.

    For each frequency index n and basis-function index m:
      A_re[n, m] = integral phi_m(log tau) / (1 + (omega_n*tau)^2) d(log tau)
      A_im[n, m] = integral phi_m(log tau) * (-omega_n*tau)
                    / (1 + (omega_n*tau)^2) d(log tau)

    The integrals are evaluated by trapezoidal quadrature on a fine
    log-tau grid that extends one grid-spacing beyond each side.
    """
    n_tau = len(log_tau_grid)
    N = len(omega)
    delta = log_tau_grid[1] - log_tau_grid[0] if n_tau > 1 else 1.0

    # Fine quadrature grid (5x density, +/- 3 delta padding)
    pad = 3.0 * delta
    n_quad = max(5 * n_tau, 200)
    lt_fine = np.linspace(log_tau_grid[0] - pad,
                          log_tau_grid[-1] + pad, n_quad)
    tau_fine = 10.0 ** lt_fine  # shape (n_quad,)

    # Pre-compute RBF values: shape (n_tau, n_quad)
    phi_all = np.exp(-(epsilon * (lt_fine[None, :] -
                                   log_tau_grid[:, None])) ** 2)

    # Pre-compute kernel values for every (freq, quad_point)
    # wt[n, q] = omega[n] * tau_fine[q]
    wt = omega[:, None] * tau_fine[None, :]          # (N, n_quad)
    denom = 1.0 + wt ** 2                            # (N, n_quad)
    kernel_re = 1.0 / denom                          # (N, n_quad)
    kernel_im = -wt / denom                          # (N, n_quad)

    # Integrate:  A[n,m] = trapz( phi_m * kernel_n , lt_fine )
    # Use matrix multiplication after element-wise weighting.
    d_lt = np.diff(lt_fine)
    # Trapezoidal weights
    trap_w = np.zeros(n_quad)
    trap_w[0] = d_lt[0] / 2.0
    trap_w[-1] = d_lt[-1] / 2.0
    trap_w[1:-1] = (d_lt[:-1] + d_lt[1:]) / 2.0

    # phi_w[m, q] = phi_all[m, q] * trap_w[q]
    phi_w = phi_all * trap_w[None, :]  # (n_tau, n_quad)

    # A_re[n, m] = sum_q  kernel_re[n,q] * phi_w[m,q]
    A_re = kernel_re @ phi_w.T   # (N, n_tau)
    A_im = kernel_im @ phi_w.T   # (N, n_tau)

    return A_re, A_im


def _build_L_matrix(n, order=2):
    """Build finite-difference differentiation matrix of given order.

    Vectorized construction using np.eye with diagonal offsets.
    """
    if order == 1:
        L = -np.eye(n - 1, n) + np.eye(n - 1, n, k=1)
    elif order == 2:
        L = (np.eye(n - 2, n) - 2.0 * np.eye(n - 2, n, k=1)
             + np.eye(n - 2, n, k=2))
    else:
        L = np.eye(n)
    return L


def _solve_tikhonov(A, b, L, lam, non_negative=True):
    """
    Solve the Tikhonov-regularised problem:
        x_opt = argmin || A x - b ||^2  + lam * || L x ||^2
    with optional non-negativity constraint on x.

    When non_negative is True, uses CVXPY if available, otherwise
    falls back to iterative clipping (simple projected gradient).
    """
    ATA = A.T @ A
    LTL = L.T @ L
    ATb = A.T @ b
    H = ATA + lam * LTL

    if not non_negative:
        # Unconstrained closed-form solution
        return np.linalg.solve(H, ATb)

    # --- Non-negative solve ---
    # Try CVXPY for a proper QP solution
    try:
        import cvxpy as cp
        n = A.shape[1]
        x = cp.Variable(n)
        objective = cp.Minimize(
            cp.sum_squares(A @ x - b) + lam * cp.sum_squares(L @ x)
        )
        constraints = [x >= 0]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP, warm_start=True, verbose=False,
                   max_iter=10000, eps_abs=1e-8, eps_rel=1e-8)
        if x.value is not None:
            return np.array(x.value).flatten()
    except Exception:
        pass

    # Fallback: solve unconstrained then clip + iterate
    x = np.linalg.solve(H, ATb)
    for _ in range(50):
        x = np.maximum(x, 0.0)
        # Re-solve with active-set heuristic: fix zeros, solve rest
        active = x > 0
        if active.sum() == 0:
            return np.zeros(len(x))
        A_a = A[:, active]
        L_a = L[:, active]
        H_a = A_a.T @ A_a + lam * L_a.T @ L_a
        g_a = A_a.T @ b
        x_a = np.linalg.solve(H_a, g_a)
        x_new = np.zeros_like(x)
        x_new[active] = x_a
        if np.allclose(np.maximum(x_new, 0), np.maximum(x, 0),
                       atol=1e-10):
            break
        x = np.maximum(x_new, 0.0)
    return np.maximum(x, 0.0)


def _compute_gcv(A, b, L, lam):
    """Compute GCV score for a given lambda.

    Uses Cholesky decomposition instead of full matrix inverse for
    ~2× speedup and better numerical stability.
    """
    n = A.shape[0]
    H = A.T @ A + lam * L.T @ L
    try:
        cf = cho_factor(H)
    except (np.linalg.LinAlgError, ValueError):
        return np.inf
    x = cho_solve(cf, A.T @ b)
    # trace(Hat) = trace(A @ H_inv @ A.T) = sum(A * (H_inv @ A.T).T)
    V = cho_solve(cf, A.T)       # H \ A.T, shape (n_total, n_data)
    Hat_trace = np.sum(A * V.T)  # trace without forming full Hat matrix
    residual = b - A @ x
    rss = np.sum(residual ** 2)
    trace_I_minus_H = n - Hat_trace
    if trace_I_minus_H <= 0:
        return np.inf
    return (rss / n) / (trace_I_minus_H / n) ** 2


def _compute_mgcv(A, b, L, lam):
    """Modified GCV (mGCV) — adds a stabilisation factor.

    Uses Cholesky decomposition instead of full matrix inverse.
    """
    n = A.shape[0]
    H = A.T @ A + lam * L.T @ L
    try:
        cf = cho_factor(H)
    except (np.linalg.LinAlgError, ValueError):
        return np.inf
    x = cho_solve(cf, A.T @ b)
    V = cho_solve(cf, A.T)
    Hat_trace = np.sum(A * V.T)
    residual = b - A @ x
    rss = np.sum(residual ** 2)
    trace_I_minus_H = n - Hat_trace
    if trace_I_minus_H <= 0:
        return np.inf
    # Modification factor rho (typically 2.0 gives good results)
    rho = 2.0
    return (rss / n) / ((trace_I_minus_H + rho * (n - trace_I_minus_H)) / n) ** 2


def _select_lambda_gcv(A, b, L, method="mGCV"):
    """Auto-select lambda by minimising GCV or mGCV score.

    The search is bounded to [1e-7, 1e3] to avoid extremely small
    lambda values that lead to overfitting with edge artefacts.
    Extended upper bound (1e3) allows heavy regularization for noisy data.
    A coarse grid scan is performed first to find a good initial
    bracket, followed by bounded scalar minimisation.
    """
    score_fn = _compute_mgcv if method == "mGCV" else _compute_gcv

    # Coarse grid scan to find a robust starting bracket
    log_lams = np.linspace(-7, 3, 60)
    scores = [score_fn(A, b, L, 10.0 ** ll) for ll in log_lams]
    scores = np.array(scores)
    valid = np.isfinite(scores)
    if valid.sum() < 3:
        warn("GCV/mGCV scan failed — using default lambda = 1e-3")
        return 1e-3
    best_coarse_idx = np.argmin(scores[valid])
    best_coarse_ll = log_lams[valid][best_coarse_idx]

    # Refine around the best coarse value
    lo = max(-7, best_coarse_ll - 1.5)
    hi = min(3, best_coarse_ll + 1.5)

    def objective(log_lam):
        return score_fn(A, b, L, 10.0 ** log_lam)

    result = minimize_scalar(objective, bounds=(lo, hi), method="bounded",
                             options={"xatol": 1e-3, "maxiter": 200})
    lam = 10.0 ** result.x

    # Safety floor: lambda below 1e-6 almost always overfits EIS data
    if lam < 1e-6:
        warn(f"  {method} selected lambda = {lam:.2e} (very small). "
             f"Clamping to 1e-5 to prevent overfitting.")
        lam = 1e-5
    return lam


def _select_lambda_lcurve(A, b, L, n_points=80):
    """Select lambda using the L-curve method (maximum curvature)."""
    log_lams = np.linspace(-8, 2, n_points)
    rho_vals = []
    eta_vals = []

    for log_lam in log_lams:
        lam = 10.0 ** log_lam
        H = A.T @ A + lam * L.T @ L
        try:
            x = np.linalg.solve(H, A.T @ b)
        except np.linalg.LinAlgError:
            rho_vals.append(np.inf)
            eta_vals.append(np.inf)
            continue
        residual = b - A @ x
        rho_vals.append(np.log10(np.linalg.norm(residual) + 1e-30))
        eta_vals.append(np.log10(np.linalg.norm(L @ x) + 1e-30))

    rho_vals = np.array(rho_vals)
    eta_vals = np.array(eta_vals)

    # Compute curvature using finite differences
    valid = np.isfinite(rho_vals) & np.isfinite(eta_vals)
    if valid.sum() < 5:
        return 1e-3  # fallback
    rho_v = rho_vals[valid]
    eta_v = eta_vals[valid]
    ll_v = log_lams[valid]

    # First and second derivatives via central differences
    dr = np.gradient(rho_v)
    de = np.gradient(eta_v)
    d2r = np.gradient(dr)
    d2e = np.gradient(de)

    # Curvature of the parametric curve (rho, eta)
    kappa = np.abs(dr * d2e - de * d2r) / (dr ** 2 + de ** 2) ** 1.5
    idx_max = np.argmax(kappa[2:-2]) + 2  # avoid edge effects
    return 10.0 ** ll_v[idx_max]


def compute_drt(freq, zreal, zimag,
                lambda_reg=1e-3, lambda_method="custom",
                rbf_type="gaussian", fwhm_coeff=0.5,
                reg_order=2, data_use="combined",
                inductance_mode="without",
                non_negative=True):
    """
    Compute DRT using Tikhonov regularisation with Gaussian RBF
    discretisation, following DRTtools methodology.

    Parameters
    ----------
    freq : 1-D array
        Measurement frequencies (Hz), must be positive.
    zreal, zimag : 1-D arrays
        Real and imaginary parts of impedance (Ohm).
    lambda_reg : float
        Regularisation parameter (used when lambda_method='custom').
    lambda_method : str
        'custom', 'GCV', 'mGCV', or 'LC'.
    rbf_type : str
        Currently only 'gaussian' is implemented.
    fwhm_coeff : float
        FWHM coefficient controlling RBF width (default 0.5).
    reg_order : int
        Differentiation order (1 or 2) for the regularisation matrix.
    data_use : str
        'combined' (default), 'real', or 'imaginary'.
    inductance_mode : str
        'without' or 'with' (include inductance term).
    non_negative : bool
        Enforce gamma >= 0.

    Returns
    -------
    dict with keys:
        gamma, log_tau, tau, R_inf, L_ind, R_pol,
        Z_fit, rmse, r2, lambda_used, and settings.
    """
    freq, zreal, zimag = validate_spectrum(freq, zreal, zimag)
    omega = 2.0 * np.pi * freq
    N = len(freq)

    # ---- Build log-tau grid ----
    tau_min = 1.0 / (2.0 * np.pi * freq.max())
    tau_max = 1.0 / (2.0 * np.pi * freq.min())
    log_tau_min = np.log10(tau_min) - 1.0   # extend 1 decade
    log_tau_max = np.log10(tau_max) + 1.0
    n_tau = max(2 * N, 80)
    log_tau_grid = np.linspace(log_tau_min, log_tau_max, n_tau)
    tau_grid = 10.0 ** log_tau_grid
    delta = log_tau_grid[1] - log_tau_grid[0]

    # ---- Shape parameter for Gaussian RBF ----
    # FWHM = fwhm_coeff * delta  =>  epsilon = 2*sqrt(ln2) / FWHM
    fwhm = fwhm_coeff * delta
    epsilon = 2.0 * np.sqrt(np.log(2.0)) / fwhm

    info(f"DRT grid: {n_tau} points, log(tau) = "
         f"[{log_tau_min:.2f}, {log_tau_max:.2f}], "
         f"epsilon = {epsilon:.2f}")

    # ---- Build discretisation matrices ----
    A_re, A_im = _build_drt_A_matrix(omega, log_tau_grid, epsilon)

    # ---- Assemble full system ----
    include_L = (inductance_mode == "with")
    n_extra = 0  # count of extra columns (R_inf, L)

    if data_use == "combined":
        # Stacked: [Re; Im]
        n_rows = 2 * N
        A_drt = np.vstack([A_re, A_im])  # (2N, n_tau)

        # R_inf column: contributes 1 to real, 0 to imag
        col_Rinf = np.zeros(n_rows)
        col_Rinf[:N] = 1.0
        n_extra += 1

        if include_L:
            # L column: contributes omega to imaginary part
            col_L = np.zeros(n_rows)
            col_L[N:] = omega
            n_extra += 1

        b = np.concatenate([zreal, zimag])

    elif data_use == "real":
        n_rows = N
        A_drt = A_re.copy()
        col_Rinf = np.ones(N)
        n_extra += 1
        if include_L:
            # Inductance has no effect on real part — ignore silently
            include_L = False
        b = zreal.copy()

    elif data_use == "imaginary":
        n_rows = N
        A_drt = A_im.copy()
        col_Rinf = None  # R_inf not identifiable from imag alone

        if include_L:
            col_L = omega.copy()
            n_extra += 1
        b = zimag.copy()

    else:
        raise ValueError(f"Unknown data_use: {data_use}")

    # Augmented matrix: [A_drt | col_Rinf | col_L]
    cols_extra = []
    if col_Rinf is not None:
        cols_extra.append(col_Rinf)
    if include_L:
        cols_extra.append(col_L)

    if cols_extra:
        A_full = np.hstack([A_drt] + [c[:, None] for c in cols_extra])
    else:
        A_full = A_drt

    n_total = n_tau + n_extra

    # ---- Build regularisation matrix (penalise only DRT part) ----
    L_drt = _build_L_matrix(n_tau, order=reg_order)
    L_full = np.zeros((L_drt.shape[0], n_total))
    L_full[:, :n_tau] = L_drt

    # ---- Select lambda ----
    if lambda_method == "custom":
        lam = lambda_reg
    elif lambda_method in ("GCV", "mGCV"):
        info(f"Selecting lambda by {lambda_method} ...")
        lam = _select_lambda_gcv(A_full, b, L_full, method=lambda_method)
        info(f"  -> lambda = {lam:.4e}")
    elif lambda_method == "LC":
        info("Selecting lambda by L-curve ...")
        lam = _select_lambda_lcurve(A_full, b, L_full)
        info(f"  -> lambda = {lam:.4e}")
    else:
        warn(f"Unknown lambda method '{lambda_method}', using custom.")
        lam = lambda_reg

    # ---- Solve ----
    # For non-negativity, we only constrain the DRT coefficients
    # (not R_inf or L).  Strategy: solve augmented system, then
    # project gamma >= 0 via the _solve_tikhonov helper which
    # constrains all variables.  We allow R_inf and L to be free
    # by using a two-step approach.

    if non_negative and n_extra > 0:
        # Two-step: first get unconstrained R_inf/L estimate,
        # then solve for gamma with non-negativity.
        x_unc = np.linalg.solve(
            A_full.T @ A_full + lam * L_full.T @ L_full,
            A_full.T @ b)
        R_inf_est = x_unc[n_tau] if col_Rinf is not None else 0.0
        L_ind_est = x_unc[-1] if include_L else 0.0

        # Subtract R_inf and L contributions from b
        b_reduced = b.copy()
        if col_Rinf is not None:
            b_reduced -= col_Rinf * R_inf_est
        if include_L:
            b_reduced -= cols_extra[-1] * L_ind_est

        # Solve for gamma only, with non-negativity
        L_gamma = _build_L_matrix(n_tau, order=reg_order)
        gamma = _solve_tikhonov(A_drt, b_reduced, L_gamma, lam,
                                non_negative=True)

        # Refine R_inf and L with fixed gamma
        residual_after_gamma = b - A_drt @ gamma
        if n_extra > 0:
            A_extra = np.column_stack(cols_extra)
            extra_coeffs, _, _, _ = np.linalg.lstsq(
                A_extra, residual_after_gamma, rcond=None)
            idx_e = 0
            if col_Rinf is not None:
                R_inf_est = extra_coeffs[idx_e]
                idx_e += 1
            if include_L:
                L_ind_est = extra_coeffs[idx_e]

        R_inf = R_inf_est
        L_ind = L_ind_est

    elif non_negative and n_extra == 0:
        L_gamma = _build_L_matrix(n_tau, order=reg_order)
        gamma = _solve_tikhonov(A_drt, b, L_gamma, lam,
                                non_negative=True)
        R_inf = 0.0
        L_ind = 0.0

    else:
        # Unconstrained solve
        x_opt = np.linalg.solve(
            A_full.T @ A_full + lam * L_full.T @ L_full,
            A_full.T @ b)
        gamma = x_opt[:n_tau]
        idx_e = n_tau
        R_inf = x_opt[idx_e] if col_Rinf is not None else 0.0
        if col_Rinf is not None:
            idx_e += 1
        L_ind = x_opt[idx_e] if include_L else 0.0

    # ---- Compute fitted impedance ----
    Z_fit_re = A_re @ gamma
    Z_fit_im = A_im @ gamma
    if col_Rinf is not None:
        Z_fit_re += R_inf
    if include_L:
        Z_fit_im += omega * L_ind
    Z_fit = Z_fit_re + 1j * Z_fit_im

    # ---- Polarisation resistance ----
    from scipy.integrate import trapezoid
    R_pol = trapezoid(gamma, log_tau_grid)

    # ---- Fit quality ----
    ss_res = np.sum((zreal - Z_fit.real) ** 2 + (zimag - Z_fit.imag) ** 2)
    ss_tot = np.sum((zreal - zreal.mean()) ** 2 +
                    (zimag - zimag.mean()) ** 2)
    rmse = np.sqrt(ss_res / N)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    result = {
        "gamma":       gamma,
        "log_tau":     log_tau_grid,
        "tau":         tau_grid,
        "R_inf":       R_inf,
        "L_ind":       L_ind,
        "R_pol":       R_pol,
        "Z_fit":       Z_fit,
        "rmse":        rmse,
        "r2":          r2,
        "lambda_used": lam,
        "lambda_method": lambda_method,
        "reg_order":   reg_order,
        "data_use":    data_use,
        "rbf_type":    rbf_type,
        "fwhm_coeff":  fwhm_coeff,
        "inductance":  inductance_mode,
        "non_negative": non_negative,
    }

    success(f"DRT computed: R_inf = {R_inf:.6f} Ohm, "
            f"R_pol = {R_pol:.6f} Ohm, lambda = {lam:.4e}")
    info(f"  RMSE = {rmse:.6f} Ohm,  R^2 = {r2:.6f}")
    return result


def _prompt_drt_parameters():
    """Interactive terminal prompts for DRT analysis settings."""
    banner("DRT ANALYSIS PARAMETERS", C.HEADER)

    # 1) Regularisation parameter selection method
    print(f"\n  {C.BOLD}Regularisation parameter selection:{C.RESET}")
    print(f"    [1] Custom lambda (manual)")
    print(f"    [2] GCV  (generalised cross-validation)")
    print(f"    [3] mGCV (modified GCV, recommended)")
    print(f"    [4] LC   (L-curve)")
    lam_choice = input(f"  {C.BOLD}  Choice [3]: {C.RESET}").strip()
    lam_methods = {"1": "custom", "2": "GCV", "3": "mGCV", "4": "LC"}
    lambda_method = lam_methods.get(lam_choice, "mGCV")

    lambda_reg = 1e-3
    if lambda_method == "custom":
        try:
            val = input(f"  {C.BOLD}  Lambda value [1e-3]: {C.RESET}").strip()
            if val:
                lambda_reg = float(val)
        except ValueError:
            lambda_reg = 1e-3

    # 2) Regularisation derivative order
    print(f"\n  {C.BOLD}Regularisation derivative order:{C.RESET}")
    print(f"    [1] 1st order")
    print(f"    [2] 2nd order (recommended)")
    ord_choice = input(f"  {C.BOLD}  Choice [2]: {C.RESET}").strip()
    reg_order = 1 if ord_choice == "1" else 2

    # 3) Data use
    print(f"\n  {C.BOLD}Data use:{C.RESET}")
    print(f"    [1] Combined Re-Im (recommended)")
    print(f"    [2] Real part only")
    print(f"    [3] Imaginary part only")
    du_choice = input(f"  {C.BOLD}  Choice [1]: {C.RESET}").strip()
    data_uses = {"1": "combined", "2": "real", "3": "imaginary"}
    data_use = data_uses.get(du_choice, "combined")

    # 4) Inductance
    print(f"\n  {C.BOLD}Inductance handling:{C.RESET}")
    print(f"    [1] Fitting without inductance")
    print(f"    [2] Fitting with inductance (recommended)")
    ind_choice = input(f"  {C.BOLD}  Choice [2]: {C.RESET}").strip()
    inductance_mode = "without" if ind_choice == "1" else "with"

    # 5) FWHM coefficient
    fwhm_coeff = 0.5
    try:
        val = input(f"\n  {C.BOLD}FWHM coefficient [0.5]: "
                    f"{C.RESET}").strip()
        if val:
            fwhm_coeff = float(val)
            fwhm_coeff = max(0.1, min(2.0, fwhm_coeff))
    except ValueError:
        fwhm_coeff = 0.5

    # 6) Non-negativity
    print(f"\n  {C.BOLD}Non-negativity constraint on gamma:{C.RESET}")
    print(f"    [1] Yes (recommended)")
    print(f"    [2] No")
    nn_choice = input(f"  {C.BOLD}  Choice [1]: {C.RESET}").strip()
    non_negative = nn_choice != "2"

    settings = {
        "lambda_reg":      lambda_reg,
        "lambda_method":   lambda_method,
        "reg_order":       reg_order,
        "data_use":        data_use,
        "inductance_mode": inductance_mode,
        "fwhm_coeff":      fwhm_coeff,
        "non_negative":    non_negative,
    }

    # Print summary
    section("DRT SETTINGS SUMMARY", C.GREEN)
    print(f"  Lambda method     : {lambda_method}"
          + (f" (lambda = {lambda_reg:.2e})"
             if lambda_method == "custom" else ""))
    print(f"  Reg. derivative   : {reg_order}{'st' if reg_order == 1 else 'nd'} order")
    print(f"  Data use          : {data_use}")
    print(f"  Inductance        : {inductance_mode}")
    print(f"  FWHM coefficient  : {fwhm_coeff}")
    print(f"  Non-negativity    : {'Yes' if non_negative else 'No'}")
    print(f"{C.GREEN}{'-' * 60}{C.RESET}")

    return settings


def plot_drt_results(freq, zreal, zimag, drt_result, sign_conv,
                     save_path=None, data_file=None):
    """
    Plot DRT analysis results in a 3-row layout (similar to ECM fit):
      Top row:      Settings table (left) + empty (right)
      Middle row:   DRT spectrum (left) + Nyquist (right)  [larger]
      Bottom row:   Bode (left) + Residuals (right)        [smaller]
    """
    gamma = drt_result["gamma"]
    log_tau = drt_result["log_tau"]
    Z_fit = drt_result["Z_fit"]
    R_inf = drt_result["R_inf"]
    R_pol = drt_result["R_pol"]
    lam = drt_result["lambda_used"]
    rmse = drt_result["rmse"]
    r2 = drt_result["r2"]

    omega = 2.0 * np.pi * freq
    tau_grid = 10.0 ** log_tau
    f_tau = 1.0 / (2.0 * np.pi * tau_grid)  # corresponding freq

    # ---- Adaptive figure size ----
    fig_w = 19
    fig_h = 11.5
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = GridSpec(3, 2, figure=fig,
                  height_ratios=[0.40, 1.40, 0.55],
                  hspace=0.35, wspace=0.28)

    # ---- Suptitle with file name ----
    _title_file = ''
    if data_file:
        _title_file = f'\n{os.path.basename(data_file)}'
    fig.suptitle(
        f"Distribution of Relaxation Times (DRT) Analysis"
        f"{_title_file}",
        fontsize=14, fontweight='bold',
        x=0.50, ha='center', va='top')

    # ---- Top-right: Settings table (no header) ----
    ax_tbl = fig.add_subplot(gs[0, 1])
    ax_tbl.set_axis_off()

    # Build table data — DRT settings and fit parameters
    lam_method = drt_result.get("lambda_method", "N/A")
    reg_order = drt_result.get("reg_order", 2)
    data_use = drt_result.get("data_use", "combined")
    inductance = drt_result.get("inductance", "N/A")
    rbf_type = drt_result.get("rbf_type", "gaussian")
    fwhm_coeff = drt_result.get("fwhm_coeff", 0.5)
    non_neg = drt_result.get("non_negative", True)
    L_ind = drt_result.get("L_ind", 0.0)

    ord_label = f"{reg_order}{'st' if reg_order == 1 else 'nd'} derivative"
    lam_display = f"{lam_method}"
    if lam_method == "custom":
        lam_display += f" (\u03bb = {lam:.2e})"
    else:
        lam_display += f" \u2192 \u03bb = {lam:.2e}"

    tbl_data = [
        ['Model',             'DRT (Tikhonov + Gaussian RBF)'],
        ['\u03bb selection',  lam_display],
        ['Regularisation',    ord_label],
        ['Data use',          data_use],
        ['Inductance',        inductance],
        ['FWHM coeff.',       f'{fwhm_coeff}'],
        ['Non-negativity',    'Yes' if non_neg else 'No'],
        ['R\u221e',           f'{R_inf:.6f} \u03a9'],
        ['R_pol',             f'{R_pol:.6f} \u03a9'],
        ['L_ind',             f'{L_ind:.4e} H'],
        ['RMSE',              f'{rmse:.6f} \u03a9'],
        ['R\u00b2',           f'{r2:.6f}'],
    ]

    tbl = ax_tbl.table(
        cellText=tbl_data,
        cellLoc='center',
        loc='center',
        colWidths=[0.35, 0.62])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.25)

    # Style the table (no header row)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor('#cccccc')
        if row % 2 == 0:
            cell.set_facecolor('#f2f3f4')
        else:
            cell.set_facecolor('#ffffff')
        if col == 0:
            cell.set_text_props(fontweight='bold', fontsize=8)

    # ---- Top-left: empty ----
    ax_tbl_l = fig.add_subplot(gs[0, 0])
    ax_tbl_l.set_axis_off()

    # ---- Middle & bottom subplots ----
    ax1 = fig.add_subplot(gs[1, 0])   # DRT spectrum
    ax2 = fig.add_subplot(gs[1, 1])   # Nyquist
    ax3 = fig.add_subplot(gs[2, 0])   # Bode
    ax4 = fig.add_subplot(gs[2, 1])   # Residuals

    # ---- 1) DRT spectrum ----
    ax1.fill_between(log_tau, 0, gamma, color="steelblue", alpha=0.3)
    ax1.plot(log_tau, gamma, "-", lw=2, color="navy",
             label=r"$\gamma(\log\tau)$")
    ax1.set_xlabel(r"$\log_{10}(\tau\,/\,\mathrm{s})$", fontsize=12)
    ax1.set_ylabel(r"$\gamma\;(\Omega)$", fontsize=12)
    ax1.set_title("DRT Spectrum", fontsize=12, fontweight="bold")
    ax1.set_xlim(log_tau[0], log_tau[-1])
    ax1.set_ylim(bottom=0, top=gamma.max() * 1.35 if gamma.max() > 0 else 1)
    ax1.grid(True, alpha=0.3, ls="--")

    # Add secondary x-axis showing frequency
    ax1_top = ax1.twiny()
    ax1_top.set_xlim(ax1.get_xlim())
    # Place tick labels at a few representative positions
    tick_positions = np.linspace(log_tau[0], log_tau[-1], 7)
    tick_freqs = 1.0 / (2.0 * np.pi * 10.0 ** tick_positions)
    ax1_top.set_xticks(tick_positions)
    ax1_top.set_xticklabels([eng_str(f, "Hz") for f in tick_freqs],
                             fontsize=7)
    ax1_top.set_xlabel("Frequency", fontsize=9, color="gray")

    # Mark peaks
    from scipy.signal import find_peaks as _find_peaks
    peak_indices, peak_props = _find_peaks(
        gamma, height=0.05 * gamma.max(), distance=5, prominence=0.02 * gamma.max())
    peak_colors_drt = plt.cm.tab10(np.linspace(0, 0.9,
                                                max(len(peak_indices), 1)))
    for ip, pi in enumerate(peak_indices):
        f_peak = 1.0 / (2.0 * np.pi * tau_grid[pi])
        ax1.annotate(
            f"f={eng_str(f_peak, 'Hz')}\n"
            f"\u03c4={tau_grid[pi]:.2e} s\n"
            f"\u03b3={gamma[pi]:.4f} \u03a9",
            (log_tau[pi], gamma[pi]),
            fontsize=7, fontweight="bold",
            color=peak_colors_drt[ip],
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            arrowprops=dict(arrowstyle="->",
                            color=peak_colors_drt[ip],
                            lw=1.2),
            bbox=dict(boxstyle="round,pad=0.2",
                      facecolor="white", alpha=0.8))

    # Info box
    info_txt = (f"R\u221e = {R_inf:.4f} \u03a9\n"
                f"R_pol = {R_pol:.4f} \u03a9\n"
                f"\u03bb = {lam:.2e}\n"
                f"RMSE = {rmse:.4e}\n"
                f"R\u00b2 = {r2:.6f}\n"
                f"Peaks: {len(peak_indices)}")
    ax1.text(0.98, 0.98, info_txt, transform=ax1.transAxes,
             fontsize=7, ha="right", va="top",
             fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85))

    ax1.legend(fontsize=9, loc="upper left", framealpha=0.8)

    # ---- 2) Nyquist ----
    y_meas = _nyquist_sign(zimag, sign_conv)
    y_fit = _nyquist_sign(Z_fit.imag, sign_conv)

    ax2.plot(zreal, y_meas, "o", ms=6, mfc="none", mec="dimgray",
             mew=1.5, label="Measured", zorder=2, alpha=0.85)
    ax2.plot(Z_fit.real, y_fit, "-", lw=2.5, color="crimson",
             label="DRT Fit", zorder=5)

    ylabel_nq = "-Z'' (\u03a9)" if sign_conv == "negative" else "Z'' (\u03a9)"
    ax2.set_xlabel("Z' (\u03a9)", fontsize=11)
    ax2.set_ylabel(ylabel_nq, fontsize=11)
    ax2.set_title("Nyquist Plot (DRT Fit)", fontsize=11, fontweight="bold")
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.grid(True, alpha=0.3, ls="--")
    ax2.legend(fontsize=8, loc="upper right", framealpha=0.8)
    apply_eng_format(ax2, "both")

    # Lock Nyquist axes to measured data range
    x_pad = (zreal.max() - zreal.min()) * 0.05 + 1e-9
    y_pad = (y_meas.max() - y_meas.min()) * 0.05 + 1e-9
    ax2.set_xlim(zreal.min() - x_pad, zreal.max() + x_pad)
    ax2.set_ylim(y_meas.min() - y_pad, y_meas.max() + y_pad)

    _annotate_freq_on_nyquist(ax2, freq, zreal, y_meas)

    # ---- 3) Bode ----
    Zmod_m = np.sqrt(zreal ** 2 + zimag ** 2)
    Zph_m = np.degrees(np.arctan2(zimag, zreal))
    Zmod_f = np.abs(Z_fit)
    Zph_f = np.degrees(np.arctan2(Z_fit.imag, Z_fit.real))

    ax3.set_xlabel("Frequency (Hz)", fontsize=11)
    ax3.set_xscale("log")
    ax3.set_ylabel("|Z| (\u03a9)", fontsize=11, color="tab:blue")
    ax3.loglog(freq, Zmod_m, "o", ms=4, color="tab:blue",
               label="|Z| meas", alpha=0.6)
    ax3.loglog(freq, Zmod_f, "-", lw=2.5, color="navy",
               label="|Z| DRT fit")
    ax3.tick_params(axis="y", labelcolor="tab:blue")

    ax3b = ax3.twinx()
    ax3b.set_ylabel("Phase (\u00b0)", fontsize=11, color="tab:red")
    ax3b.semilogx(freq, Zph_m, "s", ms=3, color="tab:red",
                  label="Phase meas", alpha=0.6)
    ax3b.semilogx(freq, Zph_f, "-", lw=2.5, color="darkred",
                  label="Phase DRT")
    ax3b.tick_params(axis="y", labelcolor="tab:red")

    ax3.set_title("Bode Plot (DRT Fit)", fontsize=11, fontweight="bold")
    ax3.grid(True, alpha=0.3, ls="--")
    h1, l1 = ax3.get_legend_handles_labels()
    h2, l2 = ax3b.get_legend_handles_labels()
    ax3.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8, framealpha=0.8)
    apply_eng_format(ax3, "x")
    apply_eng_format(ax3, "y")

    # ---- 4) Residuals ----
    res_real = zreal - Z_fit.real
    res_imag = zimag - Z_fit.imag
    Z_mag = np.sqrt(zreal ** 2 + zimag ** 2)
    Z_mag_safe = np.where(Z_mag > 1e-15, Z_mag, 1e-15)
    res_real_pct = 100.0 * res_real / Z_mag_safe
    res_imag_pct = 100.0 * res_imag / Z_mag_safe

    ax4.semilogx(freq, res_real_pct, "o-", ms=3, color="steelblue",
                 label="Z' residual %")
    ax4.semilogx(freq, res_imag_pct, "s-", ms=3, color="tomato",
                 label="Z'' residual %")
    ax4.axhline(0, color="black", lw=0.8, ls="--")
    ax4.axhline(1.0, color="gray", lw=0.6, ls=":", alpha=0.5)
    ax4.axhline(-1.0, color="gray", lw=0.6, ls=":", alpha=0.5)
    ax4.set_xlabel("Frequency (Hz)", fontsize=11)
    ax4.set_ylabel("Relative Residual (%)", fontsize=11)
    ax4.set_title("DRT Fit Residuals", fontsize=11, fontweight="bold")
    ax4.legend(fontsize=8, loc="upper right", framealpha=0.8)
    ax4.grid(True, alpha=0.3, ls="--")
    apply_eng_format(ax4, "x")

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        success(f"DRT plot saved to: {save_path}")

    plt.show(block=True)
    plt.close(fig)


def export_drt_csv(filepath, freq, zreal, zimag, drt_result):
    """Export DRT results to CSV."""
    gamma = drt_result["gamma"]
    log_tau = drt_result["log_tau"]
    tau = drt_result["tau"]
    Z_fit = drt_result["Z_fit"]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([f"# EIS Fitter {__version__} - DRT Analysis Results"])
        w.writerow([f"# R_inf = {drt_result['R_inf']:.6e} Ohm"])
        w.writerow([f"# R_pol = {drt_result['R_pol']:.6e} Ohm"])
        w.writerow([f"# L_ind = {drt_result['L_ind']:.6e} H"])
        w.writerow([f"# Lambda = {drt_result['lambda_used']:.6e}"])
        w.writerow([f"# RMSE = {drt_result['rmse']:.6e} Ohm"])
        w.writerow([f"# R^2  = {drt_result['r2']:.6f}"])
        w.writerow([f"# Reg. order = {drt_result['reg_order']}"])
        w.writerow([f"# Data use = {drt_result['data_use']}"])
        w.writerow([f"# FWHM coeff = {drt_result['fwhm_coeff']}"])
        w.writerow([])

        # Part 1: DRT spectrum
        w.writerow(["# --- DRT Spectrum ---"])
        w.writerow(["log_tau", "tau_s", "gamma_Ohm", "f_Hz"])
        for i in range(len(gamma)):
            f_i = 1.0 / (2.0 * np.pi * tau[i])
            w.writerow([f"{log_tau[i]:.6e}", f"{tau[i]:.6e}",
                         f"{gamma[i]:.6e}", f"{f_i:.6e}"])
        w.writerow([])

        # Part 2: Impedance fit
        w.writerow(["# --- Impedance Fit ---"])
        w.writerow(["Freq_Hz", "Zreal_meas", "Zimag_meas",
                     "Zreal_DRT_fit", "Zimag_DRT_fit",
                     "Residual_real", "Residual_imag"])
        for i in range(len(freq)):
            w.writerow([
                f"{freq[i]:.6e}", f"{zreal[i]:.6e}", f"{zimag[i]:.6e}",
                f"{Z_fit.real[i]:.6e}", f"{Z_fit.imag[i]:.6e}",
                f"{zreal[i] - Z_fit.real[i]:.6e}",
                f"{zimag[i] - Z_fit.imag[i]:.6e}",
            ])
    success(f"DRT CSV exported to: {filepath}")


def save_drt_results_txt(filepath, chosen_file, freq, drt_result):
    """Save DRT results summary to text file."""
    from scipy.signal import find_peaks as _find_peaks

    gamma = drt_result["gamma"]
    log_tau = drt_result["log_tau"]
    tau = drt_result["tau"]

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"EIS Fitter {__version__} - DRT ANALYSIS RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Data file  : {os.path.relpath(chosen_file, '.')}\n")
        f.write(f"Data points: {len(freq)}\n")
        f.write(f"Freq range : {eng_str(freq.min(), 'Hz')} - "
                f"{eng_str(freq.max(), 'Hz')}\n\n")

        f.write("SETTINGS:\n")
        f.write(f"  Lambda method    : {drt_result.get('lambda_method', 'N/A')}\n")
        f.write(f"  Lambda used      : {drt_result['lambda_used']:.6e}\n")
        f.write(f"  Reg. order       : {drt_result['reg_order']}\n")
        f.write(f"  Data use         : {drt_result['data_use']}\n")
        f.write(f"  Inductance       : {drt_result['inductance']}\n")
        f.write(f"  RBF type         : {drt_result['rbf_type']}\n")
        f.write(f"  FWHM coefficient : {drt_result['fwhm_coeff']}\n")
        f.write(f"  Non-negativity   : {drt_result['non_negative']}\n\n")

        f.write("RESULTS:\n")
        f.write(f"  R_inf            : {drt_result['R_inf']:.6f} Ohm\n")
        f.write(f"  R_pol            : {drt_result['R_pol']:.6f} Ohm\n")
        f.write(f"  L_ind            : {drt_result['L_ind']:.6e} H\n")
        f.write(f"  RMSE             : {drt_result['rmse']:.6f} Ohm\n")
        f.write(f"  R^2              : {drt_result['r2']:.6f}\n\n")

        # Peak analysis
        peak_idx, _ = _find_peaks(
            gamma, height=0.05 * gamma.max(), distance=5,
            prominence=0.02 * gamma.max())
        f.write(f"IDENTIFIED PEAKS ({len(peak_idx)}):\n")
        for ip, pi in enumerate(peak_idx):
            f_peak = 1.0 / (2.0 * np.pi * tau[pi])
            f.write(f"  Peak {ip + 1}: "
                    f"tau = {tau[pi]:.4e} s, "
                    f"f = {eng_str(f_peak, 'Hz')}, "
                    f"gamma = {gamma[pi]:.6f} Ohm\n")
        f.write("\n" + "=" * 60 + "\n")
    success(f"DRT results saved to: {filepath}")


def log_drt_action(log_path, chosen_file, freq, drt_result):
    """Log DRT analysis to the persistent fit log."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("-" * 70 + "\n")
        f.write(f"[{now}]\n")
        f.write(f"File   : {os.path.relpath(chosen_file, '.')}\n")
        f.write(f"Method : DRT Analysis (Tikhonov + Gaussian RBF)\n")
        f.write(f"Freq   : {eng_str(freq.min(), 'Hz')} - "
                f"{eng_str(freq.max(), 'Hz')}  ({len(freq)} pts)\n")
        f.write(f"  Lambda     = {drt_result['lambda_used']:.4e}\n")
        f.write(f"  R_inf      = {drt_result['R_inf']:.6f} Ohm\n")
        f.write(f"  R_pol      = {drt_result['R_pol']:.6f} Ohm\n")
        f.write(f"  RMSE       = {drt_result['rmse']:.6f} Ohm\n")
        f.write(f"  R^2        = {drt_result['r2']:.6f}\n")
        f.write("\n")
    info(f"DRT result logged to: {log_path}")


# ============================================================
# SECTION 16: Main
# ============================================================

# ============================================================
# SECTION 16-A: Manual Fitting (peak-guided initial guess)
# ============================================================
#
# The user visually identifies semi-circle peaks on the Nyquist
# plot.  Each clicked point provides:
#   - The y-coordinate (-Z'') gives an estimate of R/2 for that arc
#     (exact for alpha=1; a reasonable starting point otherwise).
#   - The nearest data frequency gives f_char, hence tau = 1/(2*pi*f_char).
#   - The x-coordinate (Z') constrains the offset / Rs.
#
# These estimates are used as a warm-start for a bounded local
# optimiser (L-BFGS-B) plus a narrow-range Differential Evolution
# to refine the fit, which converges much faster than the blind
# global search because the initial guess is already in the right
# basin of attraction.
# ============================================================

def select_semicircle_peaks(freq, zreal, zimag, sign_conv, n_peaks=None):
    """
    Show Nyquist plot; user clicks on the tops of semi-circles.

    If n_peaks is None (default), the user can click 1-10 peaks freely
    and close the window when done.  If n_peaks is an integer, exactly
    that many clicks are expected.

    Right-click undoes the last clicked peak.

    Returns a list of dicts with keys:
      'idx'       - nearest data-point index
      'freq'      - frequency at that point
      'zreal'     - Z' at that point
      'zimag_abs' - |Z''| at that point (always positive = arc height)
    """
    max_peaks = n_peaks if n_peaks is not None else 10
    free_mode = n_peaks is None  # user decides how many

    y_nq = _nyquist_sign(zimag, sign_conv)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(zreal, y_nq, "o-", ms=6, color="royalblue",
            mfc="white", mec="royalblue", lw=1, picker=5)
    ax.set_xlabel("Z' (Ohm)", fontsize=12)
    ylabel = "-Z'' (Ohm)" if sign_conv == "negative" else "Z'' (Ohm)"
    ax.set_ylabel(ylabel, fontsize=12)

    if free_mode:
        ax.set_title(
            "MANUAL FIT: Click on the TOP of each semi-circle\n"
            "(1-10 peaks, right-click to undo, close window when done)",
            fontsize=13, fontweight="bold")
    else:
        ax.set_title(
            f"Click on the TOP of {n_peaks} semi-circle(s)\n"
            f"(right-click to undo last peak)",
            fontsize=13, fontweight="bold")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3, ls="--")
    apply_eng_format(ax, "both")

    # Annotate frequencies
    step = max(1, len(freq) // 15)
    for i in range(0, len(freq), step):
        ax.annotate(
            eng_str(freq[i], "Hz"), (zreal[i], y_nq[i]),
            fontsize=7.5, color="dimgray", alpha=0.85,
            textcoords="offset points", xytext=(5, 5))

    if free_mode:
        init_msg = ("Left-click on semi-circle peaks (1-10).\n"
                    "Right-click to undo. Close window when done.")
    else:
        init_msg = f"Click peak 1 of {max_peaks}"

    instruction_text = ax.text(
        0.5, 0.02, init_msg,
        transform=ax.transAxes, ha="center", fontsize=11,
        fontweight="bold", color="red",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                  alpha=0.9))

    # Counter text in top-right corner showing how many peaks selected
    counter_text = ax.text(
        0.98, 0.98, "Peaks: 0",
        transform=ax.transAxes, ha="right", va="top", fontsize=12,
        fontweight="bold", color="purple",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  alpha=0.85))

    fig.tight_layout()

    clicked_peaks = []       # list of data-point indices
    peak_artists = []        # list of (marker_artist, line_artist) for undo
    peak_colors = plt.cm.tab10(np.linspace(0, 0.9, max_peaks))

    def find_nearest(cx, cy):
        x_range = max(zreal.max() - zreal.min(), 1e-12)
        y_range = max(y_nq.max() - y_nq.min(), 1e-12)
        dist = ((zreal - cx) / x_range)**2 + ((y_nq - cy) / y_range)**2
        return np.argmin(dist)

    def update_instruction():
        n = len(clicked_peaks)
        counter_text.set_text(f"Peaks: {n}")
        if free_mode:
            if n == 0:
                instruction_text.set_text(
                    "Left-click on semi-circle peaks (1-10).\n"
                    "Right-click to undo. Close window when done.")
                instruction_text.set_color("red")
            elif n < max_peaks:
                instruction_text.set_text(
                    f"{n} peak(s) selected. Click more or "
                    f"close window to fit.\n"
                    f"Right-click to undo last peak.")
                instruction_text.set_color("darkorange")
            else:
                instruction_text.set_text(
                    f"Maximum {max_peaks} peaks selected! "
                    f"Close window to continue.")
                instruction_text.set_color("darkgreen")
        else:
            remaining = max_peaks - n
            if remaining > 0:
                instruction_text.set_text(
                    f"Click peak {n + 1} of {max_peaks}")
                instruction_text.set_color("red")
            else:
                instruction_text.set_text(
                    f"All {max_peaks} peak(s) selected! "
                    f"Close window to continue.")
                instruction_text.set_color("darkgreen")

    def on_click(event):
        if event.inaxes != ax:
            return

        # Right-click = undo last peak
        if event.button == 3 and len(clicked_peaks) > 0:
            clicked_peaks.pop()
            marker_art, line_art = peak_artists.pop()
            marker_art.remove()
            line_art[0].remove()
            # Rebuild legend
            ax.legend(fontsize=8, loc="upper right", framealpha=0.8)
            update_instruction()
            fig.canvas.draw_idle()
            return

        # Left-click = add peak
        if event.button != 1:
            return
        if len(clicked_peaks) >= max_peaks:
            return

        idx = find_nearest(event.xdata, event.ydata)
        peak_num = len(clicked_peaks) + 1
        color = peak_colors[peak_num - 1]

        clicked_peaks.append(idx)

        marker = ax.plot(zreal[idx], y_nq[idx], "v", ms=14, color=color,
                         markeredgecolor="black", markeredgewidth=1.5,
                         label=f"Peak {peak_num}: "
                               f"{eng_str(freq[idx], 'Hz')}",
                         zorder=10)
        vline = ax.plot([zreal[idx], zreal[idx]], [0, y_nq[idx]],
                        "--", lw=1, color=color, alpha=0.5, zorder=5)
        peak_artists.append((marker[0], vline))
        ax.legend(fontsize=8, loc="upper right", framealpha=0.8)
        update_instruction()
        fig.canvas.draw_idle()

    cid = fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show(block=True)
    fig.canvas.mpl_disconnect(cid)
    plt.close(fig)

    # Build peak info list
    peaks = []
    for idx in clicked_peaks:
        peaks.append({
            "idx": idx,
            "freq": freq[idx],
            "zreal": zreal[idx],
            "zimag_abs": abs(zimag[idx]),
        })

    # Sort peaks by frequency descending (highest freq = fastest tau first)
    peaks.sort(key=lambda p: p["freq"], reverse=True)
    return peaks


# ============================================================
# SECTION 16-B: Bode Phase Analysis (phase-peak-guided fitting)
# ============================================================
#
# Each ZARC element produces a local extremum (dip / most negative)
# in the phase angle at its characteristic frequency f_c = 1/(2πτ).
# The user clicks on these convex inflection points on the Bode
# phase plot.  From each clicked point:
#   - The x-coordinate gives f_char → tau = 1/(2π f_char).
#   - The corresponding |Z''| at that frequency gives R ≈ 2|Z''|.
#
# The resulting peak-info list has the same format as the Nyquist
# manual mode, so manual_fit_eis() is reused for the actual fit.
# ============================================================

def select_bode_phase_peaks(freq, zreal, zimag, sign_conv):
    """
    Show Bode phase plot; user clicks on the convex points (local
    phase extrema) that correspond to individual RC arcs.  1-10
    peaks, right-click to undo, close window when done.

    Returns a list of dicts compatible with manual_fit_eis():
      'idx', 'freq', 'zreal', 'zimag_abs'
    """
    max_peaks = 10

    # Compute phase angle in degrees
    phase_deg = np.degrees(np.arctan2(zimag, zreal))

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.semilogx(freq, phase_deg, "o-", ms=5, color="darkred",
                mfc="white", mec="darkred", lw=1.2, picker=5)
    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("Phase Angle (°)", fontsize=12)
    ax.set_title(
        "BODE PHASE ANALYSIS: Click on convex points "
        "(phase dips / extrema)\n"
        "(1-10 peaks, right-click to undo, close window when done)",
        fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, ls="--")
    apply_eng_format(ax, "x")

    # Add a secondary y-axis showing |Z| for reference
    ax2 = ax.twinx()
    Zmod = np.sqrt(zreal**2 + zimag**2)
    ax2.loglog(freq, Zmod, "-", lw=1, color="steelblue", alpha=0.3,
               label="|Z|")
    ax2.set_ylabel("|Z| (Ohm)", fontsize=10, color="steelblue",
                   alpha=0.5)
    ax2.tick_params(axis="y", labelcolor="steelblue")
    # Make twin axis transparent to mouse events so clicks reach ax
    ax2.set_zorder(ax.get_zorder() - 1)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)  # let ax2 background show through

    instruction_text = ax.text(
        0.5, 0.02,
        "Left-click on phase dips/extrema (1-10).\n"
        "Right-click to undo. Close window when done.",
        transform=ax.transAxes, ha="center", fontsize=11,
        fontweight="bold", color="red",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                  alpha=0.9))

    counter_text = ax.text(
        0.98, 0.98, "Peaks: 0",
        transform=ax.transAxes, ha="right", va="top", fontsize=12,
        fontweight="bold", color="purple",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  alpha=0.85))

    fig.tight_layout()

    clicked_peaks = []
    peak_artists = []
    peak_colors = plt.cm.tab10(np.linspace(0, 0.9, max_peaks))

    # Use log-frequency for nearest-point search (x is log-scaled)
    log_freq = np.log10(np.clip(freq, 1e-30, None))

    def find_nearest(cx, cy):
        """Find nearest data point in log-freq vs phase space."""
        log_cx = np.log10(max(cx, 1e-30))
        x_range = max(log_freq.max() - log_freq.min(), 1e-12)
        y_range = max(phase_deg.max() - phase_deg.min(), 1e-12)
        dist = ((log_freq - log_cx) / x_range)**2 + \
               ((phase_deg - cy) / y_range)**2
        return np.argmin(dist)

    def update_instruction():
        n = len(clicked_peaks)
        counter_text.set_text(f"Peaks: {n}")
        if n == 0:
            instruction_text.set_text(
                "Left-click on phase dips/extrema (1-10).\n"
                "Right-click to undo. Close window when done.")
            instruction_text.set_color("red")
        elif n < max_peaks:
            instruction_text.set_text(
                f"{n} peak(s) selected. Click more or "
                f"close window to fit.\n"
                f"Right-click to undo last peak.")
            instruction_text.set_color("darkorange")
        else:
            instruction_text.set_text(
                f"Maximum {max_peaks} peaks selected! "
                f"Close window to continue.")
            instruction_text.set_color("darkgreen")

    def on_click(event):
        if event.inaxes != ax:
            return

        # Right-click = undo
        if event.button == 3 and len(clicked_peaks) > 0:
            clicked_peaks.pop()
            marker_art, vline_art = peak_artists.pop()
            marker_art.remove()
            vline_art[0].remove()
            ax.legend(fontsize=8, loc="upper left", framealpha=0.8)
            update_instruction()
            fig.canvas.draw_idle()
            return

        if event.button != 1:
            return
        if len(clicked_peaks) >= max_peaks:
            return

        idx = find_nearest(event.xdata, event.ydata)
        peak_num = len(clicked_peaks) + 1
        color = peak_colors[peak_num - 1]

        clicked_peaks.append(idx)

        marker = ax.plot(
            freq[idx], phase_deg[idx], "v", ms=14, color=color,
            markeredgecolor="black", markeredgewidth=1.5,
            label=f"Peak {peak_num}: {eng_str(freq[idx], 'Hz')} "
                  f"({phase_deg[idx]:.1f}°)",
            zorder=10)
        vline = ax.plot(
            [freq[idx], freq[idx]],
            [ax.get_ylim()[0], phase_deg[idx]],
            "--", lw=1, color=color, alpha=0.5, zorder=5)
        peak_artists.append((marker[0], vline))
        ax.legend(fontsize=8, loc="upper left", framealpha=0.8)
        update_instruction()
        fig.canvas.draw_idle()

    cid = fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show(block=True)
    fig.canvas.mpl_disconnect(cid)
    plt.close(fig)

    # Build peak info list (same format as Nyquist manual mode)
    peaks = []
    for idx in clicked_peaks:
        peaks.append({
            "idx": idx,
            "freq": freq[idx],
            "zreal": zreal[idx],
            "zimag_abs": abs(zimag[idx]),
        })

    # Sort by frequency descending (highest freq = fastest tau first)
    peaks.sort(key=lambda p: p["freq"], reverse=True)
    return peaks


def manual_fit_eis(freq, zreal, zimag, peaks, n_voigt,
                   maxiter=400, popsize=25, n_restarts=5):
    """
    Fit EIS data using user-selected semi-circle peaks as initial guesses.

    Strategy:
      1. Derive R_i, tau_i, alpha_i estimates from each clicked peak.
      2. Build tight bounds around those estimates.
      3. Run Differential Evolution in that narrowed search space
         (converges fast because the basin is already identified).
      4. Polish with L-BFGS-B.

    Returns: params, Z_fit, Z_base, voigt_list, metrics
             (same format as fit_eis)
    """
    freq, zreal, zimag = validate_spectrum(freq, zreal, zimag)
    omega = 2.0 * np.pi * freq
    Rs_est = np.min(np.abs(zreal))

    # ---- Derive initial guesses from peaks ----
    R_guesses = []
    tau_guesses = []
    for p in peaks:
        # Use an ideal-RC peak height as a heuristic starting estimate.
        # Overlapping arcs and CPE depression can bias this estimate.
        R_est = 2.0 * p["zimag_abs"]
        R_est = max(R_est, 1e-6)  # safety floor
        tau_est = 1.0 / (2.0 * np.pi * p["freq"])
        R_guesses.append(R_est)
        tau_guesses.append(tau_est)

    info(f"Manual fitting with {n_voigt} Voigt element(s) "
         f"from {len(peaks)} peak(s) ...")
    for i, p in enumerate(peaks):
        info(f"  Peak {i+1}: f_char ~ {eng_str(p['freq'], 'Hz')}, "
             f"R_est ~ {eng_str(R_guesses[i], 'Ohm')}, "
             f"tau_est ~ {eng_str(tau_guesses[i], 's')}")

    # ---- Build TIGHT bounds around estimates ----
    # L and Rs get reasonable ranges
    R_total = np.ptp(zreal)
    if R_total <= 0:
        R_total = 1.0

    bounds = [
        (0, 1e-3),                                         # L
        (max(Rs_est * 0.3, 0), max(Rs_est * 2.0, 0.01)),  # Rs
    ]
    for i in range(n_voigt):
        R_lo = R_guesses[i] * 0.05
        R_hi = R_guesses[i] * 10.0
        R_hi = max(R_hi, R_total * 2.0)  # don't clip too aggressively

        tau_lo = tau_guesses[i] * 0.001
        tau_hi = tau_guesses[i] * 1000.0
        # Clamp to physically sensible range
        tau_lo = max(tau_lo, 1e-12)
        tau_hi = min(tau_hi, 1e6)

        bounds.append((max(R_lo, 1e-8), R_hi))     # R_i
        bounds.append((tau_lo, tau_hi))             # tau_i
        bounds.append((0.7, 1.0))                   # alpha_i

    n_params = 2 + 3 * n_voigt

    # ---- Build initial guess vector (for seeding DE population) ----
    x0 = [1e-7, Rs_est]  # L, Rs
    for i in range(n_voigt):
        x0.extend([R_guesses[i], tau_guesses[i], 0.85])
    x0 = np.array(x0)

    # Clip x0 to bounds
    for j in range(len(x0)):
        x0[j] = np.clip(x0[j], bounds[j][0], bounds[j][1])

    info(f"Narrow-range DE + polish ({n_restarts} restarts, "
         f"{n_params} params) ...")

    best_params_tau = None
    best_cost = np.inf

    for restart in range(n_restarts):
        seed = 42 + restart * 13
        rng = np.random.RandomState(seed)  # one RNG per restart

        # Create a custom init population seeded around x0
        # DE population: each member is a perturbation of x0
        n_pop = popsize * n_params
        init_pop = np.zeros((n_pop, n_params))
        for member in range(n_pop):
            for j in range(n_params):
                lo, hi = bounds[j]
                if member == 0:
                    # First member is exactly x0
                    init_pop[member, j] = x0[j]
                else:
                    # Random perturbation in log-space for R and tau,
                    # linear for alpha and small params
                    if j >= 2 and (j - 2) % 3 in (0, 1):
                        # R or tau: perturb in log-space
                        log_val = np.log10(max(x0[j], 1e-15))
                        log_lo = np.log10(max(lo, 1e-15))
                        log_hi = np.log10(max(hi, 1e-15))
                        log_pert = log_val + rng.uniform(-0.5, 0.5)
                        log_pert = np.clip(log_pert, log_lo, log_hi)
                        init_pop[member, j] = 10**log_pert
                    else:
                        # Linear perturbation
                        pert = x0[j] + rng.uniform(-1, 1) * (hi - lo) * 0.15
                        init_pop[member, j] = np.clip(pert, lo, hi)

        try:
            result = differential_evolution(
                objective_tau, bounds,
                args=(omega, zreal, zimag, n_voigt, "none"),
                maxiter=maxiter, popsize=popsize, tol=1e-8,
                seed=seed, disp=False, init=init_pop,
                workers=-1, updating="deferred",
            )
        except Exception:
            result = differential_evolution(
                objective_tau, bounds,
                args=(omega, zreal, zimag, n_voigt, "none"),
                maxiter=maxiter, popsize=popsize, tol=1e-8,
                seed=seed, disp=False, init=init_pop,
                workers=1,
            )

        # Polish with L-BFGS-B (soft penalty for smooth gradients)
        result2 = minimize(
            objective_tau, result.x,
            args=(omega, zreal, zimag, n_voigt, "soft"),
            method="L-BFGS-B", bounds=bounds,
        )

        # Also polish with no penalty
        result3 = minimize(
            objective_tau, result.x,
            args=(omega, zreal, zimag, n_voigt, "none"),
            method="L-BFGS-B", bounds=bounds,
        )

        # Also try starting polish directly from x0
        result4 = minimize(
            objective_tau, x0,
            args=(omega, zreal, zimag, n_voigt, "none"),
            method="L-BFGS-B", bounds=bounds,
        )

        candidates = [
            (result.fun, result.x),
            (result2.fun, result2.x),
            (result3.fun, result3.x),
            (result4.fun, result4.x),
        ]
        cost, params = min(candidates, key=lambda c: c[0])

        is_new_best = cost < best_cost
        if is_new_best:
            best_cost = cost
            best_params_tau = params.copy()

        print(f"    restart {restart + 1}/{n_restarts} [manual]: "
              f"cost = {cost:.4e}"
              + (" *best*" if is_new_best else ""))

    # Convert tau -> Q, sort by tau, compute metrics (using shared helpers)
    best_params = _convert_tau_to_Q(best_params_tau, n_voigt)
    best_params = sort_voigt_params(best_params, n_voigt)
    Z_fit, Z_base, voigt_list = Z_model_components(
        omega, best_params, n_voigt)
    metrics = _compute_fit_metrics(zreal, zimag, Z_fit, best_cost, n_params)
    info(f"Manual fit best: cost = {best_cost:.4e},  "
         f"RMSE = {metrics['rmse']:.6f} Ohm,  R^2 = {metrics['r2']:.6f}")
    return best_params, Z_fit, Z_base, voigt_list, metrics


def print_fit_menu():
    section("FITTING OPTIONS", C.CYAN)
    print(f"  {C.BOLD}{C.GREEN}[N]{C.RESET}  Enter a number (1-10) "
          f"for fixed Voigt elements")
    print(f"  {C.BOLD}{C.YELLOW}[B]{C.RESET}  BIC auto-scan "
          f"(1-10 elements, pick best by BIC)")
    print(f"  {C.BOLD}{C.YELLOW}[A]{C.RESET}  Auto best-R^2 scan "
          f"(1-10 elements)")
    print(f"  {C.BOLD}{C.HEADER}[M]{C.RESET}  Manual fitting "
          f"(click semi-circle peaks on Nyquist)")
    print(f"  {C.BOLD}{C.HEADER}[P]{C.RESET}  Bode phase analysis "
          f"(click phase dips on Bode plot)")
    print(f"  {C.BOLD}{C.UNDERLINE}[D]{C.RESET}  "
          f"{C.BOLD}DRT analysis{C.RESET}  "
          f"(Distribution of Relaxation Times)")
    print(f"  {C.BOLD}{C.RED}[R]{C.RESET}  Re-select frequency range "
          f"(re-runs KK)")
    print(f"  {C.BOLD}{C.BLUE}[F]{C.RESET}  Load a different data file")
    print(f"  {C.BOLD}{C.DIM}[Q]{C.RESET}  Quit")
    print(f"{C.CYAN}{'-' * 60}{C.RESET}")


def main(data_dir=".", output_dir="EIS_Fit_Results"):
    banner(f"EIS Fitter {__version__}", C.CYAN)
    print(f"  {C.DIM}Model: L + Rs + N x (R || CPE)  "
          f"[overlapping arcs supported]{C.RESET}")
    print(f"  {C.DIM}KK validation + multi-restart DE "
          f"+ engineering notation{C.RESET}")

    # Shared output directory & log file
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "fit_log.txt")

    fit_count = 0        # global counter so file names never collide

    # ================================================================
    # OUTER LOOP: file selection  (restart here when user picks [F])
    # ================================================================
    while True:
        # 1) Find and list .DTA files
        dta_files = find_all_dta_files(data_dir)
        if not dta_files:
            error("No .DTA files found in the current directory tree!")
            return 1
        display_file_list(dta_files)

        # 2) User selects a file
        chosen_file = select_dta_file(dta_files)
        success(f"Selected: {os.path.relpath(chosen_file, '.')}")

        # 3) Read data
        freq, zreal, zimag = read_dta_file(chosen_file)

        # 4) Filter out inductive region
        freq_f, zreal_f, zimag_f, sign_conv = filter_capacitive_region(
            freq, zreal, zimag)

        base_name = os.path.splitext(os.path.basename(chosen_file))[0]

        # 5) Interactive frequency range selection
        info("Select your frequency range of interest on the "
             "Nyquist plot...")
        info("Click START and END points, then close the window.")
        sel_mask = select_frequency_range(
            freq_f, zreal_f, zimag_f, sign_conv,
            title_extra=os.path.basename(chosen_file))

        freq_sel = freq_f[sel_mask]
        zreal_sel = zreal_f[sel_mask]
        zimag_sel = zimag_f[sel_mask]
        validate_spectrum(freq_sel, zreal_sel, zimag_sel)
        success(f"Working with {len(freq_sel)} data points "
                f"({eng_str(freq_sel.min(), 'Hz')} - "
                f"{eng_str(freq_sel.max(), 'Hz')})")

        # 6) Kramers-Kronig validation on selected data
        kk_result = kramers_kronig_test(freq_sel, zreal_sel, zimag_sel)
        log_kk_standalone(log_path, chosen_file, kk_result, freq_sel)

        # ============================================================
        # INNER LOOP: fitting (another method / [R] / [F] / [Q])
        # ============================================================
        switch_file = False
        while True:
            print_fit_menu()

            try:
                user_input = input(
                    f"  {C.BOLD}Your choice: {C.RESET}").strip().lower()
            except EOFError:
                warn("No input available, exiting.")
                print(f"\n  {C.DIM}All fit results logged in: "
                      f"{log_path}{C.RESET}")
                return

            # --- Quit ---
            if user_input in ("q", "quit", "exit"):
                print(f"\n  {C.GREEN}{C.BOLD}Goodbye!{C.RESET}")
                print(f"\n  {C.DIM}All fit results logged in: "
                      f"{log_path}{C.RESET}")
                return

            # --- Load a different data file ---
            if user_input == "f":
                banner("SWITCHING DATA FILE", C.BLUE)
                switch_file = True
                break

            # --- Re-select frequency range ---
            if user_input == "r":
                info("Re-selecting frequency range...")
                sel_mask = select_frequency_range(
                    freq_f, zreal_f, zimag_f, sign_conv,
                    title_extra=os.path.basename(chosen_file))
                freq_sel = freq_f[sel_mask]
                zreal_sel = zreal_f[sel_mask]
                zimag_sel = zimag_f[sel_mask]
                validate_spectrum(freq_sel, zreal_sel, zimag_sel)
                success(
                    f"Now working with {len(freq_sel)} points "
                    f"({eng_str(freq_sel.min(), 'Hz')} - "
                    f"{eng_str(freq_sel.max(), 'Hz')})")
                kk_result = kramers_kronig_test(
                    freq_sel, zreal_sel, zimag_sel)
                log_kk_standalone(log_path, chosen_file,
                                  kk_result, freq_sel)
                continue

            # --- BIC auto-scan ---
            if user_input == "b":
                best_n, best_result, _ = auto_scan_voigt(
                    freq_sel, zreal_sel, zimag_sel,
                    n_min=1, n_max=10)
                params = best_result["params"]
                Z_fit = best_result["Z_fit"]
                Z_base = best_result["Z_base"]
                voigt_list = best_result["voigt_list"]
                metrics = best_result["metrics"]
                n_voigt = best_n
                method_label = f"bic(chose {best_n})"

            # --- Auto best R^2 ---
            elif user_input == "a":
                best_n, best_result, _ = auto_best_r2(
                    freq_sel, zreal_sel, zimag_sel,
                    n_min=1, n_max=10)
                params = best_result["params"]
                Z_fit = best_result["Z_fit"]
                Z_base = best_result["Z_base"]
                voigt_list = best_result["voigt_list"]
                metrics = best_result["metrics"]
                n_voigt = best_n
                method_label = f"auto(best-R^2, chose {best_n})"

            # --- Manual fitting (click semi-circle peaks) ---
            elif user_input == "m":
                banner("MANUAL FITTING (peak-guided)", C.HEADER)
                info("Opening Nyquist plot — click on the TOP of "
                     "each semi-circle peak (1-10).")
                info("Right-click to undo the last peak. "
                     "Close the window when done.")

                # Go directly to interactive peak selection
                # (no terminal prompt — everything is on the plot)
                peaks = select_semicircle_peaks(
                    freq_sel, zreal_sel, zimag_sel,
                    sign_conv)  # n_peaks=None → free mode

                n_voigt = len(peaks)

                if n_voigt == 0:
                    warn("No peaks selected. Aborting manual fit.")
                    continue

                success(f"{n_voigt} peak(s) selected — "
                        f"fitting {n_voigt} Voigt element(s) ...")

                params, Z_fit, Z_base, voigt_list, metrics = \
                    manual_fit_eis(
                        freq_sel, zreal_sel, zimag_sel,
                        peaks, n_voigt)
                method_label = f"manual-peaks(N={n_voigt})"

            # --- Bode phase analysis (click phase dips) ---
            elif user_input == "p":
                banner("BODE PHASE ANALYSIS (phase-guided)",
                       C.HEADER)
                info("Opening Bode phase plot — click on the "
                     "convex points / phase dips (1-10).")
                info("Right-click to undo the last peak. "
                     "Close the window when done.")

                peaks = select_bode_phase_peaks(
                    freq_sel, zreal_sel, zimag_sel,
                    sign_conv)

                n_voigt = len(peaks)

                if n_voigt == 0:
                    warn("No phase peaks selected. "
                         "Aborting Bode phase fit.")
                    continue

                success(f"{n_voigt} phase peak(s) selected — "
                        f"fitting {n_voigt} Voigt element(s) ...")

                params, Z_fit, Z_base, voigt_list, metrics = \
                    manual_fit_eis(
                        freq_sel, zreal_sel, zimag_sel,
                        peaks, n_voigt)
                method_label = f"bode-phase(N={n_voigt})"

            # --- DRT Analysis ---
            elif user_input == "d":
                banner("DRT ANALYSIS (Distribution of Relaxation "
                       "Times)", C.HEADER)
                info("Method: Tikhonov regularisation + Gaussian "
                     "RBF discretisation")
                info("Ref: Wang et al., ACS Electrochem. 2025, "
                     "1, 2680-2689")

                # Prompt for DRT parameters
                drt_settings = _prompt_drt_parameters()

                # For combined/imaginary DRT, include the full
                # unfiltered spectrum (with inductive region)
                # within the user's selected frequency range.
                # The jωL term in the DRT model absorbs the
                # inductive contribution cleanly.  For real-part
                # only, inductive data has no effect so we keep
                # using the capacitive-filtered selection.
                if drt_settings["data_use"] in (
                        "combined", "imaginary"):
                    f_lo = freq_sel.min()
                    f_hi = freq_sel.max()
                    drt_mask = ((freq >= f_lo) &
                                (freq <= f_hi))
                    freq_drt = freq[drt_mask]
                    zreal_drt = zreal[drt_mask]
                    zimag_drt = zimag[drt_mask]
                    info(f"DRT using full spectrum "
                         f"(incl. inductive region): "
                         f"{len(freq_drt)} pts")
                else:
                    freq_drt = freq_sel
                    zreal_drt = zreal_sel
                    zimag_drt = zimag_sel

                # Compute DRT
                info("Computing DRT ...")
                drt_result = compute_drt(
                    freq_drt, zreal_drt, zimag_drt,
                    lambda_reg=drt_settings["lambda_reg"],
                    lambda_method=drt_settings["lambda_method"],
                    rbf_type="gaussian",
                    fwhm_coeff=drt_settings["fwhm_coeff"],
                    reg_order=drt_settings["reg_order"],
                    data_use=drt_settings["data_use"],
                    inductance_mode=drt_settings[
                        "inductance_mode"],
                    non_negative=drt_settings["non_negative"],
                )
                # Store lambda_method in result for logging
                drt_result["lambda_method"] = \
                    drt_settings["lambda_method"]

                # Log DRT
                log_drt_action(log_path, chosen_file,
                               freq_sel, drt_result)

                # Plot DRT results
                fit_count += 1
                drt_plot_path = os.path.join(
                    output_dir,
                    f"{base_name}_DRT_{fit_count}.png")
                info("Showing DRT results ...")
                plot_drt_results(
                    freq_drt, zreal_drt, zimag_drt,
                    drt_result, sign_conv,
                    save_path=drt_plot_path,
                    data_file=chosen_file)

                # Save DRT text results
                drt_txt_path = os.path.join(
                    output_dir,
                    f"{base_name}_DRT_results_{fit_count}.txt")
                save_drt_results_txt(
                    drt_txt_path, chosen_file,
                    freq_drt, drt_result)

                # Export DRT CSV
                drt_csv_path = os.path.join(
                    output_dir,
                    f"{base_name}_DRT_data_{fit_count}.csv")
                export_drt_csv(
                    drt_csv_path, freq_drt, zreal_drt,
                    zimag_drt, drt_result)

                # DRT is a standalone analysis — skip the
                # circuit-fit post-processing below.
                continue

            # --- Fixed number of Voigt elements ---
            else:
                try:
                    n_voigt = max(1, min(10, int(user_input)))
                except ValueError:
                    warn(f"Unknown option '{user_input}'. "
                         f"Enter a number, B, A, M, P, D, R, "
                         f"F, or Q.")
                    continue
                params, Z_fit, Z_base, voigt_list, metrics = fit_eis(
                    freq_sel, zreal_sel, zimag_sel, n_voigt)
                method_label = f"fixed(N={n_voigt})"

            # Params are already sorted by tau inside fit_eis/manual_fit_eis

            # Print results
            banner(
                f"FITTED PARAMETERS  ({n_voigt} Voigt, sorted by tau)",
                C.GREEN)
            print(format_params(params, n_voigt))
            print(f"\n  {C.BOLD}RMSE = {metrics['rmse']:.6f} "
                  f"Ohm{C.RESET}")
            print(f"  {C.BOLD}R^2  = {metrics['r2']:.6f}{C.RESET}")
            if "bic" in metrics:
                print(f"  {C.BOLD}BIC  = {metrics['bic']:.2f}"
                      f"{C.RESET}")
            print(f"{C.GREEN}{'=' * 70}{C.RESET}")

            # Log (includes KK)
            log_fit_action(log_path, chosen_file, method_label,
                           params, n_voigt, metrics, kk_result)

            # Plot with components + KK overlay + save
            fit_count += 1
            plot_path = os.path.join(
                output_dir, f"{base_name}_fit_{fit_count}.png")
            info("Showing fit with overlapping arcs + KK overlay ...")
            plot_fit_with_components(
                freq_sel, zreal_sel, zimag_sel,
                Z_fit, Z_base, voigt_list,
                params, n_voigt, metrics, sign_conv,
                kk_result=kk_result, save_path=plot_path,
                data_file=chosen_file)

            # Save text results
            txt_path = os.path.join(
                output_dir,
                f"{base_name}_results_{fit_count}.txt")
            save_results_txt(txt_path, chosen_file, freq_sel,
                             params, n_voigt, metrics, kk_result)

            # Export CSV
            csv_path = os.path.join(
                output_dir,
                f"{base_name}_data_{fit_count}.csv")
            export_csv(csv_path, freq_sel, zreal_sel, zimag_sel,
                       Z_fit, params, n_voigt, metrics, kk_result)

        # If inner loop ended without [F], should not reach here
        # because [Q] does 'return'.  Safety guard:
        if not switch_file:
            break



