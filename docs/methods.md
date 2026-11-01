# Methods and limitations

This guide describes the implementation in version 0.1.0. The settings below are software defaults and search limits, not experimentally established operating limits for a particular material or cell.

## Circuit and conventions

With angular frequency ω = 2πf, the equivalent circuit is:

```text
Z(ω) = jωL + Rs + Σ Ri / [1 + (jωτi)^αi]
τi = (Ri Qi)^(1/αi)       Qi = τi^αi / Ri
```

Frequency is in Hz, impedance and resistance in Ω, inductance in H, and tau in seconds. CPE exponent α is dimensionless. Q has units S·s^α and is a capacitance in farads only when α = 1. A branch has impedance equivalent to a resistor in parallel with `1 / [Q(jω)^α]`.

The engine fits signed imaginary values as provided. For conventional capacitive data, Zimag is negative and the Nyquist ordinate is −Zimag. A display/filter heuristic infers the capacitive sign from the mean imaginary value of the second half of the input rows, then retains points of that sign. It neither sorts the data nor converts positive-imaginary capacitive inputs to the circuit's negative-imaginary convention. Use conventional Gamry signed data for fitting. The heuristic may be inappropriate for predominantly inductive or unusual spectra.

ECM uses the retained, selected capacitive points. Consequently its series-inductance estimate may be poorly constrained. Combined/imaginary DRT uses raw points within the selected frequency interval; real-only DRT uses the selected filtered points. Component offsets in Nyquist plots are visualization choices; fit residuals refer to the total model.

## Fitting

The global optimizer is SciPy differential evolution, followed by L-BFGS-B polishing. Internal optimization uses tau rather than Q. Restarts combine soft, absent, and hard tau-ordering penalties; the default allocation is two soft, two free, and one hard restart. At least one of each is run, so requesting one or two still executes three. Seeds are `42 + 7 × restart_index`. Peak-guided fitting uses narrower estimated bounds and its own restart logic.

The objective sums squared real and imaginary residuals weighted by `1 / (|Zmeasured| + 1e-12)^2`, with the configured ordering penalty. Candidate selection compares objective values from those strategies. A parallel DE attempt falls back to serial execution if it raises an exception. Solver/platform differences can affect outcomes even with fixed seeds. Peaks and close time constants need not be uniquely identifiable.

For ordinary fixed-N fitting, the bounds are:

| Parameter | Implemented bound |
| --- | --- |
| L | 0 to 0.001 H |
| Rs | `max(0.3 × min(abs(Zreal)), 0)` to `max(2 × min(abs(Zreal)), 0.01)` Ω |
| Ri | 1e-8 to `3 × peak_to_peak(Zreal)` Ω; a 1 Ω span is substituted for a nonpositive span |
| τi | Three decades beyond `1/(2πfmax)` and `1/(2πfmin)` |
| αi | 0.7 to 1.0 |

Peak-guided estimates change resistance and tau bounds. Inspect the engine before relying on those bounds as physical prior information.

RMSE is `sqrt(sum(|Zmeasured − Zfit|²) / N)` in Ω. R² uses the combined real-and-imaginary unweighted sum of squares about their separate means; it is reported as zero if that total variation is zero. These metrics differ from the weighted optimization objective.

The BIC scan minimizes `N × ln(RSS/N + 1e-30) + k × ln(N)`, where N is the number of complex frequency points, k is the circuit parameter count, and RSS is the unweighted complex residual sum of squares. This is the implemented selection score; do not interchange it with definitions counting 2N real observations. Best-R² selection adds no complexity penalty.

## KK residual check

The check fits a linear model consisting of a series resistance, fixed-tau RC branches, and an inductance term using ordinary least squares. Tau is log-spaced over one decade beyond each measured-frequency limit. It scans from two RC branches up to the point count, retains the lowest mean relative complex residual, and stops after at least three branches when that residual is below 1%. A plateau check is also present.

Relative complex residual is `100 × |Zmeasured − ZKK| / max(|Zmeasured|, 1e-15)`. The pass flag means the mean is below 1%. This residual-based rule is not the full literature robustness procedure, and a pass does not establish linearity, stationarity, causality, or a unique physical model. Inspect residual structure and measurement conditions. Background: [1].

## DRT

The DRT discretization uses Gaussian radial basis functions on log10(tau), Tikhonov regularization, and optional nonnegative coefficients. The grid extends one decade beyond each reciprocal-frequency limit, with `max(2N, 80)` nodes. FWHM is the width coefficient times grid spacing. The regularizer is an unscaled first- or second-difference matrix. Settings and lambda values therefore depend on the discretization; they are not universal physical quantities.

The terminal defaults to mGCV selection; direct `compute_drt` calls default to custom λ = 0.001. Available selection methods are custom, GCV, modified GCV, and L-curve. Real, imaginary, or combined components can be used. An inductance term is optional and has no effect for real-only data. Imaginary-only analysis cannot identify the series resistance.

For nonnegative solves, CVXPY with OSQP is attempted first; the existing active-set clipping heuristic is used if that solve fails. With additional resistance/inductance terms, the engine estimates those terms, solves for nonnegative DRT coefficients, then refines the additional terms. This is not a single jointly constrained optimization. Singular or ill-conditioned systems may fail. The polarisation-resistance estimate integrates the fitted coefficients over log10(tau). Inspect reconstruction error and stability under regularization changes before interpreting it physically.

The returned `gamma` array contains fitted RBF coefficients, rather than the continuous distribution reconstructed by summing those basis functions. The `R_pol` field integrates that coefficient array. Its normalization has not been established as a quantitative polarisation resistance; do not use absolute DRT amplitudes or `R_pol` as validated physical measurements. This initial release preserves the existing calculation.

The methodology relates to DRT literature [2, 3]; this software does not implement all features of DRTtools, establish numerical equivalence to it, or provide Bayesian uncertainty intervals. DRT peaks alone do not identify a mechanism.

## Verification limits

Automated tests use generated circuit spectra, parser edge cases, equation limits, optimizer recovery, KK/DRT smoke cases, and GUI worker/plot/export integration. They do not validate an experimental setup, measurement protocol, chemistry, or parameter uncertainty. No uncertainty propagation or confidence-interval calculation is provided.

## References

[1] M. Schönleber, D. Klotz, E. Ivers-Tiffée, “A Method for Improving the Robustness of linear Kramers-Kronig Validity Tests,” *Electrochimica Acta*, 131, 20–27, 2014. DOI: [10.1016/j.electacta.2014.01.034](https://doi.org/10.1016/j.electacta.2014.01.034). Bibliographic record checked against the [institutional record](https://cris.technion.ac.il/en/publications/a-method-for-improving-the-robustness-of-linear-kramers-kronig-va/).

[2] T. H. Wan, M. Saccoccio, C. Chen, F. Ciucci, “Influence of the discretization methods on the distribution of relaxation times deconvolution: implementing radial basis functions with DRTtools,” *Electrochimica Acta*, 184, 483–499, 2015. DOI: [10.1016/j.electacta.2015.09.097](https://doi.org/10.1016/j.electacta.2015.09.097). Checked against the authors’ [DRTtools repository](https://github.com/ciuccislab/DRTtools).

[3] Wang et al., “DRTtools: Freely Accessible Distribution of Relaxation Times Analysis for Electrochemical Impedance Spectroscopy,” *ACS Electrochemistry*, 1(12), 2680–2689, 2025. DOI: [10.1021/acselectrochem.5c00334](https://pubs.acs.org/doi/10.1021/acselectrochem.5c00334).
