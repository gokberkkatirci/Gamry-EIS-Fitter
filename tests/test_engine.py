import numpy as np
import pytest

from eis_fitter import engine


def test_parser_columns_and_order(dta_factory, spectrum):
    frequency, real, imag = spectrum
    rows = [[z, i, r, f] for i, (f, r, z) in enumerate(zip(frequency, real, imag))]
    path = dta_factory(("Zimag", "Pt", "Zreal", "Freq"), rows)
    for actual, expected in zip(engine.read_dta_file(path), spectrum):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("content, message", [
    ("not a table", "ZCURVE"),
    ("ZCURVE\tTABLE\n", "Incomplete"),
    ("ZCURVE\tTABLE\nFreq\tZreal\tOther\nunits\n1\t2\t3", "Required columns"),
    ("ZCURVE\tTABLE\nFreq\tZreal\tZimag\nunits\n1\tbad\t3", "Malformed"),
])
def test_parser_rejects_bad_tables(tmp_path, content, message):
    path = tmp_path / "invalid.DTA"
    path.write_text(content)
    with pytest.raises(ValueError, match=message):
        engine.read_dta_file(path)


@pytest.mark.parametrize("row, message", [
    ([0, 0, 2, -1], "positive"),
    ([0, 1, float("nan"), -1], "finite"),
    ([0, 1, 2, float("inf")], "finite"),
])
def test_parser_rejects_invalid_values(dta_factory, row, message):
    with pytest.raises(ValueError, match=message):
        engine.read_dta_file(dta_factory(rows=[row] * 5))


def test_parser_too_few_and_duplicate_headers(dta_factory):
    with pytest.raises(ValueError, match="At least 5"):
        engine.read_dta_file(dta_factory(rows=[[0, 1, 2, -1]]))
    with pytest.raises(ValueError, match="occur once"):
        engine.read_dta_file(dta_factory(("Freq", "Freq", "Zreal", "Zimag")))


def test_spectrum_and_filter_guards(spectrum):
    frequency, real, imag = spectrum
    with pytest.raises(ValueError, match="equal lengths"):
        engine.validate_spectrum(frequency[:-1], real, imag)
    with pytest.raises(ValueError, match="one-dimensional"):
        engine.validate_spectrum([frequency], real, imag)
    with pytest.raises(ValueError, match="At least 5"):
        engine.filter_capacitive_region(frequency, real, np.zeros_like(imag))


def test_impedance_against_rc_limits(spectrum):
    frequency, real, imag = spectrum
    params = np.array([0, 2, 8, 0.05 / 8, 1.0])
    actual = engine.Z_model(2 * np.pi * frequency, params, 1)
    np.testing.assert_allclose(actual, real + 1j * imag, rtol=1e-13)
    np.testing.assert_allclose(engine.Z_voigt(np.array([1e-12]), 8, 0.05 / 8, 1), 8,
                               atol=1e-10)
    assert abs(engine.Z_voigt(np.array([1e15]), 8, 0.05 / 8, 1)[0]) < 1e-10
    np.testing.assert_allclose(engine.Z_inductor(np.array([10.0]), 0.01), [0.1j])


def test_tau_conversion_and_sort(spectrum):
    frequency = spectrum[0]
    tau_params = np.array([1e-6, 2, 8, 0.05, 0.85])
    q_params = engine._convert_tau_to_Q(tau_params, 1)
    assert q_params[3] == pytest.approx(0.05 ** 0.85 / 8)
    np.testing.assert_allclose(engine.Z_model(2 * np.pi * frequency, q_params, 1),
                               engine.Z_model_tau(2 * np.pi * frequency, tau_params, 1))
    blocks = np.array([0, 2, 4, 0.4 / 4, 1, 8, 0.05 / 8, 1])
    assert engine.sort_voigt_params(blocks, 2)[2] == 8


def test_small_fit_recovers_spectrum(spectrum, monkeypatch):
    # Keep actual optimization but bound CPU usage for CI and spawned GUI workers.
    from scipy.optimize import differential_evolution

    def serial_de(*args, **kwargs):
        kwargs["workers"] = 1
        return differential_evolution(*args, **kwargs)

    monkeypatch.setattr(engine, "differential_evolution", serial_de)
    result = engine.fit_eis(*spectrum, 1, maxiter=180, popsize=10, n_restarts=1)
    params, fitted, base, branches, metrics = result
    assert metrics["rmse"] < 0.03
    assert params[1] == pytest.approx(2, abs=0.08)
    assert params[2] == pytest.approx(8, abs=0.15)
    np.testing.assert_allclose(fitted, base + sum(branches))


def test_kk_drt_and_exports(spectrum, tmp_path):
    frequency, real, imag = spectrum
    kk = engine.kramers_kronig_test(*spectrum)
    assert kk["passed"]
    assert np.isfinite(kk["Z_kk"]).all()
    drt = engine.compute_drt(*spectrum, lambda_method="custom", lambda_reg=0.001)
    assert np.isfinite(drt["Z_fit"]).all()
    assert drt["gamma"].min() >= -1e-8
    assert drt["rmse"] < 1.0
    params = np.array([0, 2, 8, 0.05 / 8, 1])
    fitted = real + 1j * imag
    metrics = engine._compute_fit_metrics(real, imag, fitted, 0, 5)
    engine.export_csv(tmp_path / "fit.csv", *spectrum, fitted, params, 1, metrics, kk)
    engine.save_results_txt(tmp_path / "fit.txt", "input.DTA", frequency,
                            params, 1, metrics, kk)
    engine.export_drt_csv(tmp_path / "drt.csv", *spectrum, drt)
    engine.save_drt_results_txt(tmp_path / "drt.txt", "input.DTA", frequency, drt)
    assert all((tmp_path / name).stat().st_size > 100
               for name in ["fit.csv", "fit.txt", "drt.csv", "drt.txt"])
