import time

import numpy as np
import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import QApplication, QMessageBox

from eis_fitter import engine
from eis_fitter.gui.engine_bridge import get_engine
from eis_fitter.gui.main_window import MainWindow
from eis_fitter.gui.workers.drt_worker import DRTWorker
from eis_fitter.gui.workers.fit_worker import FitWorker
from eis_fitter.gui.workers.kk_worker import KKWorker


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope,
                      str(tmp_path_factory.mktemp("settings")))
    instance = QApplication.instance() or QApplication([])
    yield instance


def finish_worker(app, worker):
    results, errors = [], []
    worker.result_ready.connect(results.append)
    worker.error_occurred.connect(errors.append)
    worker.start()
    deadline = time.monotonic() + 60
    while worker.isRunning() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    if worker.isRunning():
        worker.requestInterruption()
        worker.wait()
        pytest.fail("Worker exceeded 60 seconds")
    worker.wait()
    app.processEvents()
    assert not errors, errors
    assert len(results) == 1
    return results[0]


def test_workers_controls_plots_exports(app, tmp_path, monkeypatch, spectrum, dta_factory):
    monkeypatch.chdir(tmp_path)
    from scipy.optimize import differential_evolution

    def serial_de(*args, **kwargs):
        kwargs["workers"] = 1
        return differential_evolution(*args, **kwargs)

    monkeypatch.setattr(engine, "differential_evolution", serial_de)

    def unexpected_dialog(*args):
        pytest.fail(str(args))

    monkeypatch.setattr(QMessageBox, "critical", unexpected_dialog)
    monkeypatch.setattr(QMessageBox, "warning", unexpected_dialog)
    assert get_engine() is engine
    window = MainWindow()
    window.show()
    app.processEvents()
    try:
        window._on_file_selected(str(dta_factory()))
        assert window.state.has_raw_data()
        window.state.freq_sel, window.state.zreal_sel, window.state.zimag_sel = spectrum
        window.state.save_dir = str(tmp_path / "exports")
        frequency, real, imag = spectrum
        window.state.freq_drt = frequency
        window.state.zreal_drt = real
        window.state.zimag_drt = imag
        kk = finish_worker(app, KKWorker(*spectrum))
        window._on_kk_complete(kk)
        fit = finish_worker(app, FitWorker(*spectrum, "fixed", n_voigt=1,
                                          maxiter=180, popsize=10, n_restarts=1))
        window._on_fit_complete(fit)
        window._export_all()
        assert window.state.has_fit_results()
        emitted = []
        window.drt_controls.run_drt.connect(emitted.append)
        window.drt_controls.lambda_combo.setCurrentText("custom")
        window.drt_controls._on_run_clicked()
        drt = finish_worker(app, DRTWorker(*spectrum, emitted[0]))
        window._on_drt_complete(drt)
        assert window.state.has_drt_results()
        # Exports follow the most recently completed analysis, retaining both.
        assert window.state.has_fit_results()
        window._export_all()
        app.processEvents()
        output = list((tmp_path / "exports").iterdir())
        assert {path.suffix for path in output} >= {".csv", ".txt", ".png"}
        assert any("DRT" in path.name and path.suffix == ".png" for path in output)
        assert all(path.stat().st_size > 0 for path in output)
        for canvas in (window.ecm_results_tab.canvas_widget,
                       window.drt_results_tab.canvas_widget):
            canvas.canvas.draw()
        from eis_fitter.gui.theme import ThemeManager
        for theme in ("dark", "light"):
            ThemeManager.instance().set_theme(theme, save=False)
            window.ecm_results_tab.update_plot(
                frequency, real, imag, fit["Z_fit"], fit["Z_base"],
                fit["voigt_list"], fit["params"], 1, fit["metrics"], "negative", kk)
            window.drt_results_tab.update_plot(frequency, real, imag, drt, "negative")
            app.processEvents()
    finally:
        window.close()
        app.processEvents()
