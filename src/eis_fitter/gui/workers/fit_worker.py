"""QThread worker for ECM fitting (all methods)."""
from PyQt5.QtCore import QThread, pyqtSignal
from eis_fitter.gui.engine_bridge import get_engine, PrintCapture


class FitWorker(QThread):
    progress = pyqtSignal(str)
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, freq, zreal, zimag, method, n_voigt=2,
                 peaks=None, maxiter=400, popsize=25, n_restarts=5):
        super().__init__()
        self.freq = freq
        self.zreal = zreal
        self.zimag = zimag
        self.method = method  # 'fixed', 'bic', 'r2', 'manual'
        self.n_voigt = n_voigt
        self.peaks = peaks
        self.maxiter = maxiter
        self.popsize = popsize
        self.n_restarts = n_restarts

    def cleanup(self):
        """Release heavy data references so memory can be freed."""
        self.freq = None
        self.zreal = None
        self.zimag = None
        self.peaks = None

    def run(self):
        try:
            engine = get_engine()
            result = {}

            with PrintCapture(lambda t: self.progress.emit(t)):
                if self.method == 'fixed':
                    params, Z_fit, Z_base, voigt_list, metrics = engine.fit_eis(
                        self.freq, self.zreal, self.zimag, self.n_voigt,
                        maxiter=self.maxiter, popsize=self.popsize,
                        n_restarts=self.n_restarts)
                    result = {
                        'params': params, 'Z_fit': Z_fit, 'Z_base': Z_base,
                        'voigt_list': voigt_list, 'metrics': metrics,
                        'n_voigt': self.n_voigt,
                        'method_label': f'fixed(N={self.n_voigt})',
                    }

                elif self.method == 'bic':
                    best_n, best_result, all_results = engine.auto_scan_voigt(
                        self.freq, self.zreal, self.zimag,
                        n_min=1, n_max=10)
                    result = {
                        'params': best_result['params'],
                        'Z_fit': best_result['Z_fit'],
                        'Z_base': best_result['Z_base'],
                        'voigt_list': best_result['voigt_list'],
                        'metrics': best_result['metrics'],
                        'n_voigt': best_n,
                        'method_label': f'bic(chose {best_n})',
                    }

                elif self.method == 'r2':
                    best_n, best_result, all_results = engine.auto_best_r2(
                        self.freq, self.zreal, self.zimag,
                        n_min=1, n_max=10)
                    result = {
                        'params': best_result['params'],
                        'Z_fit': best_result['Z_fit'],
                        'Z_base': best_result['Z_base'],
                        'voigt_list': best_result['voigt_list'],
                        'metrics': best_result['metrics'],
                        'n_voigt': best_n,
                        'method_label': f'auto(best-R^2, chose {best_n})',
                    }

                elif self.method == 'manual':
                    n_voigt = len(self.peaks)
                    params, Z_fit, Z_base, voigt_list, metrics = \
                        engine.manual_fit_eis(
                            self.freq, self.zreal, self.zimag,
                            self.peaks, n_voigt,
                            maxiter=self.maxiter, popsize=self.popsize,
                            n_restarts=self.n_restarts)
                    result = {
                        'params': params, 'Z_fit': Z_fit, 'Z_base': Z_base,
                        'voigt_list': voigt_list, 'metrics': metrics,
                        'n_voigt': n_voigt,
                        'method_label': f'manual-peaks(N={n_voigt})',
                    }

                # Sort Voigt elements by time constant
                n_v = result['n_voigt']
                result['params'] = engine.sort_voigt_params(
                    result['params'], n_v)
                import numpy as np
                omega = 2.0 * np.pi * self.freq
                Z_fit, Z_base, voigt_list = engine.Z_model_components(
                    omega, result['params'], n_v)
                result['Z_fit'] = Z_fit
                result['Z_base'] = Z_base
                result['voigt_list'] = voigt_list

            self.result_ready.emit(result)

        except Exception as e:
            self.error_occurred.emit(str(e))
