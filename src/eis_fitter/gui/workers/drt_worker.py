"""QThread worker for DRT analysis."""
from PyQt5.QtCore import QThread, pyqtSignal
from eis_fitter.gui.engine_bridge import get_engine, PrintCapture


class DRTWorker(QThread):
    progress = pyqtSignal(str)
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, freq, zreal, zimag, settings):
        super().__init__()
        self.freq = freq
        self.zreal = zreal
        self.zimag = zimag
        self.settings = settings

    def cleanup(self):
        """Release heavy data references so memory can be freed."""
        self.freq = None
        self.zreal = None
        self.zimag = None
        self.settings = None

    def run(self):
        try:
            engine = get_engine()
            with PrintCapture(lambda t: self.progress.emit(t)):
                drt_result = engine.compute_drt(
                    self.freq, self.zreal, self.zimag,
                    **self.settings,
                )
                drt_result['lambda_method'] = self.settings.get(
                    'lambda_method', 'mGCV')
            self.result_ready.emit(drt_result)
        except Exception as e:
            self.error_occurred.emit(str(e))
