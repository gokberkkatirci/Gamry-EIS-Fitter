"""QThread worker for Kramers-Kronig validation."""
from PyQt5.QtCore import QThread, pyqtSignal
from eis_fitter.gui.engine_bridge import get_engine, PrintCapture


class KKWorker(QThread):
    progress = pyqtSignal(str)
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, freq, zreal, zimag):
        super().__init__()
        self.freq = freq
        self.zreal = zreal
        self.zimag = zimag

    def cleanup(self):
        """Release heavy data references so memory can be freed."""
        self.freq = None
        self.zreal = None
        self.zimag = None

    def run(self):
        try:
            engine = get_engine()
            with PrintCapture(lambda t: self.progress.emit(t)):
                kk_result = engine.kramers_kronig_test(
                    self.freq, self.zreal, self.zimag)
            self.result_ready.emit(kk_result)
        except Exception as e:
            self.error_occurred.emit(str(e))
