"""
ECM Fit Results tab — displays the 6-panel fit figure.
"""
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from eis_fitter.gui.plotting.canvas import EmbeddedCanvas


class ECMResultsTab(QWidget):
    """Tab 3: ECM fit results (6-panel figure)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        self.canvas_widget = EmbeddedCanvas(figsize=(19, 12), dpi=100, scrollable=True)
        layout.addWidget(self.canvas_widget)

    def update_plot(self, freq, zreal, zimag, Z_fit, Z_base, voigt_list,
                    params, n_voigt, metrics, sign_conv,
                    kk_result=None, data_file=None):
        """Render the ECM fit plot onto the embedded canvas."""
        from eis_fitter.gui.plotting.ecm_plot import render_ecm_fit
        fig = self.canvas_widget.figure
        fig.clear()
        render_ecm_fit(fig, freq, zreal, zimag, Z_fit, Z_base, voigt_list,
                       params, n_voigt, metrics, sign_conv,
                       kk_result=kk_result, data_file=data_file)
        self.canvas_widget.canvas.draw_idle()

    def clear(self):
        self.canvas_widget.clear()
