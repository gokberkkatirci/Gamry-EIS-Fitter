"""
DRT Results tab — displays the 6-panel DRT figure.
"""
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from eis_fitter.gui.plotting.canvas import EmbeddedCanvas


class DRTResultsTab(QWidget):
    """Tab 4: DRT analysis results (6-panel figure)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        self.canvas_widget = EmbeddedCanvas(figsize=(19, 12), dpi=100, scrollable=True)
        layout.addWidget(self.canvas_widget)

    def update_plot(self, freq, zreal, zimag, drt_result, sign_conv,
                    data_file=None):
        """Render the DRT results plot onto the embedded canvas."""
        from eis_fitter.gui.plotting.drt_plot import render_drt_results
        fig = self.canvas_widget.figure
        fig.clear()
        render_drt_results(fig, freq, zreal, zimag, drt_result, sign_conv,
                           data_file=data_file)
        self.canvas_widget.canvas.draw_idle()

    def clear(self):
        self.canvas_widget.clear()
