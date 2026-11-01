"""
Reusable embedded matplotlib canvas widget with navigation toolbar.
"""
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QScrollArea
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg, NavigationToolbar2QT)
from matplotlib.figure import Figure

from eis_fitter.gui.theme import ThemeManager


class EmbeddedCanvas(QWidget):
    """Wrapper widget containing a matplotlib Figure, Canvas, and Toolbar."""

    def __init__(self, parent=None, figsize=(10, 8), dpi=100, scrollable=False):
        super().__init__(parent)
        self._export_size = figsize
        p = ThemeManager.instance().current_palette
        self.figure = Figure(figsize=figsize, dpi=dpi)
        self.figure.set_facecolor(p['plot_bg'])
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.toolbar)
        if scrollable:
            self.canvas.setMinimumSize(1600, 1100)
            scroll = QScrollArea()
            scroll.setWidget(self.canvas)
            scroll.setWidgetResizable(True)
            layout.addWidget(scroll)
        else:
            layout.addWidget(self.canvas)

        # Listen for theme changes
        ThemeManager.instance().theme_changed.connect(self._on_theme_changed)

    def _on_theme_changed(self, palette):
        """Update figure background when theme changes."""
        self.figure.set_facecolor(palette['plot_bg'])
        self.canvas.draw_idle()

    def clear(self):
        """Clear the figure and redraw, releasing cached renderer state."""
        self.figure.clear()
        # Force the renderer cache to be freed
        if hasattr(self.canvas, 'renderer') and self.canvas.renderer:
            self.canvas.renderer.clear()
        self.canvas.draw_idle()

    def redraw(self):
        """Trigger a canvas redraw."""
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def save_figure(self, filepath, dpi=200):
        """Save the current figure to a file."""
        original_size = self.figure.get_size_inches().copy()
        try:
            self.figure.set_size_inches(self._export_size, forward=False)
            self.figure.savefig(filepath, dpi=dpi, bbox_inches='tight')
        finally:
            self.figure.set_size_inches(original_size, forward=False)
            self.canvas.draw_idle()
        # Flush any temporary renderer used by savefig
        import gc
        gc.collect()
