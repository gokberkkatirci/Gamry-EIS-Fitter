"""
Data Preview tab — shows raw Nyquist and Bode plots side by side.
"""
import numpy as np
from PyQt5.QtWidgets import QWidget, QHBoxLayout
from eis_fitter.gui.plotting.canvas import EmbeddedCanvas
from eis_fitter.gui.theme import ThemeManager


class DataPreviewTab(QWidget):
    """Tab 1: raw data Nyquist + Bode preview."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.nyquist_canvas = EmbeddedCanvas(figsize=(7, 6), dpi=100)
        self.bode_canvas = EmbeddedCanvas(figsize=(7, 6), dpi=100)
        layout.addWidget(self.nyquist_canvas)
        layout.addWidget(self.bode_canvas)
        self._show_empty_state()

    def _show_empty_state(self):
        color = ThemeManager.instance().current_palette["text_secondary"]
        for canvas, title, detail in (
            (self.nyquist_canvas, "Nyquist plot", "Open a folder and select a .DTA file"),
            (self.bode_canvas, "Bode plot", "Inspect impedance across frequency"),
        ):
            canvas.figure.text(0.5, 0.55, title, ha="center", va="center",
                               fontsize=18, color=color)
            canvas.figure.text(0.5, 0.47, detail, ha="center", va="center",
                               fontsize=10, color=color, wrap=True)

    def update_plots(self, freq, zreal, zimag, sign_conv):
        """Redraw preview plots with new data."""
        from eis_fitter.gui.engine_bridge import get_engine
        engine = get_engine()
        p = ThemeManager.instance().current_palette

        y_nq = engine._nyquist_sign(zimag, sign_conv)

        # -- Nyquist --
        fig1 = self.nyquist_canvas.figure
        fig1.clear()
        fig1.set_facecolor(p['plot_bg'])
        ax1 = fig1.add_subplot(111)
        ax1.set_facecolor(p['plot_axes_bg'])

        ax1.plot(zreal, y_nq, 'o', ms=5, color='royalblue',
                 mfc='white', mec='royalblue')
        ax1.set_xlabel("Z' (\u03a9)", fontsize=11, color=p['plot_text'])
        ylabel = "-Z'' (\u03a9)" if sign_conv == "negative" else "Z'' (\u03a9)"
        ax1.set_ylabel(ylabel, fontsize=11, color=p['plot_text'])
        ax1.set_title("Nyquist Plot (Raw Data)", fontsize=12,
                       fontweight='bold', color=p['plot_text'])
        ax1.set_aspect('equal', adjustable='datalim')
        ax1.grid(True, alpha=0.3, ls='--', color=p['plot_grid'])
        ax1.tick_params(colors=p['plot_tick'])
        for spine in ax1.spines.values():
            spine.set_color(p['plot_tick'])
        engine.apply_eng_format(ax1, 'both')

        # Frequency annotations
        step = max(1, len(freq) // 12)
        for i in range(0, len(freq), step):
            ax1.annotate(
                engine.eng_str(freq[i], 'Hz'), (zreal[i], y_nq[i]),
                fontsize=5.5, color=p['text_muted'], alpha=0.65,
                textcoords='offset points', xytext=(4, 4))

        fig1.tight_layout()
        self.nyquist_canvas.canvas.draw_idle()

        # -- Bode --
        fig2 = self.bode_canvas.figure
        fig2.clear()
        fig2.set_facecolor(p['plot_bg'])
        ax2 = fig2.add_subplot(111)
        ax2.set_facecolor(p['plot_axes_bg'])

        Zmod = np.sqrt(zreal**2 + zimag**2)
        phase = np.degrees(np.arctan2(zimag, zreal))

        ax2.set_xlabel("Frequency (Hz)", fontsize=11, color=p['plot_text'])
        ax2.set_xscale('log')
        ax2.set_ylabel("|Z| (\u03a9)", fontsize=11, color='tab:blue')
        ax2.loglog(freq, Zmod, 'o', ms=4, color='tab:blue', alpha=0.7)
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        ax2.tick_params(axis='x', colors=p['plot_tick'])
        for spine in ax2.spines.values():
            spine.set_color(p['plot_tick'])

        ax2b = ax2.twinx()
        ax2b.set_ylabel("Phase (\u00b0)", fontsize=11, color='tab:red')
        ax2b.semilogx(freq, phase, 's', ms=3, color='tab:red', alpha=0.7)
        ax2b.tick_params(axis='y', labelcolor='tab:red')

        ax2.set_title("Bode Plot (Raw Data)", fontsize=12,
                       fontweight='bold', color=p['plot_text'])
        ax2.grid(True, alpha=0.3, ls='--', color=p['plot_grid'])
        engine.apply_eng_format(ax2, 'x')

        fig2.tight_layout()
        self.bode_canvas.canvas.draw_idle()

    def clear(self):
        self.nyquist_canvas.clear()
        self.bode_canvas.clear()
