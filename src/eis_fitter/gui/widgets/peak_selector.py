"""
Peak Selection tab — interactive Nyquist/Bode peak clicking.
Reimplements select_semicircle_peaks() and select_bode_phase_peaks().
"""
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QPushButton, QRadioButton, QButtonGroup, QStyle)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QFont
from eis_fitter.gui.plotting.canvas import EmbeddedCanvas
from eis_fitter.gui.theme import ThemeManager
import matplotlib.pyplot as plt


class PeakSelectionTab(QWidget):
    """Tab 5: interactive peak selection on Nyquist or Bode plot."""

    peaks_accepted = pyqtSignal(list)  # emits list of peak dicts

    MAX_PEAKS = 10

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mode = 'nyquist'  # or 'bode'
        self._freq = None
        self._zreal = None
        self._zimag = None
        self._sign_conv = "negative"
        self._y_nq = None
        self._phase_deg = None
        self._clicked_indices = []
        self._peak_artists = []
        self._cid = None
        self._ax = None
        self._setup_ui()
        ThemeManager.instance().theme_changed.connect(self._on_theme_changed)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Mode toggle
        mode_row = QHBoxLayout()
        self.btn_group = QButtonGroup()
        self.rb_nyquist = QRadioButton("Nyquist Peaks")
        self.rb_nyquist.setChecked(True)
        self.rb_bode = QRadioButton("Bode Phase Peaks")
        self.btn_group.addButton(self.rb_nyquist, 0)
        self.btn_group.addButton(self.rb_bode, 1)
        self.btn_group.buttonClicked.connect(self._on_mode_changed)
        mode_row.addWidget(self.rb_nyquist)
        mode_row.addWidget(self.rb_bode)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        # Instruction + counter
        info_row = QHBoxLayout()
        self.instruction_label = QLabel(
            "Left-click to add peaks (1-10). Right-click to undo.")
        self.instruction_label.setFont(QFont(self.font().family(), 10, QFont.Bold))
        info_row.addWidget(self.instruction_label)

        self.counter_label = QLabel("Peaks: 0")
        self.counter_label.setFont(QFont(self.font().family(), 12, QFont.Bold))
        self.counter_label.setProperty("cssClass", "accent")
        info_row.addWidget(self.counter_label)
        layout.addLayout(info_row)

        # Canvas
        self.canvas_widget = EmbeddedCanvas(figsize=(10, 8), dpi=100)
        layout.addWidget(self.canvas_widget, stretch=1)

        # Buttons
        btn_row = QHBoxLayout()
        style = self.style()

        self.btn_undo = QPushButton("Undo Last")
        self.btn_undo.setIcon(style.standardIcon(QStyle.SP_ArrowLeft))
        self.btn_undo.clicked.connect(self._undo_last)
        btn_row.addWidget(self.btn_undo)

        self.btn_clear = QPushButton("Clear All")
        self.btn_clear.setIcon(style.standardIcon(QStyle.SP_DialogResetButton))
        self.btn_clear.clicked.connect(self._clear_peaks)
        btn_row.addWidget(self.btn_clear)

        self.btn_accept = QPushButton("Accept && Fit")
        self.btn_accept.setProperty("cssClass", "success")
        self.btn_accept.setIcon(style.standardIcon(QStyle.SP_DialogApplyButton))
        self.btn_accept.clicked.connect(self._on_accept)
        self.btn_accept.setEnabled(False)
        btn_row.addWidget(self.btn_accept)

        layout.addLayout(btn_row)

    def setup(self, freq, zreal, zimag, sign_conv, mode='nyquist'):
        """Initialize with data and start interaction."""
        self._freq = freq
        self._zreal = zreal
        self._zimag = zimag
        self._sign_conv = sign_conv
        self._mode = mode

        from eis_fitter.gui.engine_bridge import get_engine
        engine = get_engine()
        self._y_nq = engine._nyquist_sign(zimag, sign_conv)
        self._phase_deg = np.degrees(np.arctan2(zimag, zreal))

        if mode == 'nyquist':
            self.rb_nyquist.setChecked(True)
        else:
            self.rb_bode.setChecked(True)

        self._clear_peaks()
        self._draw_base_plot()

        # Connect mouse event
        if self._cid is not None:
            self.canvas_widget.canvas.mpl_disconnect(self._cid)
        self._cid = self.canvas_widget.canvas.mpl_connect(
            'button_press_event', self._on_click)

    def _on_mode_changed(self, btn):
        self._mode = 'nyquist' if self.btn_group.checkedId() == 0 else 'bode'
        if self._freq is not None:
            self._clear_peaks()
            self._draw_base_plot()

    def _draw_base_plot(self):
        from eis_fitter.gui.engine_bridge import get_engine
        engine = get_engine()
        p = ThemeManager.instance().current_palette

        fig = self.canvas_widget.figure
        fig.clear()
        fig.set_facecolor(p['plot_bg'])
        self._ax = fig.add_subplot(111)
        ax = self._ax
        ax.set_facecolor(p['plot_axes_bg'])

        if self._mode == 'nyquist':
            ax.plot(self._zreal, self._y_nq, 'o-', ms=6, color='royalblue',
                    mfc='white', mec='royalblue', lw=1)
            ax.set_xlabel("Z' (\u03a9)", fontsize=12, color=p['plot_text'])
            ylabel = "-Z'' (\u03a9)" if self._sign_conv == "negative" \
                else "Z'' (\u03a9)"
            ax.set_ylabel(ylabel, fontsize=12, color=p['plot_text'])
            ax.set_title("Click on the TOP of each semi-circle peak",
                          fontsize=12, fontweight='bold', color=p['plot_text'])
            ax.set_aspect('equal', adjustable='datalim')
            engine.apply_eng_format(ax, 'both')

            step = max(1, len(self._freq) // 15)
            for i in range(0, len(self._freq), step):
                ax.annotate(
                    engine.eng_str(self._freq[i], 'Hz'),
                    (self._zreal[i], self._y_nq[i]),
                    fontsize=6, color=p['text_muted'], alpha=0.7,
                    textcoords='offset points', xytext=(5, 5))
        else:
            ax.semilogx(self._freq, self._phase_deg, 'o-', ms=5,
                         color='darkred', mfc='white', mec='darkred', lw=1.2)
            ax.set_xlabel("Frequency (Hz)", fontsize=12, color=p['plot_text'])
            ax.set_ylabel("Phase Angle (\u00b0)", fontsize=12,
                          color=p['plot_text'])
            ax.set_title("Click on convex points (phase dips / extrema)",
                          fontsize=12, fontweight='bold', color=p['plot_text'])
            engine.apply_eng_format(ax, 'x')

        ax.grid(True, alpha=0.3, ls='--', color=p['plot_grid'])
        ax.tick_params(colors=p['plot_tick'])
        for spine in ax.spines.values():
            spine.set_color(p['plot_tick'])
        fig.tight_layout()
        self.canvas_widget.canvas.draw_idle()

    def _find_nearest(self, cx, cy):
        if self._mode == 'nyquist':
            x_data, y_data = self._zreal, self._y_nq
        else:
            x_data = np.log10(np.clip(self._freq, 1e-30, None))
            y_data = self._phase_deg
            cx = np.log10(max(cx, 1e-30))

        x_range = max(x_data.max() - x_data.min(), 1e-12)
        y_range = max(y_data.max() - y_data.min(), 1e-12)
        dist = ((x_data - cx) / x_range)**2 + ((y_data - cy) / y_range)**2
        return np.argmin(dist)

    def _on_click(self, event):
        if event.inaxes != self._ax:
            return

        from eis_fitter.gui.engine_bridge import get_engine
        engine = get_engine()

        # Right-click = undo
        if event.button == 3:
            self._undo_last()
            return

        if event.button != 1:
            return
        if len(self._clicked_indices) >= self.MAX_PEAKS:
            return

        idx = self._find_nearest(event.xdata, event.ydata)
        peak_num = len(self._clicked_indices) + 1
        colors = plt.cm.tab10(np.linspace(0, 0.9, self.MAX_PEAKS))
        color = colors[peak_num - 1]

        self._clicked_indices.append(idx)
        ax = self._ax

        if self._mode == 'nyquist':
            marker = ax.plot(
                self._zreal[idx], self._y_nq[idx], 'v', ms=14, color=color,
                markeredgecolor='black', markeredgewidth=1.5,
                label=f"Peak {peak_num}: "
                      f"{engine.eng_str(self._freq[idx], 'Hz')}",
                zorder=10)
            vline = ax.plot(
                [self._zreal[idx], self._zreal[idx]],
                [0, self._y_nq[idx]],
                '--', lw=1, color=color, alpha=0.5, zorder=5)
        else:
            marker = ax.plot(
                self._freq[idx], self._phase_deg[idx], 'v', ms=14,
                color=color, markeredgecolor='black', markeredgewidth=1.5,
                label=f"Peak {peak_num}: "
                      f"{engine.eng_str(self._freq[idx], 'Hz')} "
                      f"({self._phase_deg[idx]:.1f}\u00b0)",
                zorder=10)
            vline = ax.plot(
                [self._freq[idx], self._freq[idx]],
                [ax.get_ylim()[0], self._phase_deg[idx]],
                '--', lw=1, color=color, alpha=0.5, zorder=5)

        self._peak_artists.append((marker[0], vline))
        ax.legend(fontsize=8, loc='best')
        self._update_ui()
        self.canvas_widget.canvas.draw_idle()

    def _undo_last(self):
        if not self._clicked_indices:
            return
        self._clicked_indices.pop()
        if self._peak_artists:
            marker_art, vline_art = self._peak_artists.pop()
            marker_art.remove()
            if vline_art:
                vline_art[0].remove()
            if self._ax:
                self._ax.legend(fontsize=8, loc='best')
        self._update_ui()
        self.canvas_widget.canvas.draw_idle()

    def _clear_peaks(self):
        self._clicked_indices = []
        self._peak_artists = []
        self.btn_accept.setEnabled(False)
        self.counter_label.setText("Peaks: 0")
        if self._freq is not None:
            self._draw_base_plot()

    def _update_ui(self):
        p = ThemeManager.instance().current_palette
        n = len(self._clicked_indices)
        self.counter_label.setText(f"Peaks: {n}")
        self.btn_accept.setEnabled(n > 0)
        if n == 0:
            self.instruction_label.setText(
                "Left-click to add peaks (1-10). Right-click to undo.")
            self.instruction_label.setStyleSheet(
                f"color: {p['danger']}; padding: 4px;")
        elif n < self.MAX_PEAKS:
            self.instruction_label.setText(
                f"{n} peak(s) selected. Click more or 'Accept & Fit'.")
            self.instruction_label.setStyleSheet(
                f"color: {p['warning']}; padding: 4px;")
        else:
            self.instruction_label.setText(
                f"Maximum {self.MAX_PEAKS} peaks! Click 'Accept & Fit'.")
            self.instruction_label.setStyleSheet(
                f"color: {p['primary']}; padding: 4px;")

    def _on_theme_changed(self, palette):
        """Refresh instruction label style and plot when theme toggles."""
        self._update_ui()
        # Only redraw if no peaks placed yet (to avoid erasing markers)
        if self._freq is not None and not self._clicked_indices:
            self._draw_base_plot()

    def _on_accept(self):
        if not self._clicked_indices:
            return

        peaks = []
        for idx in self._clicked_indices:
            peaks.append({
                'idx': idx,
                'freq': self._freq[idx],
                'zreal': self._zreal[idx],
                'zimag_abs': abs(self._zimag[idx]),
            })
        # Sort by frequency descending
        peaks.sort(key=lambda p: p['freq'], reverse=True)
        self.peaks_accepted.emit(peaks)
