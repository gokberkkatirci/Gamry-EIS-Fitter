"""
Frequency Selection tab — interactive click-based frequency range selection.
Reimplements select_frequency_range() for embedded canvas.
"""
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QPushButton, QStyle)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QFont
from eis_fitter.gui.plotting.canvas import EmbeddedCanvas
from eis_fitter.gui.theme import ThemeManager


class FrequencySelectionTab(QWidget):
    """Tab 2: interactive frequency range selection on Nyquist plot."""

    selection_complete = pyqtSignal(object)  # emits boolean mask

    # States
    IDLE = 0
    WAIT_START = 1
    WAIT_END = 2
    SELECTED = 3

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = self.IDLE
        self._clicked_points = []
        self._freq = None
        self._zreal = None
        self._zimag = None
        self._y_nq = None
        self._sign_conv = "negative"
        self._cid = None
        self._ax = None
        self._setup_ui()
        ThemeManager.instance().theme_changed.connect(self._on_theme_changed)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Instruction label
        self.instruction_label = QLabel("Load a file first")
        self.instruction_label.setFont(QFont(self.font().family(), 11, QFont.Bold))
        self.instruction_label.setAlignment(Qt.AlignCenter)
        self._update_instruction()
        layout.addWidget(self.instruction_label)

        # Canvas
        self.canvas_widget = EmbeddedCanvas(figsize=(10, 8), dpi=100)
        layout.addWidget(self.canvas_widget, stretch=1)

        # Buttons
        btn_row = QHBoxLayout()
        style = self.style()

        self.btn_reset = QPushButton("Reset Selection")
        self.btn_reset.setIcon(style.standardIcon(QStyle.SP_DialogResetButton))
        self.btn_reset.clicked.connect(self._reset)
        self.btn_reset.setEnabled(False)
        btn_row.addWidget(self.btn_reset)

        self.btn_use_all = QPushButton("Use All Data")
        self.btn_use_all.setIcon(style.standardIcon(QStyle.SP_DialogYesButton))
        self.btn_use_all.clicked.connect(self._use_all)
        self.btn_use_all.setEnabled(False)
        btn_row.addWidget(self.btn_use_all)

        self.btn_accept = QPushButton("Accept Selection")
        self.btn_accept.setProperty("cssClass", "success")
        self.btn_accept.setIcon(style.standardIcon(QStyle.SP_DialogApplyButton))
        self.btn_accept.clicked.connect(self._accept)
        self.btn_accept.setEnabled(False)
        btn_row.addWidget(self.btn_accept)

        layout.addLayout(btn_row)

    def setup(self, freq, zreal, zimag, sign_conv):
        """Initialize with filtered data and start interaction."""
        self._freq = freq
        self._zreal = zreal
        self._zimag = zimag
        self._sign_conv = sign_conv
        self._clicked_points = []
        self._state = self.WAIT_START

        from eis_fitter.gui.engine_bridge import get_engine
        engine = get_engine()
        self._y_nq = engine._nyquist_sign(zimag, sign_conv)

        self._draw_base_plot()
        self._update_instruction()
        self.btn_reset.setEnabled(True)
        self.btn_use_all.setEnabled(True)
        self.btn_accept.setEnabled(False)

        # Connect mouse event
        if self._cid is not None:
            self.canvas_widget.canvas.mpl_disconnect(self._cid)
        self._cid = self.canvas_widget.canvas.mpl_connect(
            'button_press_event', self._on_click)

    def _on_theme_changed(self, palette):
        """Refresh instruction label style when theme toggles."""
        self._update_instruction()
        # Only redraw the base plot if no selection is in progress
        # (to avoid erasing any already-clicked markers)
        if self._freq is not None and self._state in (self.IDLE, self.WAIT_START):
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

        ax.plot(self._zreal, self._y_nq, 'o', ms=6, color='royalblue',
                mfc='white', mec='royalblue')
        ax.set_xlabel("Z' (\u03a9)", fontsize=12, color=p['plot_text'])
        ylabel = "-Z'' (\u03a9)" if self._sign_conv == "negative" \
            else "Z'' (\u03a9)"
        ax.set_ylabel(ylabel, fontsize=12, color=p['plot_text'])
        ax.set_title("Click START and END points to select frequency range",
                      fontsize=12, fontweight='bold', color=p['plot_text'])
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True, alpha=0.3, ls='--', color=p['plot_grid'])
        ax.tick_params(colors=p['plot_tick'])
        for spine in ax.spines.values():
            spine.set_color(p['plot_tick'])
        engine.apply_eng_format(ax, 'both')

        # Frequency annotations
        step = max(1, len(self._freq) // 15)
        for i in range(0, len(self._freq), step):
            ax.annotate(
                engine.eng_str(self._freq[i], 'Hz'),
                (self._zreal[i], self._y_nq[i]),
                fontsize=6, color=p['text_muted'], alpha=0.7,
                textcoords='offset points', xytext=(5, 5))

        fig.tight_layout()
        self.canvas_widget.canvas.draw_idle()

    def _find_nearest(self, cx, cy):
        x_range = max(self._zreal.max() - self._zreal.min(), 1e-12)
        y_range = max(self._y_nq.max() - self._y_nq.min(), 1e-12)
        dist = ((self._zreal - cx) / x_range)**2 + \
               ((self._y_nq - cy) / y_range)**2
        return np.argmin(dist)

    def _on_click(self, event):
        if event.inaxes != self._ax or event.button != 1:
            return
        if self._state not in (self.WAIT_START, self.WAIT_END):
            return

        from eis_fitter.gui.engine_bridge import get_engine
        engine = get_engine()

        idx = self._find_nearest(event.xdata, event.ydata)
        self._clicked_points.append(idx)

        ax = self._ax
        if self._state == self.WAIT_START:
            ax.plot(self._zreal[idx], self._y_nq[idx], 'X', ms=15,
                    color='red', markeredgecolor='black', markeredgewidth=1.5,
                    label=f"START: {engine.eng_str(self._freq[idx], 'Hz')}",
                    zorder=10)
            ax.legend(fontsize=8, loc='best')
            self._state = self.WAIT_END
        elif self._state == self.WAIT_END:
            ax.plot(self._zreal[idx], self._y_nq[idx], 'X', ms=15,
                    color='green', markeredgecolor='black',
                    markeredgewidth=1.5,
                    label=f"END: {engine.eng_str(self._freq[idx], 'Hz')}",
                    zorder=10)
            # Highlight selected region
            idx1, idx2 = sorted(self._clicked_points)
            ax.plot(self._zreal[idx1:idx2+1], self._y_nq[idx1:idx2+1],
                    'o', ms=8, mfc='none', mec='lime', mew=2,
                    label='Selected region', zorder=6)
            ax.legend(fontsize=8, loc='best')
            self._state = self.SELECTED
            self.btn_accept.setEnabled(True)

        self._update_instruction()
        self.canvas_widget.canvas.draw_idle()

    def _update_instruction(self):
        p = ThemeManager.instance().current_palette
        msgs = {
            self.IDLE: "Load a file first",
            self.WAIT_START: "Click on the START point of your frequency range",
            self.WAIT_END: "Now click on the END point",
            self.SELECTED: "Region selected! Click 'Accept Selection' or 'Reset'",
        }
        colors = {
            self.IDLE: p['disabled_bg'],
            self.WAIT_START: p['danger'],
            self.WAIT_END: p['warning'],
            self.SELECTED: p['primary'],
        }
        self.instruction_label.setText(msgs.get(self._state, ""))
        c = colors.get(self._state, p['disabled_bg'])
        self.instruction_label.setStyleSheet(
            f"background-color: {c}20; color: {c}; padding: 8px; "
            f"border-radius: 4px; border: 2px solid {c};")

    def _reset(self):
        self._clicked_points = []
        self._state = self.WAIT_START
        self.btn_accept.setEnabled(False)
        self._draw_base_plot()
        self._update_instruction()

    def _use_all(self):
        if self._freq is None:
            return
        mask = np.ones(len(self._freq), dtype=bool)
        self.selection_complete.emit(mask)

    def _accept(self):
        if len(self._clicked_points) < 2 or self._freq is None:
            return
        idx1, idx2 = sorted(self._clicked_points)
        mask = np.zeros(len(self._freq), dtype=bool)
        mask[idx1:idx2+1] = True
        self.selection_complete.emit(mask)
