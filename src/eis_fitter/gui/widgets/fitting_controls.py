"""
ECM Fitting controls panel for the left sidebar.
"""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QComboBox, QSpinBox, QPushButton, QGroupBox,
                              QStyle)
from PyQt5.QtCore import pyqtSignal


class FittingControlsPanel(QWidget):
    """Left sidebar panel for ECM fitting options."""

    run_fit = pyqtSignal(dict)   # emits fitting parameters dict
    stop_fit = pyqtSignal()
    switch_to_peaks = pyqtSignal(str)  # 'nyquist' or 'bode'

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        grp = QGroupBox("ECM Fitting")
        grp_layout = QVBoxLayout(grp)
        grp_layout.setSpacing(6)

        # Method
        lbl_method = QLabel("Method:")
        lbl_method.setProperty("cssClass", "secondary")
        grp_layout.addWidget(lbl_method)

        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "Fixed N Voigt",
            "BIC Auto-Scan",
            "R\u00b2 Auto-Scan",
            "Manual (Nyquist Peaks)",
            "Bode Phase Peaks",
        ])
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        grp_layout.addWidget(self.method_combo)

        # N Voigt
        n_row = QHBoxLayout()
        n_row.addWidget(QLabel("N Voigt:"))
        self.n_voigt_spin = QSpinBox()
        self.n_voigt_spin.setRange(1, 10)
        self.n_voigt_spin.setValue(2)
        n_row.addWidget(self.n_voigt_spin)
        grp_layout.addLayout(n_row)

        # Advanced settings
        lbl_adv = QLabel("Advanced:")
        lbl_adv.setProperty("cssClass", "muted")
        grp_layout.addWidget(lbl_adv)

        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Restarts:"))
        self.restarts_spin = QSpinBox()
        self.restarts_spin.setRange(1, 20)
        self.restarts_spin.setValue(5)
        r1.addWidget(self.restarts_spin)
        grp_layout.addLayout(r1)

        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Max iter:"))
        self.maxiter_spin = QSpinBox()
        self.maxiter_spin.setRange(50, 2000)
        self.maxiter_spin.setValue(400)
        self.maxiter_spin.setSingleStep(50)
        r2.addWidget(self.maxiter_spin)
        grp_layout.addLayout(r2)

        r3 = QHBoxLayout()
        r3.addWidget(QLabel("Pop size:"))
        self.popsize_spin = QSpinBox()
        self.popsize_spin.setRange(10, 100)
        self.popsize_spin.setValue(25)
        r3.addWidget(self.popsize_spin)
        grp_layout.addLayout(r3)

        # Buttons
        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("Run Fit")
        self.btn_run.setProperty("cssClass", "success")
        self.btn_run.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.btn_run.clicked.connect(self._on_run_clicked)
        btn_row.addWidget(self.btn_run)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setProperty("cssClass", "danger")
        self.btn_stop.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_fit.emit)
        btn_row.addWidget(self.btn_stop)
        grp_layout.addLayout(btn_row)

        layout.addWidget(grp)

    def _on_method_changed(self, index):
        # Enable N Voigt only for fixed method
        self.n_voigt_spin.setEnabled(index == 0)

    def _on_run_clicked(self):
        method_map = {
            0: 'fixed', 1: 'bic', 2: 'r2', 3: 'manual_nyquist', 4: 'manual_bode'
        }
        method = method_map.get(self.method_combo.currentIndex(), 'fixed')

        if method in ('manual_nyquist', 'manual_bode'):
            # Signal to switch to peak selection tab
            mode = 'nyquist' if method == 'manual_nyquist' else 'bode'
            self.switch_to_peaks.emit(mode)
            return

        self.run_fit.emit({
            'method': method,
            'n_voigt': self.n_voigt_spin.value(),
            'maxiter': self.maxiter_spin.value(),
            'popsize': self.popsize_spin.value(),
            'n_restarts': self.restarts_spin.value(),
        })

    def set_running(self, running):
        """Toggle between running/idle state."""
        self.btn_run.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        self.method_combo.setEnabled(not running)
        self.n_voigt_spin.setEnabled(
            not running and self.method_combo.currentIndex() == 0)
