"""DRT controls matching the shared engine's supported settings."""
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox,
    QLabel, QPushButton, QVBoxLayout, QWidget,
)


class DRTControlsPanel(QWidget):
    run_drt = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        group = QGroupBox("DRT Analysis")
        form = QFormLayout(group)
        self.lambda_combo = QComboBox()
        self.lambda_combo.addItems(["mGCV", "GCV", "LC", "custom"])
        form.addRow("λ method:", self.lambda_combo)
        self.lambda_spin = QDoubleSpinBox()
        self.lambda_spin.setDecimals(8)
        self.lambda_spin.setRange(0.00000001, 100.0)
        self.lambda_spin.setValue(0.001)
        form.addRow("Custom λ:", self.lambda_spin)
        self.lambda_spin.setEnabled(False)
        self.lambda_combo.currentTextChanged.connect(
            lambda text: self.lambda_spin.setEnabled(text == "custom"))
        self.order_combo = QComboBox()
        self.order_combo.addItems(["2", "1"])
        form.addRow("Derivative order:", self.order_combo)
        self.data_combo = QComboBox()
        self.data_combo.addItems(["combined", "real", "imaginary"])
        form.addRow("Data use:", self.data_combo)
        self.inductance_combo = QComboBox()
        self.inductance_combo.addItems(["without", "with"])
        form.addRow("Inductance:", self.inductance_combo)
        self.fwhm_spin = QDoubleSpinBox()
        self.fwhm_spin.setRange(0.1, 2.0)
        self.fwhm_spin.setSingleStep(0.1)
        self.fwhm_spin.setValue(0.5)
        form.addRow("FWHM coefficient:", self.fwhm_spin)
        self.nonnegative_check = QCheckBox("Nonnegative DRT")
        self.nonnegative_check.setChecked(True)
        form.addRow(self.nonnegative_check)
        note = QLabel("τ grid follows the selected frequency range.")
        note.setWordWrap(True)
        form.addRow(note)
        self.btn_run = QPushButton("Run DRT")
        self.btn_run.setProperty("cssClass", "primary")
        self.btn_run.clicked.connect(self._on_run_clicked)
        form.addRow(self.btn_run)
        layout.addWidget(group)

    def _on_run_clicked(self):
        self.run_drt.emit({
            "lambda_method": self.lambda_combo.currentText(),
            "lambda_reg": self.lambda_spin.value(),
            "reg_order": int(self.order_combo.currentText()),
            "data_use": self.data_combo.currentText(),
            "inductance_mode": self.inductance_combo.currentText(),
            "fwhm_coeff": self.fwhm_spin.value(),
            "non_negative": self.nonnegative_check.isChecked(),
        })

    def set_running(self, running):
        self.btn_run.setEnabled(not running)
        self.btn_run.setText("Running…" if running else "Run DRT")
