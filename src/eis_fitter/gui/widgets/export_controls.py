"""
Export controls panel for the left sidebar.
"""
import os
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QGroupBox,
                              QMessageBox, QStyle)
from PyQt5.QtCore import pyqtSignal


class ExportControlsPanel(QWidget):
    """Left sidebar panel with export buttons."""

    export_csv = pyqtSignal()
    export_txt = pyqtSignal()
    export_png = pyqtSignal()
    export_all = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        grp = QGroupBox("Export")
        grp_layout = QVBoxLayout(grp)
        grp_layout.setSpacing(4)

        style = self.style()

        self.btn_csv = QPushButton("Export CSV")
        self.btn_csv.setIcon(style.standardIcon(QStyle.SP_FileIcon))
        self.btn_csv.clicked.connect(self.export_csv.emit)
        grp_layout.addWidget(self.btn_csv)

        self.btn_txt = QPushButton("Save Results TXT")
        self.btn_txt.setIcon(style.standardIcon(QStyle.SP_FileDialogDetailedView))
        self.btn_txt.clicked.connect(self.export_txt.emit)
        grp_layout.addWidget(self.btn_txt)

        self.btn_png = QPushButton("Save Plot PNG")
        self.btn_png.setIcon(style.standardIcon(QStyle.SP_DesktopIcon))
        self.btn_png.clicked.connect(self.export_png.emit)
        grp_layout.addWidget(self.btn_png)

        self.btn_all = QPushButton("Save All")
        self.btn_all.setProperty("cssClass", "primary")
        self.btn_all.setIcon(style.standardIcon(QStyle.SP_DialogSaveButton))
        self.btn_all.clicked.connect(self.export_all.emit)
        grp_layout.addWidget(self.btn_all)

        layout.addWidget(grp)

    def set_enabled(self, has_results):
        """Enable/disable buttons based on whether results exist."""
        self.btn_csv.setEnabled(has_results)
        self.btn_txt.setEnabled(has_results)
        self.btn_png.setEnabled(has_results)
        self.btn_all.setEnabled(has_results)
