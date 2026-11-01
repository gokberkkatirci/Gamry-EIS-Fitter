"""
File browser panel — working dir, save dir, file list, data info.
"""
import os
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QLineEdit, QPushButton, QListWidget,
                              QFileDialog, QGroupBox, QListWidgetItem, QStyle)
from PyQt5.QtCore import pyqtSignal, Qt


class FileBrowserPanel(QWidget):
    """Left sidebar panel for directory and file selection."""

    file_selected = pyqtSignal(str)  # emits full file path
    working_dir_changed = pyqtSignal(str)
    save_dir_changed = pyqtSignal(str)

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        style = self.style()

        # -- Working Directory --
        grp_dir = QGroupBox("Directories")
        dir_layout = QVBoxLayout(grp_dir)

        lbl_work = QLabel("Working Directory:")
        lbl_work.setProperty("cssClass", "secondary")
        dir_layout.addWidget(lbl_work)

        work_row = QHBoxLayout()
        self.work_dir_edit = QLineEdit(self.state.working_dir)
        self.work_dir_edit.setReadOnly(True)
        btn_work = QPushButton("Browse...")
        btn_work.setIcon(style.standardIcon(QStyle.SP_DirOpenIcon))
        btn_work.setMaximumWidth(100)
        btn_work.clicked.connect(self._browse_working_dir)
        work_row.addWidget(self.work_dir_edit)
        work_row.addWidget(btn_work)
        dir_layout.addLayout(work_row)

        lbl_save = QLabel("Save Directory:")
        lbl_save.setProperty("cssClass", "secondary")
        dir_layout.addWidget(lbl_save)

        save_row = QHBoxLayout()
        self.save_dir_edit = QLineEdit(self.state.save_dir)
        self.save_dir_edit.setReadOnly(True)
        btn_save = QPushButton("Browse...")
        btn_save.setIcon(style.standardIcon(QStyle.SP_DirOpenIcon))
        btn_save.setMaximumWidth(100)
        btn_save.clicked.connect(self._browse_save_dir)
        save_row.addWidget(self.save_dir_edit)
        save_row.addWidget(btn_save)
        dir_layout.addLayout(save_row)

        layout.addWidget(grp_dir)

        # -- File List --
        grp_files = QGroupBox("DTA Files")
        files_layout = QVBoxLayout(grp_files)

        self.file_list = QListWidget()
        self.file_list.itemDoubleClicked.connect(self._on_file_double_clicked)
        self.file_list.itemClicked.connect(self._on_file_clicked)
        files_layout.addWidget(self.file_list)

        self.file_count_label = QLabel("No files found")
        self.file_count_label.setProperty("cssClass", "muted")
        files_layout.addWidget(self.file_count_label)

        layout.addWidget(grp_files, stretch=1)

        # -- Data Info --
        grp_info = QGroupBox("Data Info")
        info_layout = QVBoxLayout(grp_info)
        self.info_label = QLabel("No file loaded")
        self.info_label.setWordWrap(True)
        self.info_label.setProperty("cssClass", "secondary")
        info_layout.addWidget(self.info_label)
        layout.addWidget(grp_info)

    def _browse_working_dir(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select Working Directory", self.state.working_dir)
        if d:
            self.state.working_dir = d
            self.work_dir_edit.setText(d)
            # Auto-set save dir to EIS_Fit_Results under working dir
            save_d = os.path.join(d, "EIS_Fit_Results")
            self.state.save_dir = save_d
            self.save_dir_edit.setText(save_d)
            self.working_dir_changed.emit(d)
            self.save_dir_changed.emit(save_d)
            self.refresh_file_list()

    def _browse_save_dir(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select Save Directory", self.state.save_dir)
        if d:
            self.state.save_dir = d
            self.save_dir_edit.setText(d)
            self.save_dir_changed.emit(d)

    def refresh_file_list(self):
        """Refresh the DTA file list from working directory."""
        from eis_fitter.gui.engine_bridge import get_engine
        engine = get_engine()
        self.state.dta_files = engine.find_all_dta_files(
            self.state.working_dir)
        # Sort by modification time, newest first
        self.state.dta_files.sort(key=lambda f: os.path.getmtime(f),
                                   reverse=True)
        self.file_list.clear()
        for fpath in self.state.dta_files:
            rel = os.path.relpath(fpath, self.state.working_dir)
            item = QListWidgetItem(rel)
            item.setData(Qt.UserRole, fpath)
            item.setToolTip(fpath)
            self.file_list.addItem(item)
        count = len(self.state.dta_files)
        self.file_count_label.setText(f"{count} file(s) found")

    def _on_file_clicked(self, item):
        """Load file on single mouse click (not keyboard navigation)."""
        if item is not None:
            fpath = item.data(Qt.UserRole)
            # Avoid re-loading the same file
            if fpath != self.state.chosen_file:
                self.file_selected.emit(fpath)

    def _on_file_double_clicked(self, item):
        """Load file on double-click (always, even if already loaded)."""
        fpath = item.data(Qt.UserRole)
        self.file_selected.emit(fpath)

    def update_data_info(self, freq, zreal, zimag, sign_conv):
        """Update the data info label after loading a file."""
        from eis_fitter.gui.engine_bridge import get_engine
        engine = get_engine()
        n = len(freq)
        f_lo = engine.eng_str(freq.min(), 'Hz')
        f_hi = engine.eng_str(freq.max(), 'Hz')
        self.info_label.setText(
            f"Points: {n}\n"
            f"Freq: {f_lo} - {f_hi}\n"
            f"Zreal: {zreal.min():.4f} - {zreal.max():.4f} \u03a9\n"
            f"Zimag: {zimag.min():.4f} - {zimag.max():.4f} \u03a9\n"
            f"Sign: {sign_conv}")
