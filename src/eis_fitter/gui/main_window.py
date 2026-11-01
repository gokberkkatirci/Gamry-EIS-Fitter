"""
Main window — orchestrates all panels, tabs, workers, and state.
"""
from eis_fitter import __version__

import gc
import os
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                              QTabWidget, QSplitter, QStatusBar, QProgressBar,
                              QMessageBox, QMenuBar, QAction, QScrollArea,
                              QFileDialog, QToolBar, QStyle, QLabel)
from PyQt5.QtCore import Qt, QSettings, QTimer

from eis_fitter.gui.state import AppState
from eis_fitter.gui.engine_bridge import get_engine
from eis_fitter.gui.theme import ThemeManager

from eis_fitter.gui.widgets.file_browser import FileBrowserPanel
from eis_fitter.gui.widgets.fitting_controls import FittingControlsPanel
from eis_fitter.gui.widgets.drt_controls import DRTControlsPanel
from eis_fitter.gui.widgets.export_controls import ExportControlsPanel
from eis_fitter.gui.widgets.data_preview import DataPreviewTab
from eis_fitter.gui.widgets.freq_selector import FrequencySelectionTab
from eis_fitter.gui.widgets.peak_selector import PeakSelectionTab
from eis_fitter.gui.widgets.ecm_results import ECMResultsTab
from eis_fitter.gui.widgets.drt_results import DRTResultsTab
from eis_fitter.gui.widgets.log_viewer import LogViewer
from eis_fitter.gui.widgets.collapsible_section import CollapsibleSection

from eis_fitter.gui.workers.kk_worker import KKWorker
from eis_fitter.gui.workers.fit_worker import FitWorker
from eis_fitter.gui.workers.drt_worker import DRTWorker


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.state = AppState()
        self._kk_worker = None
        self._fit_worker = None
        self._drt_worker = None
        self._settings = QSettings("EISFitter", "EISFitter-0.1")
        self._pending_file = None  # debounce file selection

        self.setWindowTitle(f"EIS Fitter {__version__}")
        self.setMinimumSize(1400, 900)

        self._setup_menubar()
        self._setup_toolbar()
        self._setup_ui()
        self._setup_statusbar()
        self._connect_signals()

        # Restore geometry after all widgets (including docks) are created
        self._restore_geometry()

        # Restore last working directory
        last_dir = self._settings.value("last_working_dir", ".")
        if os.path.isdir(last_dir):
            self.state.working_dir = last_dir
            self.file_browser.work_dir_edit.setText(last_dir)
            save_d = self._settings.value(
                "last_save_dir",
                os.path.join(last_dir, "EIS_Fit_Results"))
            self.state.save_dir = save_d
            self.file_browser.save_dir_edit.setText(save_d)
            self.file_browser.refresh_file_list()

    # ----------------------------------------------------------------
    # UI Setup
    # ----------------------------------------------------------------

    def _setup_menubar(self):
        menubar = self.menuBar()
        style = self.style()

        # File menu
        file_menu = menubar.addMenu("File")
        act_open = QAction("Open Directory...", self)
        act_open.setShortcut("Ctrl+O")
        act_open.setIcon(style.standardIcon(QStyle.SP_DirOpenIcon))
        act_open.triggered.connect(
            lambda: self.file_browser._browse_working_dir())
        file_menu.addAction(act_open)

        act_save_dir = QAction("Set Save Directory...", self)
        act_save_dir.setIcon(style.standardIcon(QStyle.SP_DirIcon))
        act_save_dir.triggered.connect(
            lambda: self.file_browser._browse_save_dir())
        file_menu.addAction(act_save_dir)

        file_menu.addSeparator()
        act_quit = QAction("Quit", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        # Analysis menu
        analysis_menu = menubar.addMenu("Analysis")
        act_kk = QAction("Run Kramers-Kronig", self)
        act_kk.setIcon(style.standardIcon(QStyle.SP_BrowserReload))
        act_kk.triggered.connect(self._run_kk)
        analysis_menu.addAction(act_kk)

        # View menu
        view_menu = menubar.addMenu("View")
        self._act_toggle_log = QAction("Toggle Log Console", self)
        self._act_toggle_log.setCheckable(True)
        self._act_toggle_log.setChecked(True)
        view_menu.addAction(self._act_toggle_log)

        view_menu.addSeparator()
        self._act_toggle_theme = QAction("Dark Theme", self)
        self._act_toggle_theme.setCheckable(True)
        self._act_toggle_theme.setChecked(ThemeManager.instance().is_dark)
        self._act_toggle_theme.triggered.connect(self._on_theme_toggled)
        view_menu.addAction(self._act_toggle_theme)

    def _setup_toolbar(self):
        style = self.style()
        toolbar = QToolBar("Main Toolbar")
        toolbar.setObjectName("main_toolbar")
        toolbar.setMovable(False)
        toolbar.setIconSize(toolbar.iconSize())  # default size

        act_open = toolbar.addAction(
            style.standardIcon(QStyle.SP_DirOpenIcon), "Open Directory")
        act_open.triggered.connect(
            lambda: self.file_browser._browse_working_dir())

        toolbar.addSeparator()

        self._tb_run = toolbar.addAction(
            style.standardIcon(QStyle.SP_MediaPlay), "Run Fit")
        self._tb_run.triggered.connect(
            lambda: self.fitting_controls._on_run_clicked())

        self._tb_stop = toolbar.addAction(
            style.standardIcon(QStyle.SP_MediaStop), "Stop")
        self._tb_stop.triggered.connect(self._on_stop_fit)

        toolbar.addSeparator()

        act_export = toolbar.addAction(
            style.standardIcon(QStyle.SP_DialogSaveButton), "Save All")
        act_export.triggered.connect(self._export_all)

        toolbar.addSeparator()

        # Theme toggle
        self._tb_theme = toolbar.addAction(
            style.standardIcon(QStyle.SP_DesktopIcon), "Toggle Theme")
        self._tb_theme.setCheckable(True)
        self._tb_theme.setChecked(ThemeManager.instance().is_dark)
        self._tb_theme.triggered.connect(self._on_theme_toggled)

        self.addToolBar(Qt.TopToolBarArea, toolbar)

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)

        style = self.style()

        # -- Left sidebar (scrollable with collapsible sections) --
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(4)

        # Files section
        self.file_browser = FileBrowserPanel(self.state)
        section_files = CollapsibleSection(
            "Files", icon=style.standardIcon(QStyle.SP_DirOpenIcon),
            expanded=True)
        section_files.set_content_widget(self.file_browser)
        sidebar_layout.addWidget(section_files, stretch=1)

        # ECM Fitting section
        self.fitting_controls = FittingControlsPanel()
        section_fit = CollapsibleSection(
            "ECM Fitting", icon=style.standardIcon(QStyle.SP_ComputerIcon),
            expanded=True)
        section_fit.set_content_widget(self.fitting_controls)
        sidebar_layout.addWidget(section_fit)

        # DRT Analysis section
        self.drt_controls = DRTControlsPanel()
        section_drt = CollapsibleSection(
            "DRT Analysis", icon=style.standardIcon(QStyle.SP_DialogApplyButton),
            expanded=False)
        section_drt.set_content_widget(self.drt_controls)
        sidebar_layout.addWidget(section_drt)

        # Export section
        self.export_controls = ExportControlsPanel()
        self.export_controls.set_enabled(False)
        section_export = CollapsibleSection(
            "Export", icon=style.standardIcon(QStyle.SP_DialogSaveButton),
            expanded=True)
        section_export.set_content_widget(self.export_controls)
        sidebar_layout.addWidget(section_export)

        sidebar_layout.addStretch()

        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidget(sidebar)
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setMinimumWidth(380)

        # -- Right: Tab Widget --
        self.tabs = QTabWidget()

        self.data_preview_tab = DataPreviewTab()
        self.tabs.addTab(self.data_preview_tab, "Data Preview")
        self.tabs.setTabIcon(0, style.standardIcon(
            QStyle.SP_FileDialogContentsView))

        self.freq_selector_tab = FrequencySelectionTab()
        self.tabs.addTab(self.freq_selector_tab, "Frequency Selection")
        self.tabs.setTabIcon(1, style.standardIcon(QStyle.SP_CommandLink))

        self.ecm_results_tab = ECMResultsTab()
        self.tabs.addTab(self.ecm_results_tab, "ECM Fit Results")
        self.tabs.setTabIcon(2, style.standardIcon(QStyle.SP_ComputerIcon))

        self.drt_results_tab = DRTResultsTab()
        self.tabs.addTab(self.drt_results_tab, "DRT Results")
        self.tabs.setTabIcon(3, style.standardIcon(
            QStyle.SP_DialogApplyButton))

        self.peak_selector_tab = PeakSelectionTab()
        self.tabs.addTab(self.peak_selector_tab, "Peak Selection")
        self.tabs.setTabIcon(4, style.standardIcon(QStyle.SP_ArrowUp))

        # -- Splitter for sidebar + tabs --
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(sidebar_scroll)
        splitter.addWidget(self.tabs)
        splitter.setSizes([400, 1100])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)

        # -- Log Viewer dock --
        self.log_viewer = LogViewer(self)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_viewer)
        self.resizeDocks([self.log_viewer], [160], Qt.Vertical)
        self._act_toggle_log.toggled.connect(self.log_viewer.setVisible)

    def _setup_statusbar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

        # Theme indicator label
        self._theme_label = QLabel()
        self._theme_label.setProperty("cssClass", "muted")
        self._update_theme_label()
        self.status_bar.addPermanentWidget(self._theme_label)

        self.status_bar.showMessage("Ready")

    # ----------------------------------------------------------------
    # Theme
    # ----------------------------------------------------------------

    def _on_theme_toggled(self, checked=None):
        ThemeManager.instance().toggle_theme()
        is_dark = ThemeManager.instance().is_dark
        self._act_toggle_theme.setChecked(is_dark)
        self._tb_theme.setChecked(is_dark)
        self._update_theme_label()

    def _update_theme_label(self):
        name = ThemeManager.instance().current_palette['name'].title()
        self._theme_label.setText(f"  {name} Theme  ")

    # ----------------------------------------------------------------
    # Signal Connections
    # ----------------------------------------------------------------

    def _connect_signals(self):
        # File browser
        self.file_browser.file_selected.connect(self._on_file_selected)
        self.file_browser.working_dir_changed.connect(
            self._on_working_dir_changed)
        self.file_browser.save_dir_changed.connect(
            self._on_save_dir_changed)

        # Frequency selection
        self.freq_selector_tab.selection_complete.connect(
            self._on_freq_selected)

        # Fitting
        self.fitting_controls.run_fit.connect(self._on_run_fit)
        self.fitting_controls.stop_fit.connect(self._on_stop_fit)
        self.fitting_controls.switch_to_peaks.connect(
            self._on_switch_to_peaks)

        # DRT
        self.drt_controls.run_drt.connect(self._on_run_drt)

        # Peak selection
        self.peak_selector_tab.peaks_accepted.connect(
            self._on_peaks_accepted)

        # Export
        self.export_controls.export_csv.connect(self._export_csv)
        self.export_controls.export_txt.connect(self._export_txt)
        self.export_controls.export_png.connect(self._export_png)
        self.export_controls.export_all.connect(self._export_all)

        # Auto-setup peak selector when tab is activated
        self.tabs.currentChanged.connect(self._on_tab_changed)

    # ----------------------------------------------------------------
    # Worker lifecycle helpers
    # ----------------------------------------------------------------

    def _cleanup_worker(self, attr_name):
        """Safely disconnect, wait for, and dispose of a worker thread.

        This prevents old QThread objects (and their large numpy data)
        from accumulating in memory after each fit/KK/DRT run.
        """
        worker = getattr(self, attr_name, None)
        if worker is None:
            return
        # Disconnect all signals so old results don't fire into handlers
        try:
            worker.progress.disconnect()
        except (TypeError, RuntimeError):
            pass
        try:
            worker.result_ready.disconnect()
        except (TypeError, RuntimeError):
            pass
        try:
            worker.error_occurred.disconnect()
        except (TypeError, RuntimeError):
            pass
        # Wait for the thread to finish (with timeout to avoid deadlock)
        if worker.isRunning():
            worker.quit()
            worker.wait(3000)  # 3 s timeout
            if worker.isRunning():
                worker.terminate()
                worker.wait(1000)
        # Release the worker's heavy data arrays
        worker.cleanup()
        # Schedule Qt-side deletion
        worker.deleteLater()
        setattr(self, attr_name, None)

    # ----------------------------------------------------------------
    # File Loading
    # ----------------------------------------------------------------

    def _on_working_dir_changed(self, path):
        self._settings.setValue("last_working_dir", path)

    def _on_save_dir_changed(self, path):
        self._settings.setValue("last_save_dir", path)

    def _on_file_selected(self, filepath):
        # Stop any running workers before loading a new file
        self._cleanup_worker('_kk_worker')
        self._cleanup_worker('_fit_worker')
        self._cleanup_worker('_drt_worker')

        try:
            engine = get_engine()
            self.state.clear_all()
            self.state.chosen_file = filepath
            self.state.base_name = os.path.splitext(
                os.path.basename(filepath))[0]

            self.log_viewer.append_info(f"Loading: {filepath}")

            # Read data
            from eis_fitter.gui.engine_bridge import PrintCapture
            with PrintCapture(lambda t: self.log_viewer.append(t)):
                freq, zreal, zimag = engine.read_dta_file(filepath)

            self.state.freq_raw = freq
            self.state.zreal_raw = zreal
            self.state.zimag_raw = zimag

            # Filter inductive region
            with PrintCapture(lambda t: self.log_viewer.append(t)):
                freq_f, zreal_f, zimag_f, sign_conv = \
                    engine.filter_capacitive_region(freq, zreal, zimag)

            self.state.freq_filtered = freq_f
            self.state.zreal_filtered = zreal_f
            self.state.zimag_filtered = zimag_f
            self.state.sign_conv = sign_conv

            # Update UI
            self.file_browser.update_data_info(
                freq_f, zreal_f, zimag_f, sign_conv)
            self.data_preview_tab.update_plots(
                freq_f, zreal_f, zimag_f, sign_conv)
            self.freq_selector_tab.setup(
                freq_f, zreal_f, zimag_f, sign_conv)

            self.setWindowTitle(
                f"EIS Fitter - {os.path.basename(filepath)}")
            self.tabs.setCurrentWidget(self.data_preview_tab)
            self.status_bar.showMessage(
                f"Loaded: {os.path.basename(filepath)} "
                f"({len(freq_f)} points)")
            self.export_controls.set_enabled(False)

        except Exception as e:
            QMessageBox.critical(self, "Error Loading File", str(e))
            self.log_viewer.append(f"ERROR: {e}")

    # ----------------------------------------------------------------
    # Frequency Selection
    # ----------------------------------------------------------------

    def _on_freq_selected(self, mask):
        s = self.state
        engine = get_engine()
        try:
            engine.validate_spectrum(
                s.freq_filtered[mask], s.zreal_filtered[mask],
                s.zimag_filtered[mask])
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Frequency Selection", str(exc))
            return
        s.freq_sel = s.freq_filtered[mask]
        s.zreal_sel = s.zreal_filtered[mask]
        s.zimag_sel = s.zimag_filtered[mask]
        s.sel_mask = mask

        engine = get_engine()
        n = len(s.freq_sel)
        self.log_viewer.append_info(
            f"Selected {n} points: "
            f"{engine.eng_str(s.freq_sel.min(), 'Hz')} - "
            f"{engine.eng_str(s.freq_sel.max(), 'Hz')}")
        self.status_bar.showMessage(f"Frequency range selected: {n} points")

        # Auto-run KK
        self._run_kk()

    def _run_kk(self):
        if not self.state.has_selected_data():
            QMessageBox.warning(self, "No Data",
                                "Select a frequency range first.")
            return
        self.log_viewer.append_info("Running Kramers-Kronig validation...")
        self._show_progress(True, 0, 0)

        self._cleanup_worker('_kk_worker')
        self._kk_worker = KKWorker(
            self.state.freq_sel, self.state.zreal_sel, self.state.zimag_sel)
        self._kk_worker.progress.connect(self.log_viewer.append)
        self._kk_worker.result_ready.connect(self._on_kk_complete)
        self._kk_worker.error_occurred.connect(self._on_kk_error)
        self._kk_worker.start()

    def _on_kk_complete(self, result):
        self.state.kk_result = result
        self._show_progress(False)
        status = "PASS" if result['passed'] else "FAIL"
        self.log_viewer.append(
            f"KK Validation: {status} "
            f"(M={result['n_rc']}, res={result['mean_res_pct']:.4f}%)")
        self.status_bar.showMessage(f"KK Validation: {status}")

        # Clean up worker to free memory
        self._cleanup_worker('_kk_worker')

        # Log
        self._log_kk()

    def _on_kk_error(self, msg):
        self._show_progress(False)
        self._cleanup_worker('_kk_worker')
        self.log_viewer.append(f"KK ERROR: {msg}")
        QMessageBox.warning(self, "KK Error", msg)

    def _log_kk(self):
        try:
            engine = get_engine()
            os.makedirs(self.state.save_dir, exist_ok=True)
            log_path = os.path.join(self.state.save_dir, "fit_log.txt")
            engine.log_kk_standalone(
                log_path, self.state.chosen_file,
                self.state.kk_result, self.state.freq_sel)
        except Exception as e:
            self.log_viewer.append(f"Log error: {e}")

    # ----------------------------------------------------------------
    # ECM Fitting
    # ----------------------------------------------------------------

    def _on_run_fit(self, params):
        if not self.state.has_selected_data():
            QMessageBox.warning(self, "No Data",
                                "Select a frequency range first.")
            return

        self.fitting_controls.set_running(True)
        self._show_progress(True, 0, 0)
        self.log_viewer.append_info(
            f"Starting ECM fit: method={params['method']}")

        self._cleanup_worker('_fit_worker')
        self._fit_worker = FitWorker(
            self.state.freq_sel, self.state.zreal_sel, self.state.zimag_sel,
            method=params['method'],
            n_voigt=params.get('n_voigt', 2),
            maxiter=params.get('maxiter', 400),
            popsize=params.get('popsize', 25),
            n_restarts=params.get('n_restarts', 5))
        self._fit_worker.progress.connect(self.log_viewer.append)
        self._fit_worker.result_ready.connect(self._on_fit_complete)
        self._fit_worker.error_occurred.connect(self._on_fit_error)
        self._fit_worker.start()

    def _on_stop_fit(self):
        if self._fit_worker and self._fit_worker.isRunning():
            self.log_viewer.append("Fit stopped by user.")
        self._cleanup_worker('_fit_worker')
        self.fitting_controls.set_running(False)
        self._show_progress(False)

    def _on_fit_complete(self, result):
        self.state.active_analysis = "ecm"
        self.fitting_controls.set_running(False)
        self._show_progress(False)

        s = self.state
        s.fit_params = result['params']
        s.fit_Z = result['Z_fit']
        s.fit_Z_base = result['Z_base']
        s.fit_voigt_list = result['voigt_list']
        s.fit_metrics = result['metrics']
        s.n_voigt = result['n_voigt']
        s.method_label = result['method_label']

        engine = get_engine()
        self.log_viewer.append(
            f"Fit complete: {s.method_label}")
        self.log_viewer.append(
            engine.format_params(s.fit_params, s.n_voigt))
        self.log_viewer.append(
            f"  RMSE = {s.fit_metrics['rmse']:.6f} Ohm, "
            f"R^2 = {s.fit_metrics['r2']:.6f}")

        # Update plot
        self.ecm_results_tab.update_plot(
            s.freq_sel, s.zreal_sel, s.zimag_sel,
            s.fit_Z, s.fit_Z_base, s.fit_voigt_list,
            s.fit_params, s.n_voigt, s.fit_metrics, s.sign_conv,
            kk_result=s.kk_result, data_file=s.chosen_file)
        self.tabs.setCurrentWidget(self.ecm_results_tab)

        # Clean up worker to free memory
        self._cleanup_worker('_fit_worker')

        # Log and auto-save
        self._log_fit()
        self._auto_save()
        self.export_controls.set_enabled(True)
        self.status_bar.showMessage(
            f"Fit complete: {s.n_voigt} Voigt, "
            f"R\u00b2={s.fit_metrics['r2']:.6f}")

    def _on_fit_error(self, msg):
        self.fitting_controls.set_running(False)
        self._show_progress(False)
        self._cleanup_worker('_fit_worker')
        self.log_viewer.append(f"FIT ERROR: {msg}")
        QMessageBox.warning(self, "Fit Error", msg)

    def _log_fit(self):
        try:
            engine = get_engine()
            os.makedirs(self.state.save_dir, exist_ok=True)
            log_path = os.path.join(self.state.save_dir, "fit_log.txt")
            engine.log_fit_action(
                log_path, self.state.chosen_file, self.state.method_label,
                self.state.fit_params, self.state.n_voigt,
                self.state.fit_metrics, self.state.kk_result)
        except Exception as e:
            self.log_viewer.append(f"Log error: {e}")

    # ----------------------------------------------------------------
    # Peak Selection -> Manual Fit
    # ----------------------------------------------------------------

    def _on_switch_to_peaks(self, mode):
        if not self.state.has_selected_data():
            QMessageBox.warning(self, "No Data",
                                "Select a frequency range first.")
            return
        self.peak_selector_tab.setup(
            self.state.freq_sel, self.state.zreal_sel,
            self.state.zimag_sel, self.state.sign_conv, mode)
        self.tabs.setCurrentWidget(self.peak_selector_tab)

    def _on_peaks_accepted(self, peaks):
        if not peaks:
            return
        self.fitting_controls.set_running(True)
        self._show_progress(True, 0, 0)
        n = len(peaks)
        self.log_viewer.append_info(
            f"Manual fit with {n} peak(s)...")

        self._cleanup_worker('_fit_worker')
        self._fit_worker = FitWorker(
            self.state.freq_sel, self.state.zreal_sel,
            self.state.zimag_sel,
            method='manual', peaks=peaks,
            maxiter=self.fitting_controls.maxiter_spin.value(),
            popsize=self.fitting_controls.popsize_spin.value(),
            n_restarts=self.fitting_controls.restarts_spin.value())
        self._fit_worker.progress.connect(self.log_viewer.append)
        self._fit_worker.result_ready.connect(self._on_fit_complete)
        self._fit_worker.error_occurred.connect(self._on_fit_error)
        self._fit_worker.start()

    # ----------------------------------------------------------------
    # DRT Analysis
    # ----------------------------------------------------------------

    def _on_run_drt(self, settings):
        if not self.state.has_selected_data():
            QMessageBox.warning(self, "No Data",
                                "Select a frequency range first.")
            return

        self.drt_controls.set_running(True)
        self._show_progress(True, 0, 0)
        self.log_viewer.append_info("Starting DRT analysis...")

        engine = get_engine()
        s = self.state

        # For combined/imaginary, use full spectrum within selected range
        if settings.get('data_use', 'combined') in ('combined', 'imaginary'):
            f_lo = s.freq_sel.min()
            f_hi = s.freq_sel.max()
            drt_mask = (s.freq_raw >= f_lo) & (s.freq_raw <= f_hi)
            freq_drt = s.freq_raw[drt_mask]
            zreal_drt = s.zreal_raw[drt_mask]
            zimag_drt = s.zimag_raw[drt_mask]
        else:
            freq_drt = s.freq_sel
            zreal_drt = s.zreal_sel
            zimag_drt = s.zimag_sel

        s.freq_drt = freq_drt
        s.zreal_drt = zreal_drt
        s.zimag_drt = zimag_drt

        self._cleanup_worker('_drt_worker')
        self._drt_worker = DRTWorker(freq_drt, zreal_drt, zimag_drt, settings)
        self._drt_worker.progress.connect(self.log_viewer.append)
        self._drt_worker.result_ready.connect(self._on_drt_complete)
        self._drt_worker.error_occurred.connect(self._on_drt_error)
        self._drt_worker.start()

    def _on_drt_complete(self, result):
        self.state.active_analysis = "drt"
        self.drt_controls.set_running(False)
        self._show_progress(False)

        self.state.drt_result = result
        self.log_viewer.append(
            f"DRT complete: R_inf={result['R_inf']:.6f}, "
            f"R_pol={result['R_pol']:.6f}, "
            f"lambda={result['lambda_used']:.4e}")

        # Update plot
        self.drt_results_tab.update_plot(
            self.state.freq_drt, self.state.zreal_drt,
            self.state.zimag_drt, result, self.state.sign_conv,
            data_file=self.state.chosen_file)
        self.tabs.setCurrentWidget(self.drt_results_tab)

        # Clean up worker to free memory
        self._cleanup_worker('_drt_worker')

        # Log and auto-save
        self._log_drt()
        self._auto_save()
        self.export_controls.set_enabled(True)
        self.status_bar.showMessage(
            f"DRT complete: R\u00b2={result['r2']:.6f}")

    def _on_drt_error(self, msg):
        self.drt_controls.set_running(False)
        self._show_progress(False)
        self._cleanup_worker('_drt_worker')
        self.log_viewer.append(f"DRT ERROR: {msg}")
        QMessageBox.warning(self, "DRT Error", msg)

    def _log_drt(self):
        try:
            engine = get_engine()
            os.makedirs(self.state.save_dir, exist_ok=True)
            log_path = os.path.join(self.state.save_dir, "fit_log.txt")
            engine.log_drt_action(
                log_path, self.state.chosen_file,
                self.state.freq_drt, self.state.drt_result)
        except Exception as e:
            self.log_viewer.append(f"Log error: {e}")

    # ----------------------------------------------------------------
    # Export
    # ----------------------------------------------------------------

    def _ensure_save_dir(self):
        os.makedirs(self.state.save_dir, exist_ok=True)

    def _export_csv(self):
        s = self.state
        try:
            self._ensure_save_dir()
            engine = get_engine()
            s.fit_count += 1

            if s.has_fit_results() and s.active_analysis != "drt":
                path = os.path.join(
                    s.save_dir,
                    f"{s.base_name}_data_{s.fit_count}.csv")
                engine.export_csv(
                    path, s.freq_sel, s.zreal_sel, s.zimag_sel,
                    s.fit_Z, s.fit_params, s.n_voigt,
                    s.fit_metrics, s.kk_result)
                self.log_viewer.append(f"CSV exported: {path}")
            elif s.has_drt_results():
                path = os.path.join(
                    s.save_dir,
                    f"{s.base_name}_DRT_data_{s.fit_count}.csv")
                engine.export_drt_csv(
                    path, s.freq_drt, s.zreal_drt, s.zimag_drt,
                    s.drt_result)
                self.log_viewer.append(f"DRT CSV exported: {path}")
            else:
                QMessageBox.warning(self, "No Results",
                                    "Run a fit or DRT analysis first.")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _export_txt(self):
        s = self.state
        try:
            self._ensure_save_dir()
            engine = get_engine()
            s.fit_count += 1

            if s.has_fit_results() and s.active_analysis != "drt":
                path = os.path.join(
                    s.save_dir,
                    f"{s.base_name}_results_{s.fit_count}.txt")
                engine.save_results_txt(
                    path, s.chosen_file, s.freq_sel,
                    s.fit_params, s.n_voigt, s.fit_metrics, s.kk_result)
                self.log_viewer.append(f"TXT saved: {path}")
            elif s.has_drt_results():
                path = os.path.join(
                    s.save_dir,
                    f"{s.base_name}_DRT_results_{s.fit_count}.txt")
                engine.save_drt_results_txt(
                    path, s.chosen_file, s.freq_drt, s.drt_result)
                self.log_viewer.append(f"DRT TXT saved: {path}")
            else:
                QMessageBox.warning(self, "No Results",
                                    "Run a fit or DRT analysis first.")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _export_png(self):
        s = self.state
        try:
            self._ensure_save_dir()
            s.fit_count += 1

            if s.has_fit_results() and s.active_analysis != "drt":
                path = os.path.join(
                    s.save_dir,
                    f"{s.base_name}_fit_{s.fit_count}.png")
                self.ecm_results_tab.canvas_widget.save_figure(path)
                self.log_viewer.append(f"Plot saved: {path}")
            elif s.has_drt_results():
                path = os.path.join(
                    s.save_dir,
                    f"{s.base_name}_DRT_{s.fit_count}.png")
                self.drt_results_tab.canvas_widget.save_figure(path)
                self.log_viewer.append(f"DRT plot saved: {path}")
            else:
                QMessageBox.warning(self, "No Results",
                                    "Run a fit or DRT analysis first.")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _export_all(self):
        """Export CSV, TXT, and PNG using a single fit_count increment."""
        s = self.state
        s.fit_count += 1
        count = s.fit_count
        try:
            self._ensure_save_dir()
            engine = get_engine()

            if s.has_fit_results() and s.active_analysis != "drt":
                csv_path = os.path.join(
                    s.save_dir, f"{s.base_name}_data_{count}.csv")
                engine.export_csv(
                    csv_path, s.freq_sel, s.zreal_sel, s.zimag_sel,
                    s.fit_Z, s.fit_params, s.n_voigt,
                    s.fit_metrics, s.kk_result)
                self.log_viewer.append(f"CSV exported: {csv_path}")

                txt_path = os.path.join(
                    s.save_dir, f"{s.base_name}_results_{count}.txt")
                engine.save_results_txt(
                    txt_path, s.chosen_file, s.freq_sel,
                    s.fit_params, s.n_voigt, s.fit_metrics, s.kk_result)
                self.log_viewer.append(f"TXT saved: {txt_path}")

                png_path = os.path.join(
                    s.save_dir, f"{s.base_name}_fit_{count}.png")
                self.ecm_results_tab.canvas_widget.save_figure(png_path)
                self.log_viewer.append(f"Plot saved: {png_path}")

            elif s.has_drt_results():
                csv_path = os.path.join(
                    s.save_dir, f"{s.base_name}_DRT_data_{count}.csv")
                engine.export_drt_csv(
                    csv_path, s.freq_drt, s.zreal_drt, s.zimag_drt,
                    s.drt_result)
                self.log_viewer.append(f"DRT CSV exported: {csv_path}")

                txt_path = os.path.join(
                    s.save_dir, f"{s.base_name}_DRT_results_{count}.txt")
                engine.save_drt_results_txt(
                    txt_path, s.chosen_file, s.freq_drt, s.drt_result)
                self.log_viewer.append(f"DRT TXT saved: {txt_path}")

                png_path = os.path.join(
                    s.save_dir, f"{s.base_name}_DRT_{count}.png")
                self.drt_results_tab.canvas_widget.save_figure(png_path)
                self.log_viewer.append(f"DRT plot saved: {png_path}")
            else:
                QMessageBox.warning(self, "No Results",
                                    "Run a fit or DRT analysis first.")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    # ----------------------------------------------------------------
    # Tab auto-setup
    # ----------------------------------------------------------------

    def _on_tab_changed(self, index):
        widget = self.tabs.widget(index)
        if widget is self.peak_selector_tab and self.state.has_selected_data():
            # Auto-load data into peak selector if it has no data yet
            if self.peak_selector_tab._freq is None:
                self.peak_selector_tab.setup(
                    self.state.freq_sel, self.state.zreal_sel,
                    self.state.zimag_sel, self.state.sign_conv,
                    self.peak_selector_tab._mode)

    # ----------------------------------------------------------------
    # Auto-save (saves every plot/result automatically)
    # ----------------------------------------------------------------

    def _auto_save(self):
        """Automatically save CSV, TXT, and PNG to save directory."""
        s = self.state
        try:
            self._ensure_save_dir()
            engine = get_engine()
            s.fit_count += 1

            if s.has_fit_results() and s.active_analysis != "drt":
                csv_path = os.path.join(
                    s.save_dir,
                    f"{s.base_name}_data_{s.fit_count}.csv")
                engine.export_csv(
                    csv_path, s.freq_sel, s.zreal_sel, s.zimag_sel,
                    s.fit_Z, s.fit_params, s.n_voigt,
                    s.fit_metrics, s.kk_result)

                txt_path = os.path.join(
                    s.save_dir,
                    f"{s.base_name}_results_{s.fit_count}.txt")
                engine.save_results_txt(
                    txt_path, s.chosen_file, s.freq_sel,
                    s.fit_params, s.n_voigt, s.fit_metrics, s.kk_result)

                png_path = os.path.join(
                    s.save_dir,
                    f"{s.base_name}_fit_{s.fit_count}.png")
                self.ecm_results_tab.canvas_widget.save_figure(png_path)

                self.log_viewer.append(
                    f"Auto-saved: {os.path.basename(csv_path)}, "
                    f"{os.path.basename(txt_path)}, "
                    f"{os.path.basename(png_path)}")

            elif s.has_drt_results():
                csv_path = os.path.join(
                    s.save_dir,
                    f"{s.base_name}_DRT_data_{s.fit_count}.csv")
                engine.export_drt_csv(
                    csv_path, s.freq_drt, s.zreal_drt, s.zimag_drt,
                    s.drt_result)

                txt_path = os.path.join(
                    s.save_dir,
                    f"{s.base_name}_DRT_results_{s.fit_count}.txt")
                engine.save_drt_results_txt(
                    txt_path, s.chosen_file, s.freq_drt, s.drt_result)

                png_path = os.path.join(
                    s.save_dir,
                    f"{s.base_name}_DRT_{s.fit_count}.png")
                self.drt_results_tab.canvas_widget.save_figure(png_path)

                self.log_viewer.append(
                    f"Auto-saved: {os.path.basename(csv_path)}, "
                    f"{os.path.basename(txt_path)}, "
                    f"{os.path.basename(png_path)}")

        except Exception as e:
            self.log_viewer.append(f"Auto-save error: {e}")

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------

    def _show_progress(self, show, value=0, maximum=0):
        self.progress_bar.setVisible(show)
        if show:
            if maximum == 0:
                self.progress_bar.setRange(0, 0)  # indeterminate
            else:
                self.progress_bar.setRange(0, maximum)
                self.progress_bar.setValue(value)
        else:
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(1)

    def _restore_geometry(self):
        geom = self._settings.value("geometry")
        if geom:
            self.restoreGeometry(geom)
        state = self._settings.value("windowState")
        if state:
            self.restoreState(state)
            # Sync the menu action checked state with the restored dock visibility
            self._act_toggle_log.setChecked(self.log_viewer.isVisible())

    def closeEvent(self, event):
        self._settings.setValue("geometry", self.saveGeometry())
        self._settings.setValue("windowState", self.saveState())
        # Clean up all workers before closing
        self._cleanup_worker('_kk_worker')
        self._cleanup_worker('_fit_worker')
        self._cleanup_worker('_drt_worker')
        gc.collect()
        event.accept()
