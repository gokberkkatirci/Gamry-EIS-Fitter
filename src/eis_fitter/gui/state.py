"""
Application state — single source of truth for all shared data.
"""
import numpy as np


class AppState:
    """Holds all shared application state."""

    def __init__(self):
        # Directories
        self.working_dir = "."
        self.save_dir = "EIS_Fit_Results"

        # File management
        self.dta_files = []
        self.chosen_file = ""
        self.base_name = ""
        self.fit_count = 0

        # Raw data (as read from file)
        self.freq_raw = None
        self.zreal_raw = None
        self.zimag_raw = None

        # Filtered data (inductive region removed)
        self.freq_filtered = None
        self.zreal_filtered = None
        self.zimag_filtered = None
        self.sign_conv = "negative"

        # Selected frequency range
        self.freq_sel = None
        self.zreal_sel = None
        self.zimag_sel = None
        self.sel_mask = None

        # KK results
        self.kk_result = None

        self.active_analysis = None

        # ECM fit results
        self.fit_params = None
        self.fit_Z = None
        self.fit_Z_base = None
        self.fit_voigt_list = None
        self.fit_metrics = None
        self.n_voigt = 0
        self.method_label = ""

        # DRT results
        self.drt_result = None
        # Store the freq/zreal/zimag used for DRT (may differ from sel)
        self.freq_drt = None
        self.zreal_drt = None
        self.zimag_drt = None

    def has_raw_data(self):
        return self.freq_raw is not None and len(self.freq_raw) > 0

    def has_filtered_data(self):
        return self.freq_filtered is not None and len(self.freq_filtered) > 0

    def has_selected_data(self):
        return self.freq_sel is not None and len(self.freq_sel) > 0

    def has_fit_results(self):
        return self.fit_params is not None

    def has_drt_results(self):
        return self.drt_result is not None

    def has_kk_results(self):
        return self.kk_result is not None

    def clear_results(self):
        """Clear fit/DRT/KK results (keep data)."""
        self.active_analysis = None
        self.kk_result = None
        self.fit_params = None
        self.fit_Z = None
        self.fit_Z_base = None
        self.fit_voigt_list = None
        self.fit_metrics = None
        self.n_voigt = 0
        self.method_label = ""
        self.drt_result = None
        self.freq_drt = None
        self.zreal_drt = None
        self.zimag_drt = None

    def clear_all(self):
        """Clear everything (new file selected)."""
        self.freq_raw = None
        self.zreal_raw = None
        self.zimag_raw = None
        self.freq_filtered = None
        self.zreal_filtered = None
        self.zimag_filtered = None
        self.sign_conv = "negative"
        self.freq_sel = None
        self.zreal_sel = None
        self.zimag_sel = None
        self.sel_mask = None
        self.clear_results()
