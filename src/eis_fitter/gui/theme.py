"""
Centralized theme system with dark/light palettes, QSS generation,
matplotlib integration, and runtime theme switching.
"""
import matplotlib as mpl
from PyQt5.QtCore import QObject, pyqtSignal, QSettings
from PyQt5.QtWidgets import QApplication


# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------

LIGHT_PALETTE = {
    "name":            "light",
    "bg":              "#f5f5f5",
    "surface":         "#ffffff",
    "surface_alt":     "#f0f1f3",
    "border":          "#d0d0d0",
    "text":            "#2c3e50",
    "text_secondary":  "#5a6a7a",
    "text_muted":      "#8899a6",
    "primary":         "#27ae60",
    "primary_hover":   "#2ecc71",
    "secondary":       "#2980b9",
    "secondary_hover": "#3498db",
    "danger":          "#e74c3c",
    "danger_hover":    "#c0392b",
    "warning":         "#f39c12",
    "accent":          "#8e44ad",
    "success":         "#27ae60",
    "input_bg":        "#ffffff",
    "input_border":    "#c0c0c0",
    "scrollbar_bg":    "#e8e8e8",
    "scrollbar_handle":"#b0b0b0",
    "tab_active_bg":   "#ffffff",
    "tab_inactive_bg": "#e4e6e8",
    "header_bg":       "#e0e2e5",
    "header_text":     "#2c3e50",
    "disabled_bg":     "#bdc3c7",
    "disabled_text":   "#7f8c8d",
    "highlight":       "#3498db",
    "highlight_text":  "#ffffff",
    # matplotlib-specific
    "plot_bg":         "#ffffff",
    "plot_axes_bg":    "#ffffff",
    "plot_grid":       "#cccccc",
    "plot_text":       "#2c3e50",
    "plot_tick":       "#555555",
    "table_header_bg": "#2c3e50",
    "table_header_fg": "#ffffff",
    "table_row_even":  "#ffffff",
    "table_row_odd":   "#f2f3f4",
    "table_border":    "#cccccc",
}

DARK_PALETTE = {
    "name":            "dark",
    "bg":              "#1e1e2e",
    "surface":         "#282840",
    "surface_alt":     "#313150",
    "border":          "#44446a",
    "text":            "#e0e0ec",
    "text_secondary":  "#a0a0bc",
    "text_muted":      "#6c6c8a",
    "primary":         "#2ecc71",
    "primary_hover":   "#55efc4",
    "secondary":       "#5dade2",
    "secondary_hover": "#85c1e9",
    "danger":          "#e74c3c",
    "danger_hover":    "#ff6b6b",
    "warning":         "#f1c40f",
    "accent":          "#a29bfe",
    "success":         "#2ecc71",
    "input_bg":        "#2a2a44",
    "input_border":    "#50507a",
    "scrollbar_bg":    "#28283e",
    "scrollbar_handle":"#50507a",
    "tab_active_bg":   "#313150",
    "tab_inactive_bg": "#232338",
    "header_bg":       "#2a2a44",
    "header_text":     "#e0e0ec",
    "disabled_bg":     "#3a3a54",
    "disabled_text":   "#6c6c8a",
    "highlight":       "#5dade2",
    "highlight_text":  "#ffffff",
    # matplotlib-specific
    "plot_bg":         "#1e1e2e",
    "plot_axes_bg":    "#282840",
    "plot_grid":       "#44446a",
    "plot_text":       "#e0e0ec",
    "plot_tick":       "#a0a0bc",
    "table_header_bg": "#2a2a44",
    "table_header_fg": "#e0e0ec",
    "table_row_even":  "#282840",
    "table_row_odd":   "#313150",
    "table_border":    "#44446a",
}


# ---------------------------------------------------------------------------
# QSS stylesheet generator
# ---------------------------------------------------------------------------

def generate_stylesheet(p: dict) -> str:
    """Generate a complete QSS stylesheet from a palette dict."""
    return f"""
/* ===== Global ===== */
QWidget {{
    background-color: {p['bg']};
    color: {p['text']};
    font-size: 9pt;
}}

QMainWindow {{
    background-color: {p['bg']};
}}

/* ===== Menu Bar ===== */
QMenuBar {{
    background-color: {p['surface']};
    color: {p['text']};
    border-bottom: 1px solid {p['border']};
    padding: 2px;
}}
QMenuBar::item {{
    padding: 4px 10px;
    border-radius: 4px;
}}
QMenuBar::item:selected {{
    background-color: {p['highlight']};
    color: {p['highlight_text']};
}}
QMenu {{
    background-color: {p['surface']};
    color: {p['text']};
    border: 1px solid {p['border']};
    border-radius: 6px;
    padding: 4px;
}}
QMenu::item {{
    padding: 6px 24px;
    border-radius: 4px;
}}
QMenu::item:selected {{
    background-color: {p['highlight']};
    color: {p['highlight_text']};
}}
QMenu::separator {{
    height: 1px;
    background: {p['border']};
    margin: 4px 8px;
}}

/* ===== Toolbar ===== */
QToolBar {{
    background-color: {p['surface']};
    border-bottom: 1px solid {p['border']};
    padding: 3px;
    spacing: 4px;
}}
QToolBar QToolButton {{
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 4px;
    color: {p['text']};
}}
QToolBar QToolButton:hover {{
    background-color: {p['surface_alt']};
    border: 1px solid {p['border']};
}}
QToolBar QToolButton:pressed {{
    background-color: {p['header_bg']};
}}
QToolBar QToolButton:checked {{
    background-color: {p['surface_alt']};
    border: 1px solid {p['highlight']};
}}

/* ===== Status Bar ===== */
QStatusBar {{
    background-color: {p['surface']};
    color: {p['text_secondary']};
    border-top: 1px solid {p['border']};
}}
QStatusBar::item {{
    border: none;
}}

/* ===== Progress Bar ===== */
QProgressBar {{
    background-color: {p['input_bg']};
    border: 1px solid {p['border']};
    border-radius: 4px;
    text-align: center;
    color: {p['text']};
    height: 16px;
}}
QProgressBar::chunk {{
    background-color: {p['primary']};
    border-radius: 3px;
}}

/* ===== Tabs ===== */
QTabWidget::pane {{
    background-color: {p['surface']};
    border: 1px solid {p['border']};
    border-radius: 0 0 6px 6px;
    top: -1px;
}}
QTabBar::tab {{
    background-color: {p['tab_inactive_bg']};
    color: {p['text_secondary']};
    border: 1px solid {p['border']};
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 7px 14px;
    margin-right: 2px;
    font-size: 9pt;
}}
QTabBar::tab:selected {{
    background-color: {p['tab_active_bg']};
    color: {p['text']};
    font-weight: bold;
    border-bottom: 2px solid {p['primary']};
}}
QTabBar::tab:hover:!selected {{
    background-color: {p['surface_alt']};
    color: {p['text']};
}}

/* ===== Buttons ===== */
QPushButton {{
    background-color: {p['surface_alt']};
    color: {p['text']};
    border: 1px solid {p['border']};
    border-radius: 5px;
    padding: 5px 12px;
    font-weight: normal;
}}
QPushButton:hover {{
    background-color: {p['header_bg']};
    border-color: {p['text_muted']};
}}
QPushButton:pressed {{
    background-color: {p['border']};
}}
QPushButton:disabled {{
    background-color: {p['disabled_bg']};
    color: {p['disabled_text']};
    border-color: {p['disabled_bg']};
}}

/* Button variants via cssClass property */
QPushButton[cssClass="primary"] {{
    background-color: {p['secondary']};
    color: white;
    border: 1px solid {p['secondary']};
    font-weight: bold;
}}
QPushButton[cssClass="primary"]:hover {{
    background-color: {p['secondary_hover']};
    border-color: {p['secondary_hover']};
}}
QPushButton[cssClass="primary"]:disabled {{
    background-color: {p['disabled_bg']};
    color: {p['disabled_text']};
    border-color: {p['disabled_bg']};
}}

QPushButton[cssClass="success"] {{
    background-color: {p['primary']};
    color: white;
    border: 1px solid {p['primary']};
    font-weight: bold;
}}
QPushButton[cssClass="success"]:hover {{
    background-color: {p['primary_hover']};
    border-color: {p['primary_hover']};
}}
QPushButton[cssClass="success"]:disabled {{
    background-color: {p['disabled_bg']};
    color: {p['disabled_text']};
    border-color: {p['disabled_bg']};
}}

QPushButton[cssClass="danger"] {{
    background-color: {p['danger']};
    color: white;
    border: 1px solid {p['danger']};
    font-weight: bold;
}}
QPushButton[cssClass="danger"]:hover {{
    background-color: {p['danger_hover']};
    border-color: {p['danger_hover']};
}}
QPushButton[cssClass="danger"]:disabled {{
    background-color: {p['disabled_bg']};
    color: {p['disabled_text']};
    border-color: {p['disabled_bg']};
}}

/* ===== Inputs ===== */
QLineEdit {{
    background-color: {p['input_bg']};
    color: {p['text']};
    border: 1px solid {p['input_border']};
    border-radius: 4px;
    padding: 4px 6px;
}}
QLineEdit:focus {{
    border-color: {p['highlight']};
}}
QLineEdit:read-only {{
    background-color: {p['surface_alt']};
    color: {p['text_secondary']};
}}

QComboBox {{
    background-color: {p['input_bg']};
    color: {p['text']};
    border: 1px solid {p['input_border']};
    border-radius: 4px;
    padding: 4px 8px;
    min-height: 20px;
}}
QComboBox:hover {{
    border-color: {p['highlight']};
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox QAbstractItemView {{
    background-color: {p['surface']};
    color: {p['text']};
    border: 1px solid {p['border']};
    selection-background-color: {p['highlight']};
    selection-color: {p['highlight_text']};
}}

QSpinBox, QDoubleSpinBox {{
    background-color: {p['input_bg']};
    color: {p['text']};
    border: 1px solid {p['input_border']};
    border-radius: 4px;
    padding: 3px 6px;
}}
QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {p['highlight']};
}}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    border: none;
    width: 16px;
}}

/* ===== Checkboxes & Radio ===== */
QCheckBox {{
    spacing: 6px;
    color: {p['text']};
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {p['input_border']};
    border-radius: 3px;
    background-color: {p['input_bg']};
}}
QCheckBox::indicator:checked {{
    background-color: {p['primary']};
    border-color: {p['primary']};
}}

QRadioButton {{
    spacing: 6px;
    color: {p['text']};
}}
QRadioButton::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {p['input_border']};
    border-radius: 8px;
    background-color: {p['input_bg']};
}}
QRadioButton::indicator:checked {{
    background-color: {p['primary']};
    border-color: {p['primary']};
}}

/* ===== Group Box ===== */
QGroupBox {{
    background-color: {p['surface']};
    border: 1px solid {p['border']};
    border-radius: 6px;
    margin-top: 8px;
    padding: 12px 8px 8px 8px;
    font-weight: bold;
    color: {p['text']};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 8px;
    color: {p['text']};
}}

/* ===== List Widget ===== */
QListWidget {{
    background-color: {p['input_bg']};
    color: {p['text']};
    border: 1px solid {p['input_border']};
    border-radius: 4px;
    outline: none;
}}
QListWidget::item {{
    padding: 4px 6px;
    border-radius: 3px;
}}
QListWidget::item:selected {{
    background-color: {p['highlight']};
    color: {p['highlight_text']};
}}
QListWidget::item:hover:!selected {{
    background-color: {p['surface_alt']};
}}

/* ===== Scroll Bars ===== */
QScrollBar:vertical {{
    background-color: {p['scrollbar_bg']};
    width: 10px;
    border-radius: 5px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background-color: {p['scrollbar_handle']};
    border-radius: 5px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background-color: {p['text_muted']};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background-color: {p['scrollbar_bg']};
    height: 10px;
    border-radius: 5px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background-color: {p['scrollbar_handle']};
    border-radius: 5px;
    min-width: 30px;
}}
QScrollBar::handle:horizontal:hover {{
    background-color: {p['text_muted']};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* ===== Scroll Area ===== */
QScrollArea {{
    background-color: transparent;
    border: none;
}}
QScrollArea > QWidget > QWidget {{
    background-color: transparent;
}}

/* ===== Splitter ===== */
QSplitter::handle {{
    background-color: {p['border']};
    width: 2px;
    margin: 2px;
}}
QSplitter::handle:hover {{
    background-color: {p['highlight']};
}}

/* ===== Dock Widget ===== */
QDockWidget {{
    color: {p['text']};
    titlebar-close-icon: none;
}}
QDockWidget::title {{
    background-color: {p['header_bg']};
    color: {p['text']};
    padding: 6px;
    border: 1px solid {p['border']};
    border-radius: 4px 4px 0 0;
    font-weight: bold;
}}

/* ===== Plain Text Edit (Log) ===== */
QPlainTextEdit {{
    background-color: {p['surface']};
    color: {p['text']};
    border: 1px solid {p['border']};
    border-radius: 4px;
    font-size: 9pt;
}}

/* ===== Labels ===== */
QLabel {{
    background-color: transparent;
    color: {p['text']};
}}
QLabel[cssClass="muted"] {{
    color: {p['text_muted']};
}}
QLabel[cssClass="secondary"] {{
    color: {p['text_secondary']};
}}
QLabel[cssClass="accent"] {{
    color: {p['accent']};
}}

/* ===== Collapsible Section Header ===== */
QFrame[cssClass="sectionHeader"] {{
    background-color: {p['header_bg']};
    border: 1px solid {p['border']};
    border-radius: 4px;
}}
QFrame[cssClass="sectionHeader"]:hover {{
    background-color: {p['surface_alt']};
}}

/* ===== Tooltips ===== */
QToolTip {{
    background-color: {p['surface']};
    color: {p['text']};
    border: 1px solid {p['border']};
    border-radius: 4px;
    padding: 4px;
}}

/* ===== Message Box ===== */
QMessageBox {{
    background-color: {p['surface']};
    color: {p['text']};
}}
"""


# ---------------------------------------------------------------------------
# Matplotlib params
# ---------------------------------------------------------------------------

def get_mpl_params(p: dict) -> dict:
    """Return matplotlib rcParams dict matching the given palette."""
    return {
        "figure.facecolor":   p["plot_bg"],
        "axes.facecolor":     p["plot_axes_bg"],
        "axes.edgecolor":     p["plot_tick"],
        "axes.labelcolor":    p["plot_text"],
        "text.color":         p["plot_text"],
        "xtick.color":        p["plot_tick"],
        "ytick.color":        p["plot_tick"],
        "grid.color":         p["plot_grid"],
        "grid.alpha":         0.4,
        "legend.facecolor":   p["plot_axes_bg"],
        "legend.edgecolor":   p["border"],
        "savefig.facecolor":  p["plot_bg"],
    }


# ---------------------------------------------------------------------------
# ThemeManager singleton
# ---------------------------------------------------------------------------

class ThemeManager(QObject):
    """Manages application theme (dark / light) with runtime switching."""

    theme_changed = pyqtSignal(dict)  # emits current palette

    _instance = None

    def __init__(self):
        super().__init__()
        self._palette = LIGHT_PALETTE
        self._settings = QSettings("EISFitter", "EISFitter-0.1")

    # -- singleton accessor ------------------------------------------------
    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # -- public API --------------------------------------------------------
    @property
    def current_palette(self) -> dict:
        return self._palette

    @property
    def is_dark(self) -> bool:
        return self._palette is DARK_PALETTE

    def restore_saved_theme(self):
        """Apply the theme that was active in the last session."""
        name = self._settings.value("theme", "light")
        self.set_theme(name, save=False)

    def set_theme(self, name: str, save: bool = True):
        """Switch theme by name ('light' or 'dark')."""
        self._palette = DARK_PALETTE if name == "dark" else LIGHT_PALETTE
        self._apply()
        if save:
            self._settings.setValue("theme", self._palette["name"])

    def toggle_theme(self):
        """Toggle between light and dark."""
        self.set_theme("light" if self.is_dark else "dark")

    # -- internals ---------------------------------------------------------
    def _apply(self):
        app = QApplication.instance()
        if app:
            app.setStyleSheet(generate_stylesheet(self._palette))
        mpl.rcParams.update(get_mpl_params(self._palette))
        self.theme_changed.emit(self._palette)
