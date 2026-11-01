"""
QApplication setup and launch for the EIS GUI.
"""
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from eis_fitter.gui.theme import ThemeManager


def run():
    """Create and show the main application window."""
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("EIS Fitter")
    app.setOrganizationName("EIS")
    app.setStyle("Fusion")

    # Initialize theme system (replaces hardcoded font/stylesheet)
    theme = ThemeManager.instance()
    theme.restore_saved_theme()

    from eis_fitter.gui.main_window import MainWindow
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
