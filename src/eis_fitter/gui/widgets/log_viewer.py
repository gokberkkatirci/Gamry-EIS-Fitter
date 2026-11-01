"""
Log viewer dock widget — captures and displays colored console output.
"""
from PyQt5.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
                              QPlainTextEdit, QPushButton, QFileDialog, QStyle)
from PyQt5.QtGui import QFont, QFontDatabase, QTextCharFormat, QColor, QTextCursor
from PyQt5.QtCore import Qt, pyqtSlot

from eis_fitter.gui.theme import ThemeManager


class LogViewer(QDockWidget):
    """Bottom dock widget with a colored log console."""

    def __init__(self, parent=None):
        super().__init__("Log Console", parent)
        self.setObjectName("log_console")
        self.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)

        self.text_edit = QPlainTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.text_edit.setMaximumBlockCount(5000)
        layout.addWidget(self.text_edit)

        btn_layout = QHBoxLayout()
        style = self.style()

        btn_clear = QPushButton("Clear")
        btn_clear.setIcon(style.standardIcon(QStyle.SP_DialogResetButton))
        btn_clear.clicked.connect(self.text_edit.clear)

        btn_save = QPushButton("Save Log")
        btn_save.setIcon(style.standardIcon(QStyle.SP_DialogSaveButton))
        btn_save.clicked.connect(self._save_log)

        btn_layout.addWidget(btn_clear)
        btn_layout.addWidget(btn_save)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.setWidget(container)

    def _get_color(self, key: str) -> str:
        """Get a color from the current theme palette."""
        p = ThemeManager.instance().current_palette
        mapping = {
            'error': p['danger'],
            'warning': p['warning'],
            'success': p['primary'],
            'info': p['secondary'],
            'muted': p['text_muted'],
            'default': p['text'],
        }
        return mapping.get(key, p['text'])

    @pyqtSlot(str)
    def append(self, text):
        """Append text to the log, auto-colored by content."""
        text_lower = text.lower().strip()

        fmt = QTextCharFormat()
        if any(kw in text_lower for kw in ['error', 'fail', 'exception']):
            fmt.setForeground(QColor(self._get_color('error')))
            fmt.setFontWeight(QFont.Bold)
        elif any(kw in text_lower for kw in ['warn', 'warning']):
            fmt.setForeground(QColor(self._get_color('warning')))
        elif any(kw in text_lower for kw in ['success', 'pass', '>>>',
                                              'best', 'saved', 'exported']):
            fmt.setForeground(QColor(self._get_color('success')))
            fmt.setFontWeight(QFont.Bold)
        elif any(kw in text_lower for kw in ['*best*']):
            fmt.setForeground(QColor(self._get_color('info')))
            fmt.setFontWeight(QFont.Bold)
        elif text.strip().startswith('restart') or 'n=' in text_lower:
            fmt.setForeground(QColor(self._get_color('muted')))
        else:
            fmt.setForeground(QColor(self._get_color('default')))

        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text + '\n', fmt)
        self.text_edit.setTextCursor(cursor)
        self.text_edit.ensureCursorVisible()

    @pyqtSlot(str)
    def append_info(self, text):
        """Append info-colored text."""
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(self._get_color('info')))
        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text + '\n', fmt)
        self.text_edit.setTextCursor(cursor)
        self.text_edit.ensureCursorVisible()

    def _save_log(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Log", "eis_log.txt", "Text Files (*.txt)")
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(self.text_edit.toPlainText())
