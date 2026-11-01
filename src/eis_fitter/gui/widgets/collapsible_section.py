"""
Collapsible accordion section for the sidebar.
"""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QSizePolicy, QFrame)
from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont, QIcon


class _HeaderBar(QFrame):
    """Clickable header that toggles the section."""

    def __init__(self, title: str, icon: QIcon = None, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setProperty("cssClass", "sectionHeader")
        self.setFixedHeight(32)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 0, 8, 0)
        lay.setSpacing(6)

        # Arrow indicator
        self._arrow = QLabel("\u25B6")  # right-pointing triangle
        self._arrow.setFixedWidth(14)
        arrow_font = QFont()
        arrow_font.setPointSize(8)
        self._arrow.setFont(arrow_font)
        lay.addWidget(self._arrow)

        # Optional icon
        if icon and not icon.isNull():
            icon_label = QLabel()
            icon_label.setPixmap(icon.pixmap(16, 16))
            icon_label.setFixedSize(18, 18)
            lay.addWidget(icon_label)

        # Title
        title_label = QLabel(title)
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(9)
        title_label.setFont(title_font)
        lay.addWidget(title_label)

        lay.addStretch()

    def set_expanded(self, expanded: bool):
        self._arrow.setText("\u25BC" if expanded else "\u25B6")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.parent().toggle()
        super().mousePressEvent(event)


class CollapsibleSection(QWidget):
    """
    A collapsible section with a clickable header and animated content area.

    Usage:
        section = CollapsibleSection("Title", icon=some_icon)
        section.set_content_widget(my_widget)
        layout.addWidget(section)
    """

    def __init__(self, title: str, icon: QIcon = None, expanded: bool = True,
                 parent=None):
        super().__init__(parent)
        self._expanded = expanded
        self._animation = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(0)

        # Header
        self._header = _HeaderBar(title, icon, parent=self)
        layout.addWidget(self._header)

        # Content container
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(0)
        self._content.setSizePolicy(QSizePolicy.Preferred,
                                     QSizePolicy.Preferred)
        layout.addWidget(self._content)

        # Initial state
        if not expanded:
            self._content.setMaximumHeight(0)
        self._header.set_expanded(expanded)

    # -- public API --------------------------------------------------------

    def set_content_widget(self, widget: QWidget):
        """Set the widget shown when the section is expanded."""
        self._content_layout.addWidget(widget)

    def toggle(self):
        """Toggle expanded / collapsed."""
        self.set_expanded(not self._expanded)

    def set_expanded(self, expanded: bool):
        self._expanded = expanded
        self._header.set_expanded(expanded)

        target_height = self._content.sizeHint().height() if expanded else 0

        anim = QPropertyAnimation(self._content, b"maximumHeight")
        anim.setDuration(150)
        anim.setEasingCurve(QEasingCurve.InOutCubic)
        anim.setStartValue(self._content.maximumHeight())
        anim.setEndValue(target_height)

        # Keep reference so it's not garbage-collected
        self._animation = anim
        anim.start()

        if expanded:
            # After animation, remove the max-height cap so the content
            # can resize freely.
            anim.finished.connect(
                lambda: self._content.setMaximumHeight(16777215))

    def is_expanded(self) -> bool:
        return self._expanded
