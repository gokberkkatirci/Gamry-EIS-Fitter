"""Shared engine import and terminal-output capture for GUI workers."""
import io
import re
import sys

from eis_fitter import engine

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')


def strip_ansi(text):
    """Remove ANSI escape codes."""
    return _ANSI_RE.sub('', text)


def get_engine():
    """Return the same engine used by the terminal interface."""
    return engine


class PrintCapture:
    """Context manager that captures stdout and calls a callback with each line.

    Usage:
        def on_output(text):
            signal.emit(text)

        with PrintCapture(on_output):
            some_function_that_prints()
    """

    def __init__(self, callback):
        self.callback = callback
        self._old_stdout = None
        self._buffer = io.StringIO()

    def __enter__(self):
        self._old_stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, *args):
        sys.stdout = self._old_stdout
        # Flush remaining buffer
        remaining = self._buffer.getvalue()
        if remaining.strip():
            self.callback(strip_ansi(remaining))
        self._buffer = io.StringIO()

    def write(self, text):
        if not text:
            return
        self._buffer.write(text)
        # Emit complete lines
        content = self._buffer.getvalue()
        if '\n' in content:
            lines = content.split('\n')
            # Last element is incomplete line (or empty)
            self._buffer = io.StringIO()
            self._buffer.write(lines[-1])
            for line in lines[:-1]:
                stripped = strip_ansi(line)
                if stripped.strip():
                    self.callback(stripped)

    def flush(self):
        pass
