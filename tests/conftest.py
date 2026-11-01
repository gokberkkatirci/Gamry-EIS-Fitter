"""Generate all numerical inputs at runtime; use temporary files only."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest


@pytest.fixture
def spectrum():
    frequency = np.logspace(3, -1, 25)
    omega = 2 * np.pi * frequency
    # Independently expressed single parallel RC plus series resistance.
    impedance = 2.0 + 8.0 / (1 + 1j * omega * 0.05)
    return frequency, impedance.real, impedance.imag


@pytest.fixture
def dta_factory(tmp_path, spectrum):
    def write(headers=("Pt", "Freq", "Zreal", "Zimag"), rows=None):
        frequency, real, imag = spectrum
        if rows is None:
            rows = [[i, f, r, z] for i, (f, r, z) in
                    enumerate(zip(frequency, real, imag))]
        path = tmp_path / "input.DTA"
        body = ["EXPLAIN", "ZCURVE\tTABLE", "\t".join(headers), "#\tHz\tohm\tohm"]
        body.extend("\t".join(map(str, row)) for row in rows)
        path.write_text("\n".join(body) + "\n", encoding="latin-1")
        return path
    return write
