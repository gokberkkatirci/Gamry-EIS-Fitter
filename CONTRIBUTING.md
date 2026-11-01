# Contributing

Use Python 3.11 or 3.12 and a virtual environment. From the repository folder:

```bash
python -m pip install -e ".[gui,dev]"
python -m ruff check src tests tools
python -m pytest -q
python tools/check_release.py
```

For offscreen GUI tests on macOS or Linux:

```bash
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python -m pytest -q
```

In PowerShell, set `$env:QT_QPA_PLATFORM="offscreen"` and `$env:MPLBACKEND="Agg"` before running pytest. A passing offscreen test does not replace an interactive desktop check.

All numerical inputs must be generated in tests. Write temporary files using pytest's temporary-directory fixtures. Do not commit measurements, result files, local settings, or identifiable paths. Use reproducible generated examples when reporting problems.

Keep numerical changes separate from packaging and UI changes. Explain units, sign conventions, limits, numerical tolerances, and the evidence for changes to scientific methods. Keep tests meaningful: use independently calculated limits and small recovery examples rather than assertions that reproduce implementation logic.

The package uses one engine for both interfaces. The terminal entry point is `eis_fitter.cli:main`; the GUI entry point is `eis_fitter.cli:gui_main`. Scientific functions currently live in `eis_fitter.engine`; their signatures and return dictionaries should be treated as provisional in this initial release.

Proposed contributions must be compatible with GPL-3.0-only, preserve required attribution, and cite verified sources for scientific claims. Do not add author or institutional metadata without permission from those named.
