"""Command-line entry points; dependencies are installed explicitly by pip."""

import argparse
import multiprocessing
from pathlib import Path
import sys

from . import __version__


def main(argv=None):
    """Launch the interactive terminal workflow."""
    parser = argparse.ArgumentParser(
        description="Fit Gamry EIS .DTA files interactively. Plot windows require a desktop."
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--data-dir", type=Path, default=Path.cwd(),
                        help="directory to search recursively (default: current directory)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path.cwd() / "EIS_Fit_Results",
                        help="results directory (default: ./EIS_Fit_Results)")
    args = parser.parse_args(argv)
    if not args.data_dir.is_dir():
        parser.error("--data-dir must be an existing directory")
    from . import engine

    if not engine.find_all_dta_files(args.data_dir):
        print("No .DTA files found in the selected directory tree.", file=sys.stderr)
        return 1
    multiprocessing.freeze_support()
    try:
        return engine.main(str(args.data_dir.resolve()), str(args.output_dir.resolve())) or 0
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print("\nSession ended.", file=sys.stderr)
        return 0


def gui_main(argv=None):
    """Launch the optional PyQt5 interface."""
    parser = argparse.ArgumentParser(description="Launch the EIS Fitter GUI.")
    parser.add_argument("--version", action="version", version=__version__)
    parser.parse_args(argv)
    try:
        from .gui.main import run
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("PyQt5"):
            print('GUI dependency missing. Install with: python -m pip install ".[gui]"',
                  file=sys.stderr)
            return 1
        raise
    multiprocessing.freeze_support()
    run()
    return 0
