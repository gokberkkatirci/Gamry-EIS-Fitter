import subprocess
import sys

from eis_fitter.cli import main


def test_module_help_outside_project(tmp_path):
    result = subprocess.run([sys.executable, "-m", "eis_fitter", "--help"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0
    assert "--data-dir" in result.stdout
    assert "--output-dir" in result.stdout


def test_no_data_has_no_output_side_effect(tmp_path, capsys):
    output = tmp_path / "results"
    assert main(["--data-dir", str(tmp_path), "--output-dir", str(output)]) == 1
    assert "No .DTA" in capsys.readouterr().err
    assert not output.exists()


def test_import_does_not_launch_or_install(tmp_path):
    script = '''
import subprocess
def forbidden(*args, **kwargs):
    raise AssertionError("Import attempted to launch a subprocess")
subprocess.check_call = forbidden
subprocess.run = forbidden
import eis_fitter.engine
import sys
assert not any(name.startswith("PyQt5") for name in sys.modules)
'''
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_terminal_selects_generated_file_and_quits(tmp_path, monkeypatch, dta_factory):
    import numpy as np
    from eis_fitter import engine

    dta_factory()
    choices = iter(["1", "q"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(choices))
    monkeypatch.setattr(engine, "select_frequency_range",
                        lambda frequency, *args, **kwargs: np.ones(len(frequency), dtype=bool))
    output = tmp_path / "results"
    assert main(["--data-dir", str(tmp_path), "--output-dir", str(output)]) == 0
    assert "Kramers-Kronig" in (output / "fit_log.txt").read_text()
