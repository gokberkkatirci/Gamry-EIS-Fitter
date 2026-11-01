import importlib.util
from pathlib import Path

import pytest


def test_source_inventory_guards():
    checker_path = Path(__file__).resolve().parents[1] / "tools" / "check_release.py"
    spec = importlib.util.spec_from_file_location("release_check", checker_path)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    assert checker.allowed("src/eis_fitter/engine.py")
    assert checker.allowed("docs/images/gui-empty.png")
    for name in ("data/input.DTA", "results/fit.csv", "docs/measurement.png", "archive.zip"):
        assert not checker.allowed(name)


def test_inventory_rejects_symlinks(tmp_path):
    checker_path = Path(__file__).resolve().parents[1] / "tools" / "check_release.py"
    spec = importlib.util.spec_from_file_location("release_check", checker_path)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    target = tmp_path / "outside.txt"
    target.write_text("not a release input")
    root = tmp_path / "project"
    root.mkdir()
    try:
        (root / "link").symlink_to(target)
    except OSError:
        pytest.skip("Creating symlinks requires privileges on this platform")
    with pytest.raises(ValueError, match="Symlinks"):
        checker.inspect_source(root)
