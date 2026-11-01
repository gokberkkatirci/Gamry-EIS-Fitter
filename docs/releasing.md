# Source release guide

## Suggested GitHub presentation

**Repository name:** `gamry-eis-fitter`

**Title:** Gamry EIS Fitter

**About:** Python GUI and interactive terminal tools for fitting Gamry EIS data, with equivalent-circuit models, Kramers–Kronig residual checks, DRT analysis, and plot exports.

**Topics:** `eis`, `electrochemical-impedance-spectroscopy`, `gamry`, `equivalent-circuit`, `drt`, `python`, `pyqt5`, `scientific-computing`

## Verify a release

1. Confirm rights to distribute the application source under GPL-3.0-only and preserve all required attribution. Confirm the package name before any package-index publication.
2. Follow [CONTRIBUTING](../CONTRIBUTING.md), then run:

   ```bash
   python -m build
   python tools/check_release.py --artifacts dist
   ```

3. Inspect the source inventory, source archive, and wheel. These must include application code, documentation, license information, and tests, with no measurement datasets or generated results.
4. Install the wheel into a new virtual environment outside the source directory. Check `eis-fit --help`, `eis-fit-gui --help`, no-data behavior, and the test suite. Confirm terminal-only installation does not install PyQt5.
5. Check the GUI in a graphical desktop session. Verify file selection, frequency selection, a small fit, KK/DRT completion, themes, and exports using generated inputs.
6. Run the full GitHub Actions matrix before changing the README's platform-verification statement. Record actual Python/dependency versions and limitations.

The inventory check is a packaging guard, not proof that every source constant is scientifically justified or that all identifiers are nonconfidential. Review the content before publication.

## Publish on GitHub

Place this project's top-level files, including hidden `.github/` and `.gitignore`, into a separate empty directory. The README must be at the new repository root. Create an empty GitHub repository named `gamry-eis-fitter`, then initialize and commit the reviewed files from that separate directory:

```bash
git init
git branch -M main
git add .
git status --short
git diff --cached --stat
git commit -m "Initial EIS Fitter source release"
```

Inspect the staged content before committing. Add the remote URL supplied by GitHub and push `main` when ready. Set the About text and topics above. Wait for CI, then create a `v0.1.0` release with the changelog and verified source artifacts. No executable installers are provided in this version.
