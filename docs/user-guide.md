# User guide

## Before you start

Use your own compatible Gamry EIS `.DTA` files. Choose a working folder and a writable results folder. The software searches recursively inside the selected working folder. It does not upload measurements.

Start with a conventional descending-frequency export. Rows retain their original order; the capacitive-region heuristic examines the second half of the rows, assuming they represent lower frequencies. The code does not automatically correct an inverted imaginary-sign convention. See [methods](methods.md) before interpreting nonstandard inputs.

## Desktop workflow

Launch `eis-fit-gui` from the environment where you installed the GUI extra.

1. Use **Open Directory** or the working-directory browser to select a folder. The file list shows `.DTA` files, newest first.
2. Select a file. Inspect the data preview and the number of retained capacitive points.
3. In frequency selection, choose endpoints or use the entire retained range. At least five usable points are required. Accepting a range runs a KK residual check.
4. Select a fitting method, then run the fit. Inspect the fitted curve, individual component visualization, parameters, and residuals.
5. For DRT, choose regularization settings and run the analysis. Combined and imaginary modes use raw points within the selected frequency interval, including points omitted by the capacitive filter.
6. Set the save directory and use the export controls. Completed analyses also log and automatically save results there.

The result tabs provide scrollbars for readable multi-panel figures; PNG exports use a fixed page size.

The interface supports light and dark themes. It remembers its own window geometry, theme, and selected directories. Remembered paths are local preferences.

### Fitting methods

| Method | Use |
| --- | --- |
| Fixed N Voigt | Fit a chosen number of parallel resistor–CPE branches, from 1 to 10 |
| BIC Auto-Scan | Compare 1–10 branches and choose the lowest implemented BIC score |
| R² Auto-Scan | Compare 1–10 branches and choose the highest R²; inspect the risk of overfitting |
| Manual / Bode peaks | Select Nyquist peaks or Bode phase features to guide initial bounds |

Increasing iterations, population size, restarts, or model size increases runtime. Automatic scans use their own solver budgets; the advanced controls apply to fixed-N and peak-guided fitting. The current restart allocation runs at least one restart of each of three strategies, even when fewer than three are requested.

### DRT controls

Choose `custom`, `GCV`, `mGCV`, or `LC` (L-curve) regularization selection, first- or second-difference regularization, real/imaginary/combined data, optional inductance, Gaussian RBF width coefficient, and nonnegative coefficients. Custom λ is editable only in custom mode. The tau grid follows the input frequency range; it is not manually specified.

## Terminal workflow

```bash
eis-fit --data-dir ./data --output-dir ./results
```

Both paths are relative to the terminal's current working directory. A missing input folder is an argument error. An empty folder prints a no-data message and exits with code 1 without creating the output folder. Data discovery also accepts uppercase `.DTA` extensions.

Choose a file number. Select frequency endpoints on the plot and close its window to continue. Closing without two clicks uses all retained points. The fitting menu supports:

| Choice | Action |
| --- | --- |
| `1`–`10` | Fixed number of branches |
| `b` | BIC scan |
| `a` | Best-R² scan |
| `m` / `p` | Nyquist / Bode peak-guided fitting |
| `d` | DRT analysis |
| `r` / `f` | Select frequency range / select another file |
| `q` | End session |

The application also supports `python -m eis_fitter` with the same arguments. The GUI command accepts `--help` and `--version`.

## Outputs

Exports include frequency and impedance columns, fitted values, residuals, parameter summaries, and analysis settings where applicable. CSV exports contain comment/metadata rows; inspect them before importing into another program. The text log appends analysis summaries and local input-file paths. Output filenames include the input basename. Review these user-generated files before sharing them.

The GUI can export PNG plots; the terminal workflow saves its analysis figures. Where ECM and DRT results coexist, exports follow the most recently completed analysis. Export counters distinguish analyses within a session but restart on application relaunch. **Use a new output folder for each session to avoid replacing files with the same names.**

## Troubleshooting

| Symptom | Check |
| --- | --- |
| No files listed | Select a folder containing compatible `.DTA` files; do not select its parent unintentionally |
| GUI dependency missing | From the repository folder, run `python -m pip install ".[gui]"` in the active environment |
| Qt platform error on Linux | A graphical session and Qt system libraries are required; some systems need `libegl1` and `libopengl0` |
| Plot window does not appear | Use an interactive Matplotlib backend and a desktop session; `Agg` is for automated tests |
| Incomplete table or malformed row | Check the file structure described in the [format guide](file-format.md) |
| Too few usable points | Inspect sign convention, inductive points, and frequency selection |
| Fit reaches a bound | Inspect model suitability and frequency coverage; do not treat a boundary value as independently identified |
| DRT solve fails | Check frequency range, duplicates, λ, and conditioning; reduce model demands or revise settings |

If reporting an issue, provide software versions, the error text, and a minimal generated example. Do not attach confidential measurements or identifying local paths.
