# Supported input files

Developed for Gamry EIS `.DTA` files containing a `ZCURVE TABLE` with `Freq`, `Zreal`, and `Zimag` columns.

The reader uses Latin-1 decoding and expects:

1. A line containing both `ZCURVE` and `TABLE`.
2. A tab-separated column-name row immediately after it.
3. One units row, followed by tab-separated numeric data rows.

The three required column names are case-sensitive and must each occur exactly once. Their order can vary. Other columns are ignored. The first blank line ends the table; nonnumeric or short rows before that terminator are rejected rather than silently truncated. Files that place another section immediately after the numeric rows need a blank separator.

| Column | Meaning | Required units |
| --- | --- | --- |
| `Freq` | Frequency | Hz |
| `Zreal` | Real impedance | Ω |
| `Zimag` | Imaginary impedance | Ω |

At least five rows are required. All three values must be finite and frequencies must be positive. Units are assumed, not converted from the units row. Rows are not sorted or deduplicated; inspect repeated frequencies and use conventional descending frequency order for the current preprocessing heuristic.

The same minimum-point validation applies after capacitive filtering and frequency selection. This reader is not a general parser for every Gamry file type or every `.DTA` table layout. No data files are bundled.
