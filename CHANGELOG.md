# Changelog

## 0.1.0

- Windows dependency compatibility: constrain CVXPY below 1.9 to avoid the
  QDLDL native import crash observed with CVXPY 1.9.3.
- CI uses Node.js 24 actions, UTF-8 Python output, an explicit Ubuntu 24.04
  runner, and a native solver import check before tests.

- GUI and interactive terminal interfaces using one shared numerical engine.
- Equivalent-circuit fitting with fixed or automatically selected branch counts and peak-guided fitting.
- KK residual checks, Gaussian-RBF DRT analysis, and numerical/plot exports.
- Explicit installation, optional GUI dependency, and descriptive package/module names.
- DRT controls mapped to supported engine arguments.
- Input validation, generated-input automated tests, and source packaging checks.
- User guides covering installation, workflows, formats, methods, and limitations.
