"""PyQt5 interface for EIS Fitter."""
from eis_fitter import __version__

# Import the solver stack before PyQt modules can load native GUI libraries.
from eis_fitter import engine
