"""Renamed: the prediction contract is now the solution (dclab_rnd.studio.solution). Kept so old imports work."""
from .solution import *  # noqa: F401,F403
from .solution import Solution as Contract, propose  # noqa: F401
