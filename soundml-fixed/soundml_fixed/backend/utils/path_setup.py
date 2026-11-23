"""
backend/utils/path_setup.py
────────────────────────────
Inserts the project root (the folder containing 'backend/')
into sys.path so that  `from backend.xxx import yyy`  works
no matter which directory the script is run from.

Usage — put this at the very top of any entrypoint:

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[N]))

Or simply:

    from backend.utils.path_setup import *   # (imported for side-effect)

This file is imported automatically by config.py, so you usually
don't need to import it explicitly.
"""

import sys
from pathlib import Path

# Walk up until we find the directory that CONTAINS 'backend/'
_here = Path(__file__).resolve()          # .../soundml/backend/utils/path_setup.py
_root = _here.parents[2]                  # .../soundml/

if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
