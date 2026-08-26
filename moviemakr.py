#!/usr/bin/env python3
"""Launcher. The implementation lives in the `moviemakr` package next to this file.

Kept so `./moviemakr.py render "$MOVIEMAKR_WORKSPACE/scripts/foo.yaml"` works
without installing anything. `python -m moviemakr` and the installed `moviemakr`
script are equivalent entry points.

Script paths are ordinary arguments, not workspace-relative; the workspace
(`--workspace` or `$MOVIEMAKR_WORKSPACE`) only says where assets and renders
live, and one of the two is required.
"""

import sys

from moviemakr.cli import main

if __name__ == "__main__":
    sys.exit(main())
