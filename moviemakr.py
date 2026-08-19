#!/usr/bin/env python3
"""Launcher. The implementation lives in the `moviemakr` package next to this file.

Kept so `./moviemakr.py render scripts/foo.yaml` works without installing
anything. `python -m moviemakr` and the installed `moviemakr` script are
equivalent entry points.
"""

import sys

from moviemakr.cli import main

if __name__ == "__main__":
    sys.exit(main())
