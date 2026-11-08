"""Runs the bench command line: `uv run python -m bench gap`, `compare`, or `acceptance`."""

from __future__ import annotations

import sys

from bench.cli import main

sys.exit(main())
