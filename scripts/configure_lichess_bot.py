#!/usr/bin/env python3
"""Compatibility wrapper for ``luna-lichess-config``."""

from luna.lichess_config import main

if __name__ == "__main__":
    raise SystemExit(main())
