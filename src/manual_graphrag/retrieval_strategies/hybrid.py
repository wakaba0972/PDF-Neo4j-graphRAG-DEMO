"""Compatibility module alias for the relocated strategy implementation."""

import sys

from strategies.implementations.hybrid import strategy as _implementation

sys.modules[__name__] = _implementation
