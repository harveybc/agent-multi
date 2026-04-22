"""Direction classification mixin for ioin plugins.

Extends BinaryMixin with direction-specific signal types.  All binary
classification logic (BinaryF1Score, val_f1 early stopping, balanced
class weights) is inherited — only VALID_SIGNAL_TYPES changes.

Usage::

    class Plugin(DirectionMixin, ANNPlugin):
        ...
"""
from __future__ import annotations

from ..binary.binary_base import (
    BinaryMixin,
    BinaryF1Score,
    _as_bool,
)

VALID_DIRECTION_TYPES = ("direction_long", "direction_short")


class DirectionMixin(BinaryMixin):
    """Overrides VALID_SIGNAL_TYPES for direction classification."""
    pass
