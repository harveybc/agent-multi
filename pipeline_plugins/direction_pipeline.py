#!/usr/bin/env python
"""Direction Classification Pipeline Plugin.

Thin wrapper around BinaryPipelinePlugin with direction-specific defaults
(phase key, output file names).  All binary metrics, plots, and OLAP
logic are reused unchanged — direction classification uses the same
BinaryCrossentropy / F1 / AUC metrics as the oracle-binary pipeline.
"""
from .binary_pipeline import BinaryPipelinePlugin


class DirectionPipelinePlugin(BinaryPipelinePlugin):
    plugin_params = {
        **BinaryPipelinePlugin.plugin_params,
        "olap_phase_key": "phase_1c_direction",
    }
