"""Pluggable callbacks for solver iterations."""

from .cem import (
    EliteCostRecorder,
    EliteSpreadRecorder,
    MeanShiftRecorder,
    VarNormRecorder,
)
from .common import (
    BestCostRecorder,
    CandidateTraceRecorder,
    Callback,
    MeanCostRecorder,
)
from .gd import (
    ActionNormRecorder,
    GradNormRecorder,
)


__all__ = [
    'Callback',
    'BestCostRecorder',
    'CandidateTraceRecorder',
    'MeanCostRecorder',
    'GradNormRecorder',
    'ActionNormRecorder',
    'EliteCostRecorder',
    'VarNormRecorder',
    'MeanShiftRecorder',
    'EliteSpreadRecorder',
]
