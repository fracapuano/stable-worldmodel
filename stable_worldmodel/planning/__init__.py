"""Pluggable planning cost layer for stable world models.

Decouples the planning cost from the world model: a ``ShootingCostEvaluator`` composes
any model exposing the ``Dynamics`` surface (``encode``/``rollout``) with a
swappable ``Objective``, and duck-types as the ``Costable`` the solvers expect.
"""

from stable_worldmodel.planning.evaluator import (
    ShootingCostEvaluator,
    default_goal_encode,
)
from stable_worldmodel.planning.objective import (
    ControlPenalty,
    GoalMSE,
    WeightedSum,
)

# The planning algorithms live in the ``solver`` subpackage; re-export them here
# so ``from stable_worldmodel.planning import CEMSolver`` works alongside
# ``from stable_worldmodel.planning.solver import CEMSolver``.
from stable_worldmodel.planning.solver import (
    AcceleratedCEMSolver,
    CategoricalCEMSolver,
    CEMSolver,
    GradientSolver,
    ICEMSolver,
    LagrangianSolver,
    MPPISolver,
    PGDSolver,
    PopulationAcceleratedCEMSolver,
    PredictiveSamplingSolver,
    Solver,
)
from stable_worldmodel.planning.tensor_cost import (
    LatentGoalCost,
    PopulationLatentGoalCost,
)

# Re-exported from stable_worldmodel.protocols (their canonical home) so that
# ``from stable_worldmodel.planning import Objective, Dynamics`` keeps working.
from stable_worldmodel.protocols import (
    Constrainable,
    Costable,
    Dynamics,
    Objective,
)

__all__ = [
    'AcceleratedCEMSolver',
    'CEMSolver',
    'CategoricalCEMSolver',
    'Constrainable',
    'ControlPenalty',
    'Costable',
    'Dynamics',
    'GoalMSE',
    'GradientSolver',
    'ICEMSolver',
    'LagrangianSolver',
    'LatentGoalCost',
    'MPPISolver',
    'Objective',
    'PGDSolver',
    'PopulationAcceleratedCEMSolver',
    'PopulationLatentGoalCost',
    'PredictiveSamplingSolver',
    'ShootingCostEvaluator',
    'Solver',
    'WeightedSum',
    'default_goal_encode',
]
