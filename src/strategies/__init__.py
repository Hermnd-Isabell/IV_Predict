# Strategy implementations
from .strategy_a_residual import generate_signals as strategy_a_signals
from .strategy_b_section import generate_signals as strategy_b_signals
from .strategy_c_doublesort import generate_signals as strategy_c_signals

__all__ = [
    'strategy_a_signals',
    'strategy_b_signals',
    'strategy_c_signals',
]
