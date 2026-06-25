# do all imports here

from .ac_gae import MCAgent
from .ppo import PPOAgent
from .sac import SACAgent
from .ddqn import DDQNAgent

# NOTE: TRPOAgent was previously imported from `.trpo`, but agents/trpo.py
# was never committed to git -- only a stale compiled
# agents/__pycache__/trpo.cpython-310.pyc existed, with no matching .py
# source anywhere in history. That import was already broken before this
# env update; it's removed here rather than left to silently crash every
# `from agents import ...`. If you have the original trpo.py elsewhere,
# drop it back into agents/ and add `from .trpo import TRPOAgent` +
# "TRPOAgent" to __all__ below.

__all__ = ["MCAgent",
           "PPOAgent",
           "SACAgent",
           "DDQNAgent",]
