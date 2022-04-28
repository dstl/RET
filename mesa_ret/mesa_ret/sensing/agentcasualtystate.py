"""Casualty states for agents."""

from enum import Enum


class AgentCasualtyState(Enum):
    """Agent casualty states."""

    ALIVE = 1
    KILLED = 2
    UNKNOWN = 10
