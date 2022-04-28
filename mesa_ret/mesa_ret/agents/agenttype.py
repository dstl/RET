"""Agent types."""

from enum import Enum


class AgentType(Enum):
    """Agent types."""

    GENERIC = 1
    INFANTRY = 2
    AIR = 3
    ARMOUR = 4
    AIR_DEFENCE = 5
    MECHANISED_INFANTRY = 6
    SENSOR_FUSION = 7
    GROUP = 8
    PROTECTED_ASSET = 9
    OTHER = 10
    UNKNOWN = 11
