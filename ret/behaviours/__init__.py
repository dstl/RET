"""Behaviours.

The `behaviours` module consists of a series of behaviours which provide the basic
building blocks for each agent's capability. Any agent can be defined with any of a
series of behaviours, depending on the specific agent's constructor. The following
documentation considers the behaviours available to the RetAgent, which is the full
set of behaviours available.

Each behaviour can either be set to an instance of a specific behaviour type, or None.

Where the behaviour is None, if the agent is instructed via a task to perform the
behaviour, ret will log a warning. Further details of creating an agent with
behaviours is covered in the documentation for an Agent, and each behaviour is described
individually within the submodules contained herein.
"""

from abc import ABC


class Behaviour(ABC):  # noqa: B024
    """Base Behaviour class."""

    @property
    def name(self) -> str:
        """Returns a representative name for the behaviour.

        Returns:
            str: Name of the behaviour
        """
        return type(self).__name__
