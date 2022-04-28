"""Utility class for handling pools of agent behaviours."""

from __future__ import annotations

import warnings
from abc import ABC
from typing import TYPE_CHECKING, Iterable, TypeVar

from .communicate import (
    CommunicateMissionMessageBehaviour,
    CommunicateOrdersBehaviour,
    CommunicateWorldviewBehaviour,
)
from .deploycountermeasure import DeployCountermeasureBehaviour
from .disablecommunication import DisableCommunicationBehaviour
from .fire import FireBehaviour
from .hide import HideBehaviour
from .move import MoveBehaviour
from .sense import SenseBehaviour
from .wait import WaitBehaviour

if TYPE_CHECKING:
    from collections.abc import Sequence
    from random import Random
    from typing import Callable, Optional

    from mesa_ret.agents.agent import RetAgent

    from . import Behaviour


T = TypeVar("T")


class ListAdder(ABC):
    """Abstract base class for sequence adder."""

    @classmethod
    def try_add_new(cls, arr: list[T], new: T) -> bool:  # pragma: no cover
        """Add the new item to the array if it matches implementation-specific criteria.

        Args:
            arr (list[T]): Array of objects
            new (T): New object, which may be added to the list.

        Returns:
            bool: True if the item is added, False otherwise.
        """
        return NotImplemented


class NoEqualityAdder(ListAdder):
    """Adds new object unless there's already an exact instance."""

    @classmethod
    def try_add_new(cls, arr: list[T], new: T) -> bool:
        """Adds new item to the array if it is not already in the array.

        Args:
            arr (list[T]): Array of objects
            new (T): New object, which may be added to the list.

        Returns:
            bool: True if the item is added, false otherwise.
        """
        if new in arr:
            return False
        arr.append(new)
        return True


class NoTypeEqualityAdder(ListAdder):
    """Adds new object unless there's an instance of the same type."""

    @classmethod
    def try_add_new(cls, arr: list[T], new: T) -> bool:
        """Adds new item to the array if there is no matching type in the array.

        Args:
            arr (list[T]): Array of objects
            new (T): New object, which may be added to the list.

        Returns:
            bool: True if the item is added, false otherwise.
        """
        types = [type(o) for o in arr]
        if type(new) in types:
            return False
        arr.append(new)
        return True


class NoSubtypeEqualityAdder(ListAdder):
    """Adds new objects if a subtype is not already present."""

    @classmethod
    def try_add_new(cls, arr: list[T], new: T) -> bool:
        """Adds new item to the array if there is no matching sub-type in the array.

        Args:
            arr (list[T]): Array of objects
            new (T): New object, which may be added to the list.

        Returns:
            bool: True if the item is added, false otherwise.
        """
        if any([isinstance(c, type(new)) for c in arr]):
            return False
        arr.append(new)
        return True


class TwoWayNoSubTypeEqualityAdder(ListAdder):
    """Adds new object is a subtype or supertype is not already present."""

    @classmethod
    def try_add_new(cls, arr: list[T], new: T) -> bool:
        """Adds new object is a subtype or supertype is not already present.

        Args:
            arr (list[T]): Array of objects
            new (T): New object, which may be added to the list.

        Returns:
            bool: True if the item is added, false otherwise.
        """
        if any(isinstance(c, type(new)) for c in arr) or any(isinstance(new, type(c)) for c in arr):
            return False
        arr.append(new)
        return True


class NeverAdder(ListAdder):
    """ListAdder which never adds new items."""

    @classmethod
    def try_add_new(cls, arr: list[T], new: T) -> bool:
        """Returns False, and does not modify arr.

        Args:
            arr (list[T]): Array of objects
            new (T): New object, which can never be added to the list

        Returns:
            bool: Always returns False.

        """
        return False


class AlwaysAdder(ListAdder):
    """ListAdder which always adds new items."""

    @classmethod
    def try_add_new(cls, arr: list[T], new: T) -> bool:
        """Returns true, and always adds new to arr.

        Args:
            arr (list[T]): Array of objects
            new (T): New object, which is always added to the list

        Returns:
            bool: Always returns True.

        """
        arr.append(new)
        return True


class BehaviourPool:
    """Handler for a grouping of possibly overlapping behaviours."""

    _candidates: list[Behaviour]
    _random: Random

    def __init__(
        self,
        agent: RetAgent,
        random: Random,
        adder: ListAdder,
        behaviours: Optional[Sequence[Behaviour]] = None,
    ):
        """Return a new behaviour pool.

        Args:
            agent (RetAgent): RET Agent associated with the behaviour pool.
            random (Random): Random number source
            adder (ListAdder): Mechanism for adding new behaviours to the pool.
            behaviours (Optional[Sequence[Behaviour]], optional): behaviours. Defaults
                to None.
        """
        self._adder = adder
        self._candidates = []

        if isinstance(behaviours, Iterable):
            for behaviour in behaviours:
                self.add_behaviour(behaviour)

        self._random = random
        self._agent = agent

    def add_behaviour(self, behaviour: Behaviour) -> None:
        """Add a behaviour to the behaviour pool if not already present.

        The add behaviour methodology delegates to the user-defined list adder, which
        determines whether or not the behaviour can be added to the agents behaviour
        pool.

        Where the behaviour is not added - A logging message will be raised, as it is
        likely that this is not the intent.

        Args:
            behaviour (Behaviour): Behaviour to add
        """
        add_new = self._adder.try_add_new(self._candidates, behaviour)
        if not add_new:
            warnings.warn(f"Cannot add duplicate behaviour '{behaviour.name}'")

    def add_default_behaviour(self, behaviour: Behaviour, type_match: type):
        """Add default behaviour to the behaviour pool.

        The default behaviour is only added if there is no behaviour of type type_match
        already present.

        Args:
            behaviour (Behaviour): New behaviour to add.
            type_match (type): Type to check presence for.

        """
        if any([isinstance(c, type_match) for c in self._candidates]):
            return
        else:
            self._candidates.append(behaviour)

    def remove_behaviour(self, behaviour: Behaviour) -> None:
        """Remove a behaviour from the behaviour pool, if it is present.

        Args:
            behaviour (Behaviour): The behaviour to remove
        """
        if behaviour in self._candidates:
            self._candidates.remove(behaviour)
        else:
            name = behaviour.name
            warnings.warn(f"Cannot remove non-existent behaviour '{name}'")

    def satisfy(self, handler: str, type: type) -> Optional[Callable]:
        """Return a callable method that will satisfy the required behaviour.

        Args:
            handler (str): Name of caller function
            type (type): Type of behaviour required

        Returns:
            Optional[Callable]: Method which satisfies the behavioural step.
        """
        choice: Optional[Behaviour] = self.choose_behaviour(handler, type)
        func: Optional[Callable] = getattr(choice, handler, None)
        return func

    def choose_behaviour(self, handler: str, type: type) -> Optional[Behaviour]:
        """Choose a behaviour that matches the required capability.

        Args:
            handler (str): Name of caller function
            type (type): Type of behaviour required

        Returns:
            Optional[Behaviour]: Behaviour which satisfies the requirement.
        """
        candidates = self.expose_behaviour(handler, type)
        choice = None
        if len(candidates) > 0:
            choice = self._random.choice(candidates)  # type: ignore
            self._agent.model.logger.log_behaviour_selection_record(
                self._agent, handler, type, candidates, choice
            )
        else:
            self._agent.model.logger.log_no_available_behaviour(self._agent, handler, type)
        return choice

    def expose_behaviour(self, handler: str, type: type) -> Sequence[Behaviour]:
        """Expose a list of behaviours that match the requested capability.

        Args:
            handler (str): Name of caller function
            type (type): Type of behaviour required

        Returns:
            Sequence[Behaviour]: All behaviours which satisfy capability.
        """
        candidates = [
            t
            for t in self._candidates
            if isinstance(t, type) and callable(getattr(t, handler, None))
        ]

        return candidates


class BehaviourHandlers:
    """Behaviour pool handlers."""

    @property
    def communicate_orders_handler(self) -> str:
        """Return handler for the communicate orders step.

        Returns:
            str: Communicate orders handler
        """
        return "step"

    @property
    def communicate_orders_type(self) -> type:
        """Return class for communicate orders behaviour.

        Returns:
            type: Communicate orders type
        """
        return CommunicateOrdersBehaviour

    @property
    def communicate_worldview_handler(self) -> str:
        """Return handler for the communicate worldview step.

        Returns:
            str: Communicate worldview handler
        """
        return "step"

    @property
    def communicate_worldview_type(self) -> type:
        """Return class for communication worldview behaviour.

        Returns:
            type: Communicate worldview type
        """
        return CommunicateWorldviewBehaviour

    @property
    def communicate_mission_message_handler(self) -> str:
        """Return handler for the communicate mission message step.

        Returns:
            str: Communicate mission message handler
        """
        return "step"

    @property
    def communicate_mission_message_type(self) -> type:
        """Return class for communication mission message behaviour.

        Returns:
            type: Communicate mission message type
        """
        return CommunicateMissionMessageBehaviour

    @property
    def deploy_countermeasure_handler(self) -> str:
        """Return handler for deploy countermeasures step.

        Returns:
            str: Deploy countermeasures handler
        """
        return "step"

    @property
    def deploy_countermeasure_type(self) -> type:
        """Return class for deploy countermeasures behaviour.

        Returns:
            type: Deploy countermeasures type
        """
        return DeployCountermeasureBehaviour

    @property
    def disable_communication_handler(self) -> str:
        """Return the handler for disable communication step.

        Returns:
            str: Disable communication handler
        """
        return "step"

    @property
    def disable_communication_type(self) -> type:
        """Return class for disable communication behaviour.

        Returns:
            type: Disable communication type
        """
        return DisableCommunicationBehaviour

    @property
    def fire_handler(self) -> str:
        """Return the handler for fire step.

        Returns:
            str: Fire handler
        """
        return "step"

    @property
    def fire_type(self) -> type:
        """Return the class for fire behaviour.

        Returns:
            type: Fire type
        """
        return FireBehaviour

    @property
    def move_handler(self) -> str:
        """Return handler for move step.

        Returns:
            str: Move handler
        """
        return "step"

    @property
    def move_type(self) -> type:
        """Return class for move behaviour.

        Returns:
            type: Move type
        """
        return MoveBehaviour

    @property
    def sense_handler(self) -> str:
        """Return handler for sense step.

        Returns:
            str: Sense handler
        """
        return "step"

    @property
    def sense_type(self) -> type:
        """Return class for sense behaviour.

        Returns:
            type: sense type
        """
        return SenseBehaviour

    @property
    def wait_handler(self) -> str:
        """Return handler for wait step.

        Returns:
            str: Wait handler
        """
        return "step"

    @property
    def wait_type(self) -> type:
        """Return class for wait behaviour.

        Returns:
            type: Wait type
        """
        return WaitBehaviour

    @property
    def hide_handler(self) -> str:
        """Return handler for hide step.

        Returns:
            str: Hide handler
        """
        return "step"

    @property
    def hide_type(self) -> type:
        """Return class for hide behaviour.

        Returns:
            type: Hide type
        """
        return HideBehaviour
