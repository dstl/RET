"""Tests for Behaviour Pool."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import TestCase

from ret.agents.affiliation import Affiliation
from ret.agents.agent import RetAgent
from ret.agents.agenttype import AgentType
from ret.behaviours import Behaviour
from ret.behaviours.behaviourpool import (
    AlwaysAdder,
    BehaviourPool,
    NeverAdder,
    NoEqualityAdder,
    NoSubtypeEqualityAdder,
    NoTypeEqualityAdder,
    TwoWayNoSubTypeEqualityAdder,
)
from ret.behaviours.hide import HideBehaviour
from ret.behaviours.wait import WaitBehaviour
from ret.testing.mocks import MockAgent, MockModel2d
from numpy.random import Generator, default_rng
from parameterized import parameterized

if TYPE_CHECKING:
    from typing import Any, Sequence


class BehaviourPoolTests(TestCase):
    """Tests for behaviour pool."""

    class EquatableBehaviour(Behaviour):
        """Mock behaviour that can be compared with other behaviours."""

        def __init__(self, id: int):
            """Create a new EquatableBehaviour with unique ID.

            Args:
                id (int): Unique ID
            """
            self._id = id

        def __eq__(self, o: object) -> bool:
            """Check equality of EquatableBehaviour.

            Args:
                o (object): Object to compare to

            Returns:
                bool: True if objects are equal. False otherwise.
            """
            if isinstance(o, self.__class__):
                return self._id == o._id
            return False

    def test_behaviour_pool_initialisation_with_behaviours(self):
        """Test that behaviour pools can be initialised with behaviours.

        Note that `bp.expose_behaviour("__str__", object)` used in multiple places to
        check the presence of objects in the behaviour pool - All objects in the pool
        will be subtypes of object and will have executable `__str__` methods.
        """
        r = default_rng()
        a = MockAgent(1, (0, 0))
        adder = NoEqualityAdder()
        # Simple Behaviour Pool initialised with 2x instances of a WaitBehaviour
        wb1 = WaitBehaviour()
        wb2 = WaitBehaviour()

        bp = BehaviourPool(a, r, adder, [wb1, wb2])
        behaviours = bp.expose_behaviour("__str__", object)
        assert len(behaviours) == 2

        assert wb1 in behaviours
        assert wb2 in behaviours

    def test_behaviour_pool_initialisation_without_behaviours(self):
        """Test that behaviour pools can be initialised without behaviours."""
        r = default_rng()
        a = MockAgent(1, (0, 0))
        adder = NoEqualityAdder()

        bp = BehaviourPool(a, r, adder)
        assert [] == bp.expose_behaviour("__str__", object)

    def test_behaviour_pool_add_behaviour(self):
        """Test adding a behaviour to a pool."""
        r = default_rng()
        a = MockAgent(1, (0, 0))
        adder = NoEqualityAdder()

        bp = BehaviourPool(a, r, adder)
        assert [] == bp.expose_behaviour("__str__", object)

        wb = WaitBehaviour()
        bp.add_behaviour(wb)

        behaviours = bp.expose_behaviour("__str__", object)
        assert len(behaviours) == 1
        assert wb in behaviours

    def test_behaviour_pool_add_duplicate_behaviour(self):
        """Test adding behaviour to a pool where it already exists."""
        r = default_rng()
        a = MockAgent(1, (0, 0))
        adder = NoEqualityAdder()

        bp = BehaviourPool(a, r, adder, [self.EquatableBehaviour(1)])

        with self.assertWarns(Warning) as w:
            bp.add_behaviour(self.EquatableBehaviour(1))

        assert "Cannot add duplicate behaviour 'EquatableBehaviour'" in str(w.warnings[0].message)

        # The duplicate behaviour is not added
        assert len(bp.expose_behaviour("__str__", object)) == 1

    def test_behaviour_pool_remove_behaviour(self):
        """Test removing behaviour from a pool."""
        r = default_rng()
        a = MockAgent(1, (0, 0))
        adder = NoEqualityAdder()

        bp = BehaviourPool(a, r, adder, [self.EquatableBehaviour(1), self.EquatableBehaviour(2)])

        bp.remove_behaviour(self.EquatableBehaviour(1))
        behaviours = bp.expose_behaviour("__str__", object)
        assert len(behaviours) == 1
        assert self.EquatableBehaviour(2) in behaviours

        bp.remove_behaviour(self.EquatableBehaviour(2))
        behaviours = bp.expose_behaviour("__str__", object)
        assert len(behaviours) == 0

    def test_behaviour_pool_remove_non_existent_behaviour(self):
        """Test removing a behaviour where it doesn't exist."""
        r = default_rng()
        a = MockAgent(1, (0, 0))
        adder = NoEqualityAdder()

        bp = BehaviourPool(a, r, adder)

        b1 = WaitBehaviour()
        b2 = WaitBehaviour()
        bp = BehaviourPool(a, r, adder, [b1, b2])

        before = bp.expose_behaviour("__str__", object)

        with self.assertWarns(Warning) as w:
            bp.remove_behaviour(self.EquatableBehaviour(1))

        assert "Cannot remove non-existent behaviour 'EquatableBehaviour'" in str(
            w.warnings[0].message
        )

        after = bp.expose_behaviour("__str__", object)

        assert before == after

    def test_behaviour_pool_random_selection(self):
        """Test random selection from pool."""

        class FauxRandom(Generator):
            def __init__(self, choice_position):
                self._choice_position = choice_position

            def choice(self, a: Sequence[Any]) -> Any:
                return a[self._choice_position]

        r = FauxRandom(0)
        m = MockModel2d()
        a = RetAgent(m, (0, 0), "model", Affiliation.NEUTRAL, 2.0, 0.1, 20.0, AgentType.GENERIC)
        adder = NoEqualityAdder()

        bp = BehaviourPool(a, r, adder, [self.EquatableBehaviour(0), self.EquatableBehaviour(1)])

        for _ in range(0, 10):
            selection = bp.choose_behaviour("__str__", object)
            assert selection == self.EquatableBehaviour(0)

        # Monkey Patch the random to demonstrate selecting with a different rule
        bp._random = FauxRandom(1)  # type: ignore

        for _ in range(0, 10):
            selection = bp.choose_behaviour("__str__", object)
            assert selection == self.EquatableBehaviour(1)

    def test_behaviour_pool_select_custom_handler(self):
        """Test selection of behaviour based on a custom handler."""
        r = default_rng()
        m = MockModel2d()
        a = RetAgent(m, (0, 0), "model", Affiliation.NEUTRAL, 2.0, 0.1, 20.0, AgentType.GENERIC)
        adder = NoEqualityAdder()

        bp = BehaviourPool(a, r, adder, [self.EquatableBehaviour(0), WaitBehaviour()])

        equatable_behaviour = bp.choose_behaviour("__str__", self.EquatableBehaviour)
        assert isinstance(equatable_behaviour, self.EquatableBehaviour)

        wait_behaviour = bp.choose_behaviour("__str__", WaitBehaviour)
        assert isinstance(wait_behaviour, WaitBehaviour)

    def test_behaviour_pool_select_custom_type(self):
        """Test selection of custom behavioural method from pool."""

        class SideEffectBehaviour(Behaviour):
            def __init__(self):
                self.outcome = None

            def side_effect(self, arg):
                self.outcome = arg

        r = default_rng()
        m = MockModel2d()
        a = RetAgent(m, (0, 0), "model", Affiliation.NEUTRAL, 2.0, 0.1, 20.0, AgentType.GENERIC)
        adder = NoEqualityAdder()

        behaviour = SideEffectBehaviour()

        bp = BehaviourPool(a, r, adder, [behaviour])

        side_effect = bp.satisfy("side_effect", SideEffectBehaviour)

        assert behaviour.outcome is None

        side_effect("Some data")  # type: ignore
        assert behaviour.outcome == "Some data"

    def test_select_subclass(self):
        """Test that a subclass of the requested type can be selected from the pool."""

        class Subclass(WaitBehaviour):
            pass

        r = default_rng()
        m = MockModel2d()
        a = RetAgent(m, (0, 0), "model", Affiliation.NEUTRAL, 2.0, 0.1, 20.0, AgentType.GENERIC)
        adder = NoEqualityAdder()

        behaves_as_wait_behaviour = Subclass()
        bp = BehaviourPool(a, r, adder, [behaves_as_wait_behaviour])

        wait_behaviour = bp.choose_behaviour("__str__", WaitBehaviour)
        assert behaves_as_wait_behaviour == wait_behaviour


class NoEqualityAdderTests(TestCase):
    """Test cases for various NoEqualityAdder ListAdder."""

    def test_add_invalid(self):
        """Test that the NoEqualityAdder will not add duplicate instances."""
        wb = WaitBehaviour()
        lst = [wb]
        adder = NoEqualityAdder()

        success = adder.try_add_new(lst, wb)

        assert len(lst) == 1
        assert success is False
        assert wb in lst

    def test_add_valid(self):
        """Test that the NoEqualityAdder will add where items do not match."""
        wb1 = WaitBehaviour()
        wb2 = WaitBehaviour()
        lst = [wb1]
        adder = NoEqualityAdder()

        success = adder.try_add_new(lst, wb2)
        assert len(lst) == 2
        assert success is True
        assert wb1 in lst
        assert wb2 in lst

    def test_add_with_equality_override(self):
        """Test that the NoEqualityAdder will ignore items with matching __eq__."""

        class EquatableWait(WaitBehaviour):
            """Equatable wait, which returns True where compared to another Wait."""

            def __eq__(self, o: object) -> bool:
                """Compare to another instance.

                Args:
                    o (object): Object to compare to

                Returns:
                    bool: Result of comparison
                """
                return isinstance(o, self.__class__)

        wb1 = EquatableWait()
        wb2 = EquatableWait()
        lst = [wb1]
        adder = NoEqualityAdder()

        success = adder.try_add_new(lst, wb2)
        assert len(lst) == 1
        assert success is False
        assert wb1 in lst

        # Note - can't assert that wb1 is not in the list, because it will also use the
        # overridden equality


class CustomWaitBehaviour(WaitBehaviour):
    """Alternatively typed WaitBehaviour."""

    pass


class CustomNonWaitBehaviour(Behaviour):
    """Behaviour that doesn't extend WaitBehaviour."""

    pass


class NoTypeEqualityAdderTests(TestCase):
    """Test cases for various NoTypeEqualityAdder ListAdder."""

    def test_add_the_same_invalid(self):
        """Test that the NoTypeEqualityAdder will not add duplicate instances."""
        wb = WaitBehaviour()
        lst = [wb]
        adder = NoTypeEqualityAdder()

        success = adder.try_add_new(lst, wb)

        assert len(lst) == 1
        assert success is False
        assert wb in lst

    def test_add_same_type_invalid(self):
        """Test that the NoTypeEqualityAdder will not add two of the same type."""
        wb1 = WaitBehaviour()
        wb2 = WaitBehaviour()
        lst = [wb1]
        adder = NoTypeEqualityAdder()

        success = adder.try_add_new(lst, wb2)
        assert len(lst) == 1
        assert success is False
        assert wb1 in lst
        assert wb2 not in lst

    def test_add_valid(self):
        """Test that the NoTypeEqualityAdder will add where item type do not match."""
        wb1 = WaitBehaviour()
        wb2 = CustomWaitBehaviour()
        lst = [wb1]
        adder = NoTypeEqualityAdder()

        success = adder.try_add_new(lst, wb2)
        assert len(lst) == 2
        assert success is True
        assert wb1 in lst
        assert wb2 in lst


class NoSubTypeEqualityAdderTests(TestCase):
    """Test cases for various NoSubtypeEqualityAdder ListAdder."""

    def test_add_the_same_invalid(self):
        """Test that the NoSubtypeEqualityAdder will not add duplicate instances."""
        wb = WaitBehaviour()
        lst: list[Behaviour] = [wb]
        adder = NoSubtypeEqualityAdder()

        success = adder.try_add_new(lst, wb)

        assert len(lst) == 1
        assert success is False
        assert wb in lst

    def test_add_same_type_invalid(self):
        """Test that the NoSubtypeEqualityAdder will not add two of the same type."""
        wb1 = WaitBehaviour()
        wb2 = WaitBehaviour()
        lst: list[Behaviour] = [wb1]
        adder = NoSubtypeEqualityAdder()

        success = adder.try_add_new(lst, wb2)
        assert len(lst) == 1
        assert success is False
        assert wb1 in lst
        assert wb2 not in lst

    def test_add_subtype_invalid(self):
        """Test that the NoSubtypeEqualityAdder will not add two of the same type."""
        wb1 = CustomWaitBehaviour()
        wb2 = WaitBehaviour()
        lst: list[Behaviour] = [wb1]
        adder = NoSubtypeEqualityAdder()

        success = adder.try_add_new(lst, wb2)
        assert len(lst) == 1
        assert success is False
        assert wb1 in lst
        assert wb2 not in lst

    def test_add_subtype_first_valid(self):
        """Test that adding more specific types first is allowed."""
        wb1 = WaitBehaviour()
        wb2 = CustomWaitBehaviour()
        lst: list[Behaviour] = [wb1]
        adder = NoSubtypeEqualityAdder()

        success = adder.try_add_new(lst, wb2)
        assert len(lst) == 2
        assert success is True
        assert wb1 in lst
        assert wb2 in lst

    def test_add_valid(self):
        """Test that the NoSubTypeEqualityAdder will add where item type don't match."""
        wb1 = WaitBehaviour()
        wb2 = CustomWaitBehaviour()
        lst: list[Behaviour] = [wb1]
        adder = NoSubtypeEqualityAdder()

        success = adder.try_add_new(lst, wb2)
        assert len(lst) == 2
        assert success is True
        assert wb1 in lst
        assert wb2 in lst


class TwoWayNoSubTypeEqualityAdderTests(TestCase):
    """Test cases for various TwoWayNoSubtypeEqualityAdder ListAdder."""

    def test_add_the_same_invalid(self):
        """Test that the TwoWayNoSubtypeEqualityAdder will not add duplicate instances."""
        wb = WaitBehaviour()
        lst: list[Behaviour] = [wb]
        adder = TwoWayNoSubTypeEqualityAdder()

        success = adder.try_add_new(lst, wb)

        assert len(lst) == 1
        assert success is False
        assert wb in lst

    def test_add_same_type_invalid(self):
        """Test that the TwoWayNoSubtypeEqualityAdder will not add two of the same type."""
        wb1 = WaitBehaviour()
        wb2 = WaitBehaviour()
        lst: list[Behaviour] = [wb1]
        adder = TwoWayNoSubTypeEqualityAdder()

        success = adder.try_add_new(lst, wb2)
        assert len(lst) == 1
        assert success is False
        assert wb1 in lst
        assert wb2 not in lst

    def test_add_subtype_invalid(self):
        """Test that the adder will not add if an object subtype or supertype is present."""
        wb1 = CustomWaitBehaviour()
        wb2 = WaitBehaviour()
        lst: list[Behaviour] = [wb1]
        adder = TwoWayNoSubTypeEqualityAdder()

        success = adder.try_add_new(lst, wb2)
        assert len(lst) == 1
        assert success is False
        assert wb1 in lst
        assert wb2 not in lst

    def test_add_valid(self):
        """Test that the adder will add where subtypes and supertypes don't match."""
        wb1 = WaitBehaviour()
        wb2 = HideBehaviour()
        lst: list[Behaviour] = [wb1]
        adder = TwoWayNoSubTypeEqualityAdder()

        success = adder.try_add_new(lst, wb2)
        assert len(lst) == 2
        assert success is True
        assert wb1 in lst
        assert wb2 in lst


class NeverAdderTest(TestCase):
    """Tests for NeverAdded ListAdder."""

    @parameterized.expand([[CustomWaitBehaviour()], [CustomNonWaitBehaviour()], [WaitBehaviour()]])
    def test_add_invalid(self, other: Behaviour):
        """Test adding with a NeverAdder always fails.

        Args:
            other (Behaviour): Behaviour to add
        """
        wb1 = WaitBehaviour()
        wb2 = other
        lst: list[Behaviour] = [wb1]
        adder = NeverAdder()

        success = adder.try_add_new(lst, wb2)
        assert len(lst) == 1
        assert success is False
        assert wb1 in lst
        assert wb2 not in lst


class AlwaysAdderTest(TestCase):
    """Tests for AlwaysAdder ListAdder."""

    @parameterized.expand([[CustomWaitBehaviour()], [CustomNonWaitBehaviour()], [WaitBehaviour()]])
    def test_add_valid(self, other: Behaviour):
        """Test adding with a AlwaysAdder always passes.

        Args:
            other (Behaviour): Behaviour to add
        """
        wb1 = WaitBehaviour()
        wb2 = other
        lst: list[Behaviour] = [wb1]
        adder = AlwaysAdder()

        success = adder.try_add_new(lst, wb2)
        assert len(lst) == 2
        assert success is True
        assert wb1 in lst
        assert wb2 in lst

    def test_add_self_valid(self):
        """Test that adding a duplicate to a AlwaysAdder always passes."""
        wb = WaitBehaviour()
        lst: list[Behaviour] = [wb]
        adder = AlwaysAdder()
        success = adder.try_add_new(lst, wb)
        assert len(lst) == 2
        assert success is True
        assert wb in lst
        assert len(set(lst)) == 1
