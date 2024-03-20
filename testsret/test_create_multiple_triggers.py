"""Tests for multiple trigger creation methods."""

import unittest
from datetime import datetime

from ret.creator.triggers import (
    _check_arg_not_none,
    _check_optional_arg_not_none,
    create_triggers,
)
from ret.orders.triggers.immediate import ImmediateSensorFusionTrigger, ImmediateTrigger
from ret.orders.triggers.killed import AgentKilledTrigger, KilledAgentsAtPositionTrigger
from ret.orders.triggers.position import (
    AliveAgentsAtPositionTrigger,
    CrossedBoundaryTrigger,
    InAreaTrigger,
    MovedOutOfAreaTrigger,
    NotInAreaTrigger,
    PositionTrigger,
)
from ret.orders.triggers.time import TimeTrigger
from ret.orders.triggers.triggertype import TriggerType
from ret.orders.triggers.weapon import (
    AgentFiredWeaponTrigger,
    WeaponFiredNearAgentTrigger,
    WeaponFiredNearLocationTrigger,
)
from ret.orders.triggers.sensor import AgentsSensedTrigger, AgentsSensedInRangeTrigger
from ret.space.feature import BoxFeature, LineFeature
from ret.testing.mocks import MockAgent
from ret.agents.agent import Affiliation, AgentCasualtyState, AgentType
from ret.sensing.sensor import Confidence


class TestCheckOptionalArgValidity(unittest.TestCase):
    """Tests for the check_optional_arg_validity method."""

    def test_no_none_values(self):
        """Check that the method returns the non-default values if passed in."""
        assert _check_optional_arg_not_none(True, False, True) == [True, False, True]

    def test_all_none_values(self):
        """Check that the method returns default values if None types are passed in."""
        assert _check_optional_arg_not_none(None, None, None) == [False, True, False]

    def test_some_none_values(self):
        """Check that the method returns correct values if only some None types are passed in."""
        assert _check_optional_arg_not_none(True, None, None) == [True, True, False]


class TestCheckArgValidity(unittest.TestCase):
    """Tests for the check_arg_validity method."""

    def test_no_none_values(self):
        """Check that the method returns True for valid inputs."""
        assert _check_arg_not_none([1])

    def test_only_none_values(self):
        """Check that the method raises and exception if only None types are passed in."""
        with self.assertRaises(TypeError) as e:
            _check_arg_not_none([None])
        self.assertEqual(
            "Argument required for selected trigger not given.",
            str(e.exception),
        )

    def test_some_none_values(self):
        """Check that the method raises and exception if some None types are passed in."""
        with self.assertRaises(TypeError) as e:
            _check_arg_not_none([1, None, 1, None])
        self.assertEqual(
            "Argument required for selected trigger not given.",
            str(e.exception),
        )


class TestCreateMultipleTriggers(unittest.TestCase):
    """Test that the method successfully creates multiple correct triggers."""

    def setUp(self):
        """Creates the relevant inputs for the test."""
        self.trigger_types = [
            TriggerType.IMMEDIATE,
            TriggerType.IMMEDIATE_SENSOR_FUSION,
            TriggerType.KILLED_AGENTS_AT_POSITION,
            TriggerType.ALIVE_AGENTS_AT_POSITION,
            TriggerType.AGENT_AT_POSITION,
            TriggerType.AGENT_IN_AREA,
            TriggerType.AGENT_NOT_IN_AREA,
            TriggerType.AGENT_CROSSED_BOUNDARY,
            TriggerType.AGENT_MOVED_OUT_OF_AREA,
            TriggerType.AGENT_KILLED,
            TriggerType.TIME,
            TriggerType.AGENT_FIRED_WEAPON,
            TriggerType.WEAPON_FIRED_NEAR_AGENT,
            TriggerType.WEAPON_FIRED_NEAR_LOCATION,
            TriggerType.AGENTS_SENSED,
            TriggerType.AGENTS_SENSED_IN_RANGE,
        ]
        self.sticky = [
            True,
            True,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
        ]
        self.log = False
        self.invert = None
        self.position = [
            None,
            None,
            (1, 1),
            (0, 0),
            (0, 0),
            (1, 0),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            (1, 1),
            None,
            None,
        ]
        self.agent = MockAgent(1, (0, 1))
        self.tolerance = 0.5
        self.area = BoxFeature((0, 0), (1, 1), "Test Box Area")
        self.boundary = LineFeature((0, 0), (1, 1), "Test Box Area")
        self.time = datetime(2020, 1, 1, 0, 0)
        self.affiliation = Affiliation.FRIENDLY
        self.agent_types = AgentType.GENERIC
        self.casualty = AgentCasualtyState.ALIVE
        self.sense_range = 1000
        self.confidence = Confidence.IDENTIFY

    def test_create_multiple_triggers(self):
        """Test that the method can be used to create multiple triggers.

        This test also checks that different types of trigger can be created, a single passed into
        an arg is used for all of the relevant triggers, lists passed into arguments can contain
        None entries, default values are used for optional arguments where None has been
        passed in, and that parameters not relevant to a trigger type are ignored.
        """
        test_triggers = create_triggers(
            number=16,
            trigger_type=self.trigger_types,
            sticky=self.sticky,
            log=self.log,
            invert=self.invert,
            position=self.position,
            agent=self.agent,
            tolerance=self.tolerance,
            area=self.area,
            boundary=self.boundary,
            time=self.time,
            agent_affiliation=self.affiliation,
            agent_types=self.agent_types,
            casualty_state=self.casualty,
            confidence_level=self.confidence,
            sense_range=self.sense_range,
        )

        assert isinstance(test_triggers[0], ImmediateTrigger)
        assert isinstance(test_triggers[1], ImmediateSensorFusionTrigger)
        assert isinstance(test_triggers[2], KilledAgentsAtPositionTrigger)
        assert isinstance(test_triggers[3], AliveAgentsAtPositionTrigger)
        assert isinstance(test_triggers[4], PositionTrigger)
        assert isinstance(test_triggers[5], InAreaTrigger)
        assert isinstance(test_triggers[6], NotInAreaTrigger)
        assert isinstance(test_triggers[7], CrossedBoundaryTrigger)
        assert isinstance(test_triggers[8], MovedOutOfAreaTrigger)
        assert isinstance(test_triggers[9], AgentKilledTrigger)
        assert isinstance(test_triggers[10], TimeTrigger)
        assert isinstance(test_triggers[11], AgentFiredWeaponTrigger)
        assert isinstance(test_triggers[12], WeaponFiredNearAgentTrigger)
        assert isinstance(test_triggers[13], WeaponFiredNearLocationTrigger)
        assert isinstance(test_triggers[14], AgentsSensedTrigger)
        assert isinstance(test_triggers[15], AgentsSensedInRangeTrigger)

        assert test_triggers[6]._sticky is True
        assert test_triggers[2]._sticky is False

        assert test_triggers[1]._log is False
        assert test_triggers[5]._log is False

        assert test_triggers[3]._invert is False

    def test_wrong_type_exception(self):
        """Test that an exception is raised if an incompatible type is given."""
        with self.assertRaises(TypeError) as e:
            create_triggers(1, TriggerType.COMPOUND_AND)
        self.assertEqual(
            "Selected Trigger Type does not exist.",
            str(e.exception),
        )
