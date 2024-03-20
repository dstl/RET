"""Tests for default agent behaviours."""
from __future__ import annotations

import unittest
from datetime import timedelta
from math import inf
from typing import TYPE_CHECKING

from ret.agents.affiliation import Affiliation
from ret.agents.airagent import AirAgent
from ret.agents.airdefenceagent import AirDefenceAgent
from ret.agents.infantryagent import InfantryAgent
from ret.agents.sensorfusionagent import SensorFusionAgent
from ret.agents.artilleryagent import ArtilleryAgent
from ret.agents.armouragent import ArmourAgent
from ret.behaviours import Behaviour
from ret.behaviours.deploycountermeasure import DeployCountermeasureBehaviour
from ret.behaviours.disablecommunication import DisableAllHostileCommsInRangeBehaviour
from ret.behaviours.fire import FireBehaviour
from ret.behaviours.hide import HideBehaviour
from ret.behaviours.move import AircraftMoveBehaviour, GroundBasedMoveBehaviour
from ret.behaviours.sense import SenseBehaviour
from ret.behaviours.wait import WaitBehaviour
from ret.space.culture import Culture
from ret.space.heightband import AbsoluteHeightBand
from ret.testing.mocks import MockModel3d, MockModel3dWithCulture

if TYPE_CHECKING:
    from ret.space.heightband import HeightBand


class TestAirAgentDefaultBehaviours(unittest.TestCase):
    """Tests for air agent default behaviours."""

    def setUp(self):
        """Set up test model."""
        self.model = MockModel3d()

        self.height_bands: list[HeightBand] = [
            AbsoluteHeightBand("200m", 200),
            AbsoluteHeightBand("2500m", 2500),
            AbsoluteHeightBand("10000m", 10000),
        ]

    def test_create_air_agent_no_optional_args(self):
        """Test creating an air agent without passing in any optional arguments."""
        self.blank_agent = AirAgent(
            model=self.model,
            pos=(0, 0, 0),
            name="Air Agent No Defined Behaviour",
            affiliation=Affiliation.FRIENDLY,
        )

        behaviours = self.blank_agent.behaviour_pool.expose_behaviour("__str__", Behaviour)

        behaviour_types = list((type(behaviour)) for behaviour in behaviours)

        assert len(behaviours) == 5
        assert HideBehaviour in behaviour_types
        assert AircraftMoveBehaviour in behaviour_types
        assert WaitBehaviour in behaviour_types
        assert FireBehaviour in behaviour_types
        assert DeployCountermeasureBehaviour in behaviour_types

    def test_create_air_agent_with_height_bands(self):
        """Test creating an air agent passing in a list of height bands."""
        self.height_band_agent = AirAgent(
            model=self.model,
            pos=(0, 0, 0),
            name="Air Agent with Height Bands",
            affiliation=Affiliation.FRIENDLY,
            height_bands=self.height_bands,
        )

        behaviours = self.height_band_agent.behaviour_pool.expose_behaviour("__str__", Behaviour)

        for behaviour in behaviours:
            if type(behaviour) is AircraftMoveBehaviour:
                assert behaviour.height_bands == self.height_bands

    def test_create_air_agent_with_behaviours(self):
        """Test creating an air agent passing in non-default behaviours."""
        wait_behaviour = WaitBehaviour()
        hide_behaviour = HideBehaviour()
        move_behaviour = AircraftMoveBehaviour(5, self.height_bands)
        fire_behaviour = FireBehaviour()
        deploy_countermeasures_behaviour = DeployCountermeasureBehaviour()

        behaviours = [
            wait_behaviour,
            hide_behaviour,
            move_behaviour,
            fire_behaviour,
            deploy_countermeasures_behaviour,
        ]

        self.agent_with_behaviours = AirAgent(
            model=self.model,
            pos=(0, 0, 0),
            name="Air Agent with Defined Behaviour",
            affiliation=Affiliation.FRIENDLY,
            behaviours=behaviours,
        )

        agent_behaviours = self.agent_with_behaviours.behaviour_pool.expose_behaviour(
            "__str__", Behaviour
        )

        assert len(agent_behaviours) == 5
        for behaviour in behaviours:
            assert behaviour in agent_behaviours


class TestAirDefenceAgentDefaultBehaviours(unittest.TestCase):
    """Tests for air defence agent default behaviours."""

    def setUp(self):
        """Set up test model."""
        self.culture_red = Culture("red culture", 100.0)
        self.culture_blue = Culture("blue culture")
        self.culture_dict = {
            (255, 0, 0): self.culture_red,
            (0, 0, 255): self.culture_blue,
        }
        self.model = MockModel3dWithCulture(culture_dictionary=self.culture_dict)

        self.test_culture_speed_modifiers = dict(
            (culture, 2.0) for culture in self.culture_dict.values()
        )

    def test_create_air_defence_agent_no_optional_args(self):
        """Test creating an air defence agent without optional arguments."""
        self.blank_agent = AirDefenceAgent(
            model=self.model,
            pos=(0, 0, 0),
            name="Air Defence Agent No Defined Behaviour",
            affiliation=Affiliation.FRIENDLY,
        )

        behaviours = self.blank_agent.behaviour_pool.expose_behaviour("__str__", Behaviour)

        behaviour_types = list((type(behaviour)) for behaviour in behaviours)

        assert len(behaviours) == 6
        assert GroundBasedMoveBehaviour in behaviour_types
        assert WaitBehaviour in behaviour_types
        assert HideBehaviour in behaviour_types
        assert FireBehaviour in behaviour_types
        assert SenseBehaviour in behaviour_types
        assert DisableAllHostileCommsInRangeBehaviour in behaviour_types

    def test_create_air_defence_agent_with_culture_modifiers(self):
        """Test creating an air defence agent with culture modifiers."""
        self.culture_modifiers_agent = AirDefenceAgent(
            model=self.model,
            pos=(0, 0, 0),
            name="Air Defence Agent with Culture Speed Modifiers",
            affiliation=Affiliation.FRIENDLY,
            culture_speed_modifiers=self.test_culture_speed_modifiers,
        )

        behaviours = self.culture_modifiers_agent.behaviour_pool.expose_behaviour(
            "__str__", Behaviour
        )

        for behaviour in behaviours:
            if type(behaviour) is GroundBasedMoveBehaviour:
                assert behaviour.culture_speed_modifiers == self.test_culture_speed_modifiers

    def test_create_air_defence_agent_with_behaviours(self):
        """Test creating an air defence agent passing in non-default behaviours."""
        wait_behaviour = WaitBehaviour()
        hide_behaviour = HideBehaviour()
        move_behaviour = GroundBasedMoveBehaviour(
            base_speed=0.020,
            gradient_speed_modifiers=[
                ((-inf, -1.2), 0.7),
                ((-1.2, 1.2), 1.1),
                ((1.2, inf), 0.7),
            ],
            culture_speed_modifiers=self.test_culture_speed_modifiers,
        )
        fire_behaviour = FireBehaviour()
        sense_behaviour = SenseBehaviour(
            time_before_first_sense=timedelta(seconds=0), time_between_senses=timedelta(seconds=5)
        )
        disable_comms_behaviour = DisableAllHostileCommsInRangeBehaviour(20)

        behaviours = [
            wait_behaviour,
            hide_behaviour,
            move_behaviour,
            fire_behaviour,
            sense_behaviour,
            disable_comms_behaviour,
        ]

        self.agent_with_behaviours = AirDefenceAgent(
            model=self.model,
            pos=(0, 0, 0),
            name="Air Defence Agent with Defined Behaviour",
            affiliation=Affiliation.FRIENDLY,
            behaviours=behaviours,
        )

        agent_behaviours = self.agent_with_behaviours.behaviour_pool.expose_behaviour(
            "__str__", Behaviour
        )

        assert len(agent_behaviours) == 6
        for behaviour in behaviours:
            assert behaviour in agent_behaviours


class TestInfantryAgentDefaultBehaviours(unittest.TestCase):
    """Tests for infantry agent default behaviours."""

    def setUp(self):
        """Set up test model."""
        self.culture_red = Culture("red culture", 100.0)
        self.culture_blue = Culture("blue culture")
        self.culture_dict = {
            (255, 0, 0): self.culture_red,
            (0, 0, 255): self.culture_blue,
        }
        self.model = MockModel3dWithCulture(culture_dictionary=self.culture_dict)

        self.test_culture_speed_modifiers = dict(
            (culture, 0.8) for culture in self.culture_dict.values()
        )

    def test_create_infantry_agent_no_optional_args(self):
        """Test creating an infantry agent without optional arguments."""
        self.blank_agent = InfantryAgent(
            model=self.model,
            pos=(0, 0, 0),
            name="Infantry Agent No Defined Behaviour",
            affiliation=Affiliation.FRIENDLY,
        )

        behaviours = self.blank_agent.behaviour_pool.expose_behaviour("__str__", Behaviour)

        behaviour_types = list((type(behaviour)) for behaviour in behaviours)

        assert len(behaviours) == 5
        assert GroundBasedMoveBehaviour in behaviour_types
        assert WaitBehaviour in behaviour_types
        assert HideBehaviour in behaviour_types
        assert FireBehaviour in behaviour_types
        assert SenseBehaviour in behaviour_types

    def test_create_infantry_agent_with_culture_modifiers(self):
        """Test creating an infantry agent passing culture modifiers."""
        self.culture_modifiers_agent = InfantryAgent(
            model=self.model,
            pos=(0, 0, 0),
            name="Infantry Agent with Culture Speed Modifiers",
            affiliation=Affiliation.FRIENDLY,
            culture_speed_modifiers=self.test_culture_speed_modifiers,
        )

        behaviours = self.culture_modifiers_agent.behaviour_pool.expose_behaviour(
            "__str__", Behaviour
        )

        for behaviour in behaviours:
            if type(behaviour) is GroundBasedMoveBehaviour:
                assert behaviour.culture_speed_modifiers == self.test_culture_speed_modifiers

    def test_create_armour_agent_with_culture_modifiers(self):
        """Test creating an armour agent passing culture modifiers."""
        self.culture_modifiers_agent = ArmourAgent(
            model=self.model,
            pos=(0, 0, 0),
            name="Armour Agent with Culture Speed Modifiers",
            affiliation=Affiliation.FRIENDLY,
            culture_speed_modifiers=self.test_culture_speed_modifiers,
        )

        behaviours = self.culture_modifiers_agent.behaviour_pool.expose_behaviour(
            "__str__", Behaviour
        )

        for behaviour in behaviours:
            if type(behaviour) is GroundBasedMoveBehaviour:
                assert behaviour.culture_speed_modifiers == self.test_culture_speed_modifiers

    def test_create_artillery_agent_with_culture_modifiers(self):
        """Test creating an artillery agent passing culture modifiers."""
        self.culture_modifiers_agent = ArtilleryAgent(
            model=self.model,
            pos=(0, 0, 0),
            name="Artillery Agent with Culture Speed Modifiers",
            affiliation=Affiliation.FRIENDLY,
            culture_speed_modifiers=self.test_culture_speed_modifiers,
        )

        behaviours = self.culture_modifiers_agent.behaviour_pool.expose_behaviour(
            "__str__", Behaviour
        )

        for behaviour in behaviours:
            if type(behaviour) is GroundBasedMoveBehaviour:
                assert behaviour.culture_speed_modifiers == self.test_culture_speed_modifiers

    def test_create_infantry_agent_with_behaviours(self):
        """Test creating an infantry agent passing in non-default behaviours."""
        wait_behaviour = WaitBehaviour()
        hide_behaviour = HideBehaviour()
        move_behaviour = GroundBasedMoveBehaviour(
            base_speed=0.020,
            gradient_speed_modifiers=[
                ((-inf, -1.2), 0.7),
                ((-1.2, 1.2), 1.1),
                ((1.2, inf), 0.7),
            ],
            culture_speed_modifiers=self.test_culture_speed_modifiers,
        )
        fire_behaviour = FireBehaviour()
        sense_behaviour = SenseBehaviour(
            time_before_first_sense=timedelta(seconds=0), time_between_senses=timedelta(seconds=5)
        )

        behaviours = [
            wait_behaviour,
            hide_behaviour,
            move_behaviour,
            fire_behaviour,
            sense_behaviour,
        ]

        self.agent_with_behaviours = InfantryAgent(
            model=self.model,
            pos=(0, 0, 0),
            name="Infantry Agent with Defined Behaviour",
            affiliation=Affiliation.FRIENDLY,
            behaviours=behaviours,
        )

        agent_behaviours = self.agent_with_behaviours.behaviour_pool.expose_behaviour(
            "__str__", Behaviour
        )

        assert len(agent_behaviours) == 5
        for behaviour in behaviours:
            assert behaviour in agent_behaviours


class TestSensorFusionAgentDefaultBehaviours(unittest.TestCase):
    """Tests for sensor fusion agent default behaviours."""

    def setUp(self):
        """Set up test model."""
        self.culture_red = Culture("red culture", 100.0)
        self.culture_blue = Culture("blue culture")
        self.culture_dict = {
            (255, 0, 0): self.culture_red,
            (0, 0, 255): self.culture_blue,
        }
        self.model = MockModel3dWithCulture(culture_dictionary=self.culture_dict)

        self.test_culture_speed_modifiers = dict(
            (culture, 0.9) for culture in self.culture_dict.values()
        )

    def test_create_sensor_fusion_agent_no_optional_args(self):
        """Test creating a sensor fusion agent without optional arguments."""
        self.blank_agent = SensorFusionAgent(
            model=self.model,
            pos=(0, 0, 0),
            name="Sensor Fusion Agent No Defined Behaviour",
            affiliation=Affiliation.FRIENDLY,
        )

        behaviours = self.blank_agent.behaviour_pool.expose_behaviour("__str__", Behaviour)

        behaviour_types = list((type(behaviour)) for behaviour in behaviours)

        assert len(behaviours) == 3
        assert GroundBasedMoveBehaviour in behaviour_types
        assert WaitBehaviour in behaviour_types
        assert HideBehaviour in behaviour_types

    def test_create_sensor_fusion_agent_with_culture_modifiers(self):
        """Test creating a sensor fusion agent passing culture modifiers."""
        self.culture_modifiers_agent = SensorFusionAgent(
            model=self.model,
            pos=(0, 0, 0),
            name="Sensor Fusion Agent with Culture Speed Modifiers",
            affiliation=Affiliation.FRIENDLY,
            culture_speed_modifiers=self.test_culture_speed_modifiers,
        )

        behaviours = self.culture_modifiers_agent.behaviour_pool.expose_behaviour(
            "__str__", Behaviour
        )

        for behaviour in behaviours:
            if type(behaviour) is GroundBasedMoveBehaviour:
                assert behaviour.culture_speed_modifiers == self.test_culture_speed_modifiers

    def test_create_sensor_fusion_agent_with_behaviours(self):
        """Test creating a sensor fusion agent passing in non-default behaviours."""
        wait_behaviour = WaitBehaviour()
        hide_behaviour = HideBehaviour()
        move_behaviour = GroundBasedMoveBehaviour(
            base_speed=0.10,
            gradient_speed_modifiers=[
                ((-inf, -1.2), 0.7),
                ((-1.2, 1.2), 1.1),
                ((1.2, inf), 0.7),
            ],
            culture_speed_modifiers=self.test_culture_speed_modifiers,
        )

        behaviours = [
            wait_behaviour,
            hide_behaviour,
            move_behaviour,
        ]

        self.agent_with_behaviours = SensorFusionAgent(
            model=self.model,
            pos=(0, 0, 0),
            name="Infantry Agent with Defined Behaviour",
            affiliation=Affiliation.FRIENDLY,
            behaviours=behaviours,
        )

        agent_behaviours = self.agent_with_behaviours.behaviour_pool.expose_behaviour(
            "__str__", Behaviour
        )

        assert len(agent_behaviours) == 3
        for behaviour in behaviours:
            assert behaviour in agent_behaviours
