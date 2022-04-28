"""Tests for create_multiple_agents method."""
from __future__ import annotations

import unittest
import warnings
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from mesa_ret.agents.affiliation import Affiliation
from mesa_ret.agents.agent import RetAgent
from mesa_ret.agents.agenttype import AgentType
from mesa_ret.agents.airagent import AirAgent
from mesa_ret.agents.airdefenceagent import AirDefenceAgent
from mesa_ret.agents.groupagent import GroupAgent
from mesa_ret.agents.infantryagent import InfantryAgent
from mesa_ret.agents.protectedassetagent import ProtectedAssetAgent
from mesa_ret.agents.sensorfusionagent import SensorFusionAgent
from mesa_ret.behaviours.behaviourpool import AlwaysAdder
from mesa_ret.behaviours.communicate import (
    CommunicateOrdersBehaviour,
    CommunicateWorldviewBehaviour,
)
from mesa_ret.behaviours.deploycountermeasure import DeployCountermeasureBehaviour
from mesa_ret.behaviours.disablecommunication import DisableAllHostileCommsInRangeBehaviour
from mesa_ret.behaviours.fire import FireBehaviour
from mesa_ret.behaviours.hide import HideBehaviour
from mesa_ret.behaviours.move import AircraftMoveBehaviour, GroundBasedMoveBehaviour
from mesa_ret.behaviours.sense import SenseBehaviour
from mesa_ret.behaviours.wait import WaitBehaviour
from mesa_ret.communication.communicationreceiver import (
    CommunicationReceiver,
    GroupAgentCommunicationReceiver,
)
from mesa_ret.communication.sensorfusionreceiver import SensorFusionReceiver
from mesa_ret.creator.agents import create_agents, create_list, create_list_of_lists
from mesa_ret.sensing.perceivedworld import FriendlyAgents
from mesa_ret.space.culture import Culture
from mesa_ret.space.heightband import AbsoluteHeightBand
from mesa_ret.testing.mocks import (
    MockCommunicateMissionMessageBehaviour,
    MockModel3d,
    MockModel3dWithCulture,
    MockMoveInBandBehaviour,
    MockOrder,
    MockSensor,
)
from mesa_ret.weapons.weapon import BasicWeapon
from pandas import Timedelta
from parameterized import parameterized

if TYPE_CHECKING:
    from typing import Optional

    from mesa_ret.orders.order import Order
    from mesa_ret.space.heightband import HeightBand
    from mesa_ret.template import Template
    from mesa_ret.weapons.weapon import Weapon


class TestCheckListLength(unittest.TestCase):
    """Tests for the check_list_length method."""

    def setUp(self):
        """Create variables for tests."""
        self.single = 1
        self.wrong_list = [self.single] * 5
        self.correct_list = [self.single] * 10
        self.number = 10

    def test_pass_single(self):
        """Test that you can pass a single object into the method."""
        new_list = create_list(self.number, self.single)
        assert new_list == self.correct_list

    def test_pass_correct_list(self):
        """Test that you can pass a list of length [number] into the method."""
        new_list = create_list(self.number, self.correct_list)
        assert new_list == self.correct_list

    def test_pass_wrong_list(self):
        """Test for exception if wrong length list is passed in."""
        with self.assertRaises(IndexError) as e:
            create_list(self.number, self.wrong_list)
        self.assertEqual(
            "List length incompatible with number of agents. No agents created.",
            str(e.exception),
        )


class TestCheckListOfListLength(unittest.TestCase):
    """Tests for the check_list_of_list_length method."""

    def setUp(self):
        """Create variables for tests."""
        self.single = [1]
        self.wrong_list = [self.single] * 5
        self.correct_list = [self.single] * 10
        self.number = 10

    def test_pass_single(self):
        """Test that you can pass a single list into the method."""
        new_list = create_list_of_lists(self.number, self.single)
        assert new_list == self.correct_list

    def test_pass_correct_list(self):
        """Test that you can pass a list of lists of length [number] into the method."""
        new_list = create_list_of_lists(self.number, self.correct_list)
        assert new_list == self.correct_list

    def test_pass_wrong_list(self):
        """Test for exception if wrong length list of lists is passed in."""
        with self.assertRaises(IndexError) as e:
            create_list_of_lists(self.number, self.wrong_list)
        self.assertEqual(
            "List length incompatible with number of agents. No agents created.",
            str(e.exception),
        )


class TestCreateMultipleAgents(unittest.TestCase):
    """Test for the _create_multiple_agents method."""

    def setUp(self):
        """Create the model, space, and parameters for the test."""
        self.start_time = datetime(2020, 1, 1, 0, 0)
        self.time_step = timedelta(hours=1)
        self.end_time = datetime(2020, 1, 1, 5, 0)

        self.model = MockModel3d(self.start_time, self.time_step, self.end_time)

        self.pos = [
            (0, 1, 0),
            (0, 1, 0),
            None,
            (0, 1, 0),
            (0, 1, 0),
            (0, 1, 0),
            (0, 1, 0),
            (0, 1, 0),
        ]
        self.names = [
            "Air",
            "Air Defence",
            "Group",
            "Infantry",
            "Protected Asset",
            "Sensor Fusion",
            "Other",
            "Mechanised Infantry",
        ]
        self.affiliations = [
            Affiliation.FRIENDLY,
            Affiliation.HOSTILE,
            Affiliation.NEUTRAL,
            Affiliation.UNKNOWN,
            Affiliation.FRIENDLY,
            Affiliation.HOSTILE,
            Affiliation.NEUTRAL,
            Affiliation.UNKNOWN,
        ]
        self.agent_types = [
            AgentType.AIR,
            AgentType.AIR_DEFENCE,
            AgentType.GROUP,
            AgentType.INFANTRY,
            AgentType.PROTECTED_ASSET,
            AgentType.SENSOR_FUSION,
            AgentType.ARMOUR,
            AgentType.MECHANISED_INFANTRY,
        ]
        self.critical_dimensions = [2.0] * 8
        self.reflectivities = [0.1, 0.1, None, 0.1, 0.1, 0.1, 0.1, 0.1]
        self.temperatures = [20, 20, None, 20, 20, 20, 20, 20]
        self.temperature_std_devs = [0.123, 0.123, None, 0.123, 0.123, 0.123, 0.123, 0.123]
        self.height_bands: list[Optional[list[HeightBand]]] = [
            [AbsoluteHeightBand("50m", 50)],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ]
        self.weapons: list[Optional[list[Weapon]]] = [
            [
                BasicWeapon(
                    name="AirWeapon",
                    radius=1.0,
                    time_before_first_shot=timedelta(seconds=0),
                    time_between_rounds=timedelta(seconds=30),
                    kill_probability_per_round=1,
                )
            ],
            [
                BasicWeapon(
                    name="AirDefenceWeapon",
                    radius=1.0,
                    time_before_first_shot=timedelta(seconds=0),
                    time_between_rounds=timedelta(seconds=30),
                    kill_probability_per_round=1,
                )
            ],
            None,
            [
                BasicWeapon(
                    name="InfantryWeapon",
                    radius=1.0,
                    time_before_first_shot=timedelta(seconds=0),
                    time_between_rounds=timedelta(seconds=30),
                    kill_probability_per_round=1,
                )
            ],
            None,
            None,
            [
                BasicWeapon(
                    name="ArmourWeapon",
                    radius=1.0,
                    time_before_first_shot=timedelta(seconds=0),
                    time_between_rounds=timedelta(seconds=30),
                    kill_probability_per_round=1,
                )
            ],
            None,
        ]
        self.wait_behaviours = [WaitBehaviour()] * 8
        self.fire_behaviours = [FireBehaviour()] * 8
        self.hide_behaviours = [HideBehaviour()] * 8
        self.move_behaviours = [GroundBasedMoveBehaviour(0.1, [])] * 8
        self.communicate_worldview_behaviour = [CommunicateWorldviewBehaviour()] * 8
        self.sense_behaviour = [SenseBehaviour(Timedelta(0), Timedelta(0.1))] * 8
        self.deploy_countermeasure_behaviour = [DeployCountermeasureBehaviour()] * 8
        self.disable_communication_behaviour = [DisableAllHostileCommsInRangeBehaviour(0.1)] * 8
        self.communicate_orders_behaviour = [CommunicateOrdersBehaviour()] * 8
        self.sensors = [
            [MockSensor()],
            [MockSensor()],
            None,
            [MockSensor()],
            None,
            None,
            [MockSensor()],
            [MockSensor()],
        ]
        self.communication_receivers = [
            GroupAgentCommunicationReceiver(),
            GroupAgentCommunicationReceiver(),
            None,
            GroupAgentCommunicationReceiver(),
            None,
            None,
            GroupAgentCommunicationReceiver(),
            GroupAgentCommunicationReceiver(),
        ]
        self.communicate_mission_message_behaviours = [MockCommunicateMissionMessageBehaviour()] * 8
        self.refresh_techniques = [FriendlyAgents()] * 8
        self.behaviour_adders = [AlwaysAdder] * 8

    def test_create_multiple_agents(self):
        """Test that all agents are created when using mixed types."""
        new_agents = create_agents(
            number=8,
            model=self.model,
            pos=self.pos,
            name=self.names,
            affiliation=self.affiliations,
            agent_type=self.agent_types,
        )
        assert len(new_agents) == 8
        air_agent = new_agents[0]
        air_defence_agent = new_agents[1]
        group_agent = new_agents[2]
        infantry_agent = new_agents[3]
        protected_asset_agent = new_agents[4]
        sensor_fusion_agent = new_agents[5]
        armour_agent = new_agents[6]
        mechanised_agent = new_agents[7]

        assert isinstance(air_agent, AirAgent)
        assert isinstance(air_defence_agent, AirDefenceAgent)
        assert isinstance(group_agent, GroupAgent)
        assert isinstance(infantry_agent, InfantryAgent)
        assert isinstance(protected_asset_agent, ProtectedAssetAgent)
        assert isinstance(sensor_fusion_agent, SensorFusionAgent)
        assert type(armour_agent) == RetAgent
        assert type(mechanised_agent) == RetAgent

        for (i, agent) in enumerate(new_agents):
            assert agent.name == self.names[i]
            assert agent.affiliation == self.affiliations[i]
            if isinstance(agent, GroupAgent):
                assert agent.pos is None
            else:
                assert agent.pos == (0, 1, 0)

    def test_create_multiple_agents_height_bands(self):
        """Test that all agents are created when using mixed types."""
        new_agents = create_agents(
            number=8,
            model=self.model,
            pos=self.pos,
            name=self.names,
            affiliation=self.affiliations,
            agent_type=self.agent_types,
            height_bands=self.height_bands,
        )
        air_agent = new_agents[0]

        air_agent_move_behaviour = air_agent.behaviour_pool.choose_behaviour(
            air_agent.behaviour_handlers.move_handler, air_agent.behaviour_handlers.move_type
        )
        assert isinstance(air_agent_move_behaviour, AircraftMoveBehaviour)
        assert air_agent_move_behaviour.height_bands == self.height_bands[0]

    @parameterized.expand(
        [
            [AgentType.AIR_DEFENCE],
            [AgentType.PROTECTED_ASSET],
            [AgentType.INFANTRY],
            [AgentType.SENSOR_FUSION],
        ]
    )
    def test_create_multiple_agents_height_bands_warning(self, agent_type: AgentType):
        """Test that passing height bands throws warnings for non air agents.

        Args:
            agent_type (AgentType): The type of agent being tested.
        """
        with warnings.catch_warnings(record=True) as w:
            create_agents(
                number=8,
                model=self.model,
                pos=(0, 1, 0),
                name=self.names,
                affiliation=self.affiliations,
                agent_type=agent_type,
                height_bands=[[AbsoluteHeightBand("50m", 50)]] * 8,  # type: ignore
            )
        assert "'height_bands' has been defined" in str(w[0].message)

    def test_create_multiple_agents_weapons(self):
        """Test that all agents are created with correct weapons."""
        new_agents = create_agents(
            number=8,
            model=self.model,
            pos=self.pos,
            name=self.names,
            affiliation=self.affiliations,
            agent_type=self.agent_types,
            weapons=self.weapons,
        )
        air_agent = new_agents[0]
        air_defence_agent = new_agents[1]
        group_agent = new_agents[2]
        infantry_agent = new_agents[3]
        protected_asset_agent = new_agents[4]
        sensor_fusion_agent = new_agents[5]
        armour_agent = new_agents[6]
        mechanised_agent = new_agents[7]

        assert air_agent.weapons[0].name == "AirWeapon"
        assert air_defence_agent.weapons[0].name == "AirDefenceWeapon"
        assert group_agent.weapons == []
        assert infantry_agent.weapons[0].name == "InfantryWeapon"
        assert protected_asset_agent.weapons == []
        assert sensor_fusion_agent.weapons == []
        assert armour_agent.weapons[0].name == "ArmourWeapon"
        assert mechanised_agent.weapons == []

    def test_create_multiple_agents_reflectivity(self):
        """Test that all agents are created with correct reflectivity."""
        new_agents = create_agents(
            number=8,
            model=self.model,
            pos=self.pos,
            name=self.names,
            affiliation=self.affiliations,
            agent_type=self.agent_types,
            reflectivity=self.reflectivities,
        )

        for agent in new_agents:
            if isinstance(agent, GroupAgent):
                continue
            else:
                assert agent.reflectivity == 0.1

    def test_create_multiple_agents_temperature(self):
        """Test that all agents are created with correct temperature."""
        new_agents = create_agents(
            number=8,
            model=self.model,
            pos=self.pos,
            name=self.names,
            affiliation=self.affiliations,
            agent_type=self.agent_types,
            temperature=self.temperatures,
            temperature_std_dev=self.temperature_std_devs,
        )

        for agent in new_agents:
            if isinstance(agent, GroupAgent):
                continue
            else:
                assert agent.temperature == 20.0

    def test_create_multiple_agents_critical_dimension(self):
        """Test that all agents are created with correct critical dimension."""
        new_agents = create_agents(
            number=8,
            model=self.model,
            pos=self.pos,
            name=self.names,
            affiliation=self.affiliations,
            agent_type=self.agent_types,
            critical_dimension=self.critical_dimensions,
        )

        for agent in new_agents:
            assert agent.critical_dimension == 2.0

    def test_create_multiple_agents_sensors(self):
        """Test that all agents are created with correct sensors."""
        new_agents = create_agents(
            number=8,
            model=self.model,
            pos=self.pos,
            name=self.names,
            affiliation=self.affiliations,
            agent_type=self.agent_types,
            sensors=self.sensors,
        )
        air_agent = new_agents[0]
        air_defence_agent = new_agents[1]
        group_agent = new_agents[2]
        infantry_agent = new_agents[3]
        protected_asset_agent = new_agents[4]
        sensor_fusion_agent = new_agents[5]
        armour_agent = new_agents[6]
        mechanised_agent = new_agents[7]

        sensing_agents = [
            air_agent,
            air_defence_agent,
            infantry_agent,
            mechanised_agent,
            armour_agent,
        ]
        non_sensing_agents = [group_agent, protected_asset_agent, sensor_fusion_agent]

        for agent in sensing_agents:
            assert type(agent._sensors[0]) == MockSensor

        for agent in non_sensing_agents:
            assert len(agent._sensors) == 0

    def test_create_multiple_agents_communication_receivers(self):
        """Test that all agents are created with correct communication receivers."""
        new_agents = create_agents(
            number=8,
            model=self.model,
            pos=self.pos,
            name=self.names,
            affiliation=self.affiliations,
            agent_type=self.agent_types,
            communication_receiver=self.communication_receivers,
        )
        air_agent = new_agents[0]
        air_defence_agent = new_agents[1]
        group_agent = new_agents[2]
        infantry_agent = new_agents[3]
        protected_asset_agent = new_agents[4]
        sensor_fusion_agent = new_agents[5]
        armour_agent = new_agents[6]
        mechanised_agent = new_agents[7]

        communication_receiver_agents = [
            air_agent,
            air_defence_agent,
            infantry_agent,
            mechanised_agent,
            armour_agent,
            group_agent,
        ]

        for agent in communication_receiver_agents:
            assert isinstance(agent.communication_network.receiver, GroupAgentCommunicationReceiver)

        assert isinstance(
            protected_asset_agent.communication_network.receiver, CommunicationReceiver
        )
        assert isinstance(sensor_fusion_agent.communication_network.receiver, SensorFusionReceiver)

    def test_create_multiple_agents_icon_paths(self):
        """Test that all agents are created with correct icon paths."""
        icon_dir = "./mesa_ret/mesa_ret/agents/icons/airAgent/airagent_friendly.svg"
        new_agents = create_agents(
            number=8,
            model=self.model,
            pos=self.pos,
            name=self.names,
            affiliation=self.affiliations,
            agent_type=self.agent_types,
            icon_path=icon_dir,
            killed_icon_path=icon_dir,
        )
        for agent in new_agents:
            assert agent.icon_path == icon_dir
            assert agent._killed_icon_filepath == icon_dir

    def test_create_multiple_agent_same_type(self):
        """Test that all agents are created when using the same type."""
        new_agents = create_agents(
            number=3,
            model=self.model,
            pos=(0, 1, 0),
            name="Air",
            affiliation=Affiliation.FRIENDLY,
            agent_type=AgentType.AIR,
            critical_dimension=2.0,
            reflectivity=0.2,
            temperature=20.0,
            height_bands=self.height_bands[0],
        )
        assert len(new_agents) == 3
        for air_agent in new_agents:
            assert isinstance(air_agent, AirAgent)
            air_agent_move_behaviour = air_agent.behaviour_pool.choose_behaviour(
                air_agent.behaviour_handlers.move_handler, air_agent.behaviour_handlers.move_type
            )
            assert isinstance(air_agent_move_behaviour, AircraftMoveBehaviour)
            assert air_agent_move_behaviour.height_bands == self.height_bands[0]

    def test_air_agent_default_move_behaviour_without_height_bands(self):
        """Test the default move behaviour of air agents is added with default height bands."""
        new_agents = create_agents(
            number=3,
            model=self.model,
            pos=(0, 1, 0),
            name="Air",
            affiliation=Affiliation.FRIENDLY,
            agent_type=AgentType.AIR,
            critical_dimension=2.0,
            reflectivity=0.2,
            temperature=20.0,
        )
        assert len(new_agents) == 3
        expected_height_bands = [
            AbsoluteHeightBand("500m", 500),
            AbsoluteHeightBand("2000m", 2000),
            AbsoluteHeightBand("12000m", 12000),
        ]
        for air_agent in new_agents:
            assert isinstance(air_agent, AirAgent)
            air_agent_move_behaviour = air_agent.behaviour_pool.choose_behaviour(
                air_agent.behaviour_handlers.move_handler, air_agent.behaviour_handlers.move_type
            )
            assert isinstance(air_agent_move_behaviour, AircraftMoveBehaviour)
            for expected, actual in zip(
                expected_height_bands, air_agent_move_behaviour.height_bands
            ):
                assert expected.name == actual.name
                assert expected.height == actual.height

    def test_air_agent_move_behaviour_with_height_bands_argument(self):
        """Test the correct move behaviour is added to air agents."""
        height_bands: list[HeightBand] = [
            AbsoluteHeightBand("60m", 60),
            AbsoluteHeightBand("80m", 80),
        ]
        move_behaviour = [
            MockMoveInBandBehaviour(height_bands=height_bands),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ]

        new_agents = create_agents(
            number=8,
            model=self.model,
            pos=self.pos,
            name=self.names,
            affiliation=self.affiliations,
            agent_type=self.agent_types,
            critical_dimension=self.critical_dimensions,
            reflectivity=self.reflectivities,
            temperature=self.temperatures,
            height_bands=self.height_bands,
            move_behaviour=move_behaviour,
        )
        air_agent = new_agents[0]
        assert isinstance(air_agent, AirAgent)
        assert type(new_agents[1] == AirDefenceAgent)
        assert type(new_agents[2] == GroupAgent)
        assert type(new_agents[3] == InfantryAgent)
        assert type(new_agents[4] == ProtectedAssetAgent)
        assert type(new_agents[5] == SensorFusionAgent)
        assert type(new_agents[6] == RetAgent)
        assert type(new_agents[7] == RetAgent)

        air_agent_move_behaviour = air_agent.behaviour_pool.choose_behaviour(
            air_agent.behaviour_handlers.move_handler, air_agent.behaviour_handlers.move_type
        )
        assert isinstance(air_agent_move_behaviour, MockMoveInBandBehaviour)
        # self.height_bands is ignored as a move behaviour was provided
        assert air_agent_move_behaviour.height_bands == height_bands

    def test_air_agent_move_behaviour_without_height_bands_argument(self):
        """Test the correct move behaviour is added to air agents."""
        height_bands: list[HeightBand] = [
            AbsoluteHeightBand("60m", 60),
            AbsoluteHeightBand("80m", 80),
        ]
        move_behaviour = [
            MockMoveInBandBehaviour(height_bands=height_bands),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ]

        new_agents = create_agents(
            number=8,
            model=self.model,
            pos=self.pos,
            name=self.names,
            affiliation=self.affiliations,
            agent_type=self.agent_types,
            critical_dimension=self.critical_dimensions,
            reflectivity=self.reflectivities,
            temperature=self.temperatures,
            move_behaviour=move_behaviour,
        )
        air_agent = new_agents[0]
        assert type(air_agent) == AirAgent
        assert type(new_agents[1]) == AirDefenceAgent
        assert type(new_agents[2]) == GroupAgent
        assert type(new_agents[3]) == InfantryAgent
        assert type(new_agents[4]) == ProtectedAssetAgent
        assert type(new_agents[5]) == SensorFusionAgent
        assert type(new_agents[6]) == RetAgent
        assert type(new_agents[7]) == RetAgent

        air_agent_move_behaviour = air_agent.behaviour_pool.choose_behaviour(
            air_agent.behaviour_handlers.move_handler, air_agent.behaviour_handlers.move_type
        )
        assert isinstance(air_agent_move_behaviour, MockMoveInBandBehaviour)
        assert air_agent_move_behaviour.height_bands == height_bands

    @parameterized.expand(
        [
            [AgentType.AIR],
            [AgentType.AIR_DEFENCE],
            [AgentType.ARMOUR],
            [AgentType.INFANTRY],
            [AgentType.SENSOR_FUSION],
            [AgentType.MECHANISED_INFANTRY],
        ]
    )
    def test_agents_with_refresh_techniques(self, agent_type: AgentType):
        """Test that all agents initialised with correct refresh techniques.

        Args:
            agent_type (AgentType): The type of agent being tested.
        """
        new_agents = create_agents(
            8,
            self.model,
            (0, 1, 0),
            self.names,
            self.affiliations,
            agent_type,
            refresh_technique=self.refresh_techniques,
        )
        for agent in new_agents:
            assert isinstance(agent.perceived_world._refresh_technique, FriendlyAgents)

    def test_group_agents_with_refresh_techniques_warning(self):
        """Test that group agents throw a warning with refresh techniques."""
        with warnings.catch_warnings(record=True) as w:
            create_agents(
                8,
                self.model,
                None,
                self.names,
                self.affiliations,
                AgentType.GROUP,
                refresh_technique=self.refresh_techniques,
            )
        assert "'refresh_technique' has been defined" in str(w[0].message)

    def test_protected_asset_agents_with_refresh_techniques_warning(self):
        """Test that protected asset agents throw a warning with refresh techniques."""
        with warnings.catch_warnings(record=True) as w:
            create_agents(
                8,
                self.model,
                (0, 1, 0),
                self.names,
                self.affiliations,
                AgentType.PROTECTED_ASSET,
                refresh_technique=self.refresh_techniques,
            )
        assert "'refresh_technique' has been defined" in str(w[0].message)

    @parameterized.expand(
        [
            [AgentType.AIR],
            [AgentType.AIR_DEFENCE],
            [AgentType.ARMOUR],
            [AgentType.INFANTRY],
            [AgentType.SENSOR_FUSION],
            [AgentType.MECHANISED_INFANTRY],
        ]
    )
    def test_agents_with_behaviour_adder(self, agent_type: AgentType):
        """Test that all agents initialised with correct behaviour adder.

        Args:
            agent_type (AgentType): The type of agent being tested.
        """
        new_agents = create_agents(
            8,
            self.model,
            (0, 1, 0),
            self.names,
            self.affiliations,
            agent_type,
            behaviour_adder=self.behaviour_adders,
        )
        for agent in new_agents:
            assert isinstance(agent.behaviour_pool._adder, AlwaysAdder)

    def test_group_agents_with_behaviour_adder(self):
        """Test that all group agents initialised with correct behaviour adder."""
        new_agents = create_agents(
            8,
            self.model,
            None,
            self.names,
            self.affiliations,
            AgentType.GROUP,
            behaviour_adder=self.behaviour_adders,
        )
        for agent in new_agents:
            assert isinstance(agent.behaviour_pool._adder, AlwaysAdder)

    def test_protected_asset_agents_with_behaviour_adder_warning(self):
        """Test that all protected asset agents throw a warning with behaviour adder."""
        with warnings.catch_warnings(record=True) as w:
            create_agents(
                8,
                self.model,
                (0, 1, 0),
                self.names,
                self.affiliations,
                AgentType.PROTECTED_ASSET,
                behaviour_adder=self.behaviour_adders,
            )
        assert "'behaviour_adder' has been defined" in str(w[0].message)

    @parameterized.expand(
        [
            [AgentType.AIR_DEFENCE],
            [AgentType.INFANTRY],
            [AgentType.SENSOR_FUSION],
        ]
    )
    def test_agents_with_culture_speed_modifier(self, agent_type: AgentType):
        """Test that all agents are initialised with the correct culture speed modifier.

        Args:
            agent_type (AgentType): The type of agent being tested.
        """
        test_culture = Culture("Test culture")
        color = (1, 2, 3)
        culture_dict = {color: test_culture}
        test_modifier = {test_culture: 5.0}
        new_agents = create_agents(
            8,
            MockModel3dWithCulture(culture_dict),
            (0, 1, 0),
            self.names,
            self.affiliations,
            agent_type,
            culture_speed_modifiers=test_modifier,
        )
        for agent in new_agents:
            move_behaviour = agent.behaviour_pool.choose_behaviour(
                agent.behaviour_handlers.move_handler, agent.behaviour_handlers.move_type
            )
            assert move_behaviour.culture_speed_modifiers == test_modifier  # type: ignore

    @parameterized.expand(
        [
            [AgentType.AIR],
            [AgentType.PROTECTED_ASSET],
            [AgentType.ARMOUR],
            [AgentType.MECHANISED_INFANTRY],
        ]
    )
    def test_agents_with_culture_speed_modifier_warning(self, agent_type: AgentType):
        """Test that a warning is thrown when culture_speed_modifier is passed to an air agent.

        Args:
            agent_type (AgentType): The type of agent being tested.
        """
        test_culture = Culture("Test culture")
        color = (1, 2, 3)
        culture_dict = {color: test_culture}
        test_modifier = {test_culture: 5.0}

        with warnings.catch_warnings(record=True) as w:
            create_agents(
                8,
                MockModel3dWithCulture(culture_dict),
                (0, 1, 0),
                self.names,
                self.affiliations,
                agent_type,
                culture_speed_modifiers=test_modifier,
            )
        assert "'culture_speed_modifiers' has been defined" in str(w[0].message)

    def test_group_agents_with_culture_speed_modifier_warning(self):
        """Test that a warning is thrown when culture_speed_modifier is passed to a group agent."""
        test_culture = Culture("Test culture")
        color = (1, 2, 3)
        culture_dict = {color: test_culture}
        test_modifier = {test_culture: 5.0}

        with warnings.catch_warnings(record=True) as w:
            create_agents(
                8,
                MockModel3dWithCulture(culture_dict),
                None,
                self.names,
                self.affiliations,
                AgentType.GROUP,
                culture_speed_modifiers=test_modifier,
            )
        assert "'culture_speed_modifiers' has been defined" in str(w[0].message)

    @parameterized.expand(
        [
            [AgentType.AIR],
            [AgentType.AIR_DEFENCE],
            [AgentType.ARMOUR],
            [AgentType.INFANTRY],
            [AgentType.SENSOR_FUSION],
            [AgentType.MECHANISED_INFANTRY],
        ]
    )
    def test_agents_with_behaviours(self, agent_type: AgentType):
        """Test that all agents initialised with correct behaviours.

        Args:
            agent_type (AgentType): The type of agent being tested.
        """
        new_agents = create_agents(
            8,
            self.model,
            (0, 1, 0),
            self.names,
            self.affiliations,
            agent_type,
            wait_behaviour=self.wait_behaviours,
            fire_behaviour=self.fire_behaviours,
            hide_behaviour=self.hide_behaviours,
            move_behaviour=self.move_behaviours,
            communicate_worldview_behaviour=self.communicate_worldview_behaviour,
            sense_behaviour=self.sense_behaviour,
            deploy_countermeasure_behaviour=self.deploy_countermeasure_behaviour,
            disable_communication_behaviour=self.disable_communication_behaviour,
            communicate_orders_behaviour=self.communicate_orders_behaviour,
            communicate_mission_message_behaviour=self.communicate_mission_message_behaviours,
        )
        for agent in new_agents:
            wait_behaviour = agent.behaviour_pool.choose_behaviour(
                agent.behaviour_handlers.wait_handler, agent.behaviour_handlers.wait_type
            )
            assert isinstance(wait_behaviour, WaitBehaviour)
            fire_behaviour = agent.behaviour_pool.choose_behaviour(
                agent.behaviour_handlers.fire_handler, agent.behaviour_handlers.fire_type
            )
            assert isinstance(fire_behaviour, FireBehaviour)
            hide_behaviour = agent.behaviour_pool.choose_behaviour(
                agent.behaviour_handlers.hide_handler, agent.behaviour_handlers.hide_type
            )
            assert isinstance(hide_behaviour, HideBehaviour)
            move_behaviour = agent.behaviour_pool.choose_behaviour(
                agent.behaviour_handlers.move_handler, agent.behaviour_handlers.move_type
            )
            assert type(move_behaviour) == GroundBasedMoveBehaviour
            communicate_worldview_behaviour = agent.behaviour_pool.choose_behaviour(
                agent.behaviour_handlers.communicate_worldview_handler,
                agent.behaviour_handlers.communicate_worldview_type,
            )
            assert isinstance(communicate_worldview_behaviour, CommunicateWorldviewBehaviour)
            sense_behaviour = agent.behaviour_pool.choose_behaviour(
                agent.behaviour_handlers.sense_handler,
                agent.behaviour_handlers.sense_type,
            )
            assert isinstance(sense_behaviour, SenseBehaviour)
            deploy_countermeasure_behaviour = agent.behaviour_pool.choose_behaviour(
                agent.behaviour_handlers.deploy_countermeasure_handler,
                agent.behaviour_handlers.deploy_countermeasure_type,
            )
            assert isinstance(deploy_countermeasure_behaviour, DeployCountermeasureBehaviour)
            disable_communication_behaviour = agent.behaviour_pool.choose_behaviour(
                agent.behaviour_handlers.disable_communication_handler,
                agent.behaviour_handlers.disable_communication_type,
            )
            assert type(disable_communication_behaviour) == DisableAllHostileCommsInRangeBehaviour
            communicate_orders_behaviour = agent.behaviour_pool.choose_behaviour(
                agent.behaviour_handlers.communicate_orders_handler,
                agent.behaviour_handlers.communicate_orders_type,
            )
            assert isinstance(communicate_orders_behaviour, CommunicateOrdersBehaviour)
            communicate_mission_message_behaviour = agent.behaviour_pool.choose_behaviour(
                agent.behaviour_handlers.communicate_mission_message_handler,
                agent.behaviour_handlers.communicate_mission_message_type,
            )
            assert isinstance(
                communicate_mission_message_behaviour, MockCommunicateMissionMessageBehaviour
            )

    @parameterized.expand(
        [
            [AgentType.AIR],
            [AgentType.AIR_DEFENCE],
            [AgentType.ARMOUR],
            [AgentType.INFANTRY],
            [AgentType.MECHANISED_INFANTRY],
        ]
    )
    def test_pass_order_list_with_none(self, agent_type: AgentType):
        """Test that all agents initialised with correct orders.

        Args:
            agent_type (AgentType): The type of agent being tested.
        """
        order_for_agent: list[Template[Order]] = [MockOrder()]

        order_list = [None, None, None, order_for_agent, None, None, order_for_agent, None]

        new_agents = create_agents(
            8,
            self.model,
            (0, 1, 0),
            self.names,
            self.affiliations,
            agent_type,
            orders=order_list,
        )

        assert len(new_agents[0]._orders) == 0
        assert len(new_agents[1]._orders) == 0
        assert len(new_agents[2]._orders) == 0
        assert len(new_agents[3]._orders) == 1
        assert len(new_agents[4]._orders) == 0
        assert len(new_agents[5]._orders) == 0
        assert len(new_agents[6]._orders) == 1
        assert len(new_agents[7]._orders) == 0

    def test_pass_order_list_with_none_group(self):
        """Test that all group agents initialised with correct orders."""
        order_for_agent: list[Template[Order]] = [MockOrder()]

        order_list = [None, None, None, order_for_agent, None, None, order_for_agent, None]

        new_agents = create_agents(
            8,
            self.model,
            None,
            self.names,
            self.affiliations,
            AgentType.GROUP,
            orders=order_list,
        )

        assert len(new_agents[0]._orders) == 0
        assert len(new_agents[1]._orders) == 0
        assert len(new_agents[2]._orders) == 0
        assert len(new_agents[3]._orders) == 1
        assert len(new_agents[4]._orders) == 0
        assert len(new_agents[5]._orders) == 0
        assert len(new_agents[6]._orders) == 1
        assert len(new_agents[7]._orders) == 0

    def test_create_order_list_of_lists_exceptions(self):
        """Test that the correct exceptions are raised."""
        order_for_agent = MockOrder()
        order_list = [
            None,
            None,
            None,
            [order_for_agent, order_for_agent],
            None,
            None,
            order_for_agent,
        ]

        with self.assertRaises(TypeError) as e:
            create_list_of_lists(7, order_for_agent)
        self.assertEqual(
            "Parameter must be None, a list of single objects, or a list of lists.",
            str(e.exception),
        )

        with self.assertRaises(TypeError) as e:
            create_list_of_lists(7, order_list)
        self.assertEqual("Cannot have lists and single objects in the same list.", str(e.exception))
