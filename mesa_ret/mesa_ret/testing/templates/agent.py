"""Basic mesa RetAgent test templates."""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from mesa_ret.agents.affiliation import Affiliation
from mesa_ret.agents.agent import RetAgent
from mesa_ret.behaviours.sense import SenseSchedule
from mesa_ret.communication.communicationreceiver import CommunicationReceiver
from mesa_ret.orders.order import Order
from mesa_ret.orders.tasks.move import MoveTask
from mesa_ret.orders.tasks.wait import WaitTask
from mesa_ret.orders.triggers.time import TimeTrigger
from mesa_ret.sensing.perceivedworld import RemoveDuplicates
from mesa_ret.testing.mocks import (
    MockCommunicateMissionMessageBehaviour,
    MockCommunicateOrdersBehaviour,
    MockCommunicateWorldviewBehaviour,
    MockCommunicationNetwork,
    MockCountermeasure,
    MockDeployCountermeasureBehaviour,
    MockDisableCommunicationBehaviour,
    MockFireBehaviour,
    MockHideBehaviour,
    MockModel2d,
    MockMoveBehaviour,
    MockSenseBehaviour,
    MockSensor,
    MockWaitBehaviour,
    MockWeapon,
)

if TYPE_CHECKING:
    from typing import Optional

    from mesa_ret.sensing.perceivedworld import PerceivedAgentFilter
    from mesa_ret.sensing.sensor import Sensor
    from mesa_ret.space.clutter.countermeasure import Countermeasure
    from mesa_ret.template import Template
    from mesa_ret.types import Coordinate2d


class RetAgentNoBehavioursTemplate(ABC):
    """Template test for RetAgent with no behaviours.

    In order to run this test case for a bespoke Agent type, create a new
    test class which inherits from this class, and unittest.TestCase, and
    implement a `create_agent` method which returns the agent you wish to
    test.

    You can add any additional agent-specific test cases to the model should
    you wish.
    """

    pos: Coordinate2d
    name: str
    affiliation: Affiliation
    orders: Optional[list[Template[Order]]]
    move_behaviour: MockMoveBehaviour
    wait_behaviour: MockWaitBehaviour
    hide_behaviour: MockHideBehaviour
    sense_behaviour: MockSenseBehaviour
    sensors: Optional[list[Template[Sensor]]]
    communicate_orders_behaviour: MockCommunicateOrdersBehaviour
    communicate_worldview_behaviour: MockCommunicateWorldviewBehaviour
    communicate_mission_message_behaviour: MockCommunicateMissionMessageBehaviour
    fire_behaviour: MockFireBehaviour
    disable_communication_behaviour: MockDisableCommunicationBehaviour
    communication_receiver: Optional[CommunicationReceiver]
    refresh_technique: Optional[PerceivedAgentFilter]
    deploy_countermeasure_behaviour: MockDeployCountermeasureBehaviour
    countermeasures: Optional[list[Template[Countermeasure]]]

    def setUp(self):
        """Set up test cases."""
        self.define_variables()
        self.model = MockModel2d(
            start_time=datetime(2020, 1, 1, 0, 0),
            time_step=timedelta(minutes=1),
            end_time=datetime(2020, 1, 2, 0, 0),
        )
        self.agent = self.create_agent()

    def define_variables(self) -> None:
        """Define variables."""
        self.pos: Coordinate2d = (0, 0)
        self.name: str = "Agent"
        self.affiliation: Affiliation = Affiliation.FRIENDLY

        time_0 = datetime(2020, 1, 1, 0, 0)
        time_1 = time_0 + timedelta(minutes=1)
        time_2 = time_1 + timedelta(minutes=1)

        self.orders = [
            Order(TimeTrigger(time_0), MoveTask((0, 100), 1), priority=0, persistent=True),
            Order(TimeTrigger(time_1), MoveTask((0, 200), 1), priority=1),
            Order(TimeTrigger(time_2), MoveTask((0, 100), 1), priority=0),
        ]

    @abstractmethod
    def create_agent(self) -> RetAgent:  # pragma: no cover
        """Create agent to be tested.

        To be overridden by extending classes.

        Returns:
            RetAgent: Agent under test
        """
        pass

    def test_model_step(self):
        """Test that stepping model containing agent raises warning."""
        with warnings.catch_warnings(record=True) as w:
            self.model.step()
            assert len(w) == 1
            assert "Move behaviour requested for agent" in str(w[0].message)

    def test_agent_step(self):
        """Test that stepping agent with no behaviour raises warning."""
        with warnings.catch_warnings(record=True) as w:
            self.agent.step()
            assert len(w) == 1
            assert "Move behaviour requested for agent" in str(w[0].message)

    def test_check_for_new_active_order(self):
        """Test that agent with no behaviour can check for active orders."""
        assert self.agent.active_order is None
        assert len(self.agent._orders) == 3
        self.agent.check_for_new_active_order()
        assert self.agent.active_order._trigger._time == self.orders[0]._trigger._time
        assert len(self.agent._orders) == 3

    def test_icon_set_correctly(self):
        """Test that icon can be located."""
        with open(self.agent.icon_dict[Affiliation.FRIENDLY]) as i:
            expected = i.read()
        assert self.agent.icon == expected

    def test_kill(self):
        """Test killing the agent."""
        self.agent.kill()
        self.assertTrue(self.agent.killed)

    def test_move_step(self):
        """Test that a move step where no behaviour is defined raises warning."""
        with warnings.catch_warnings(record=True) as w:
            self.agent.move_step((0, 100))
            assert len(w) == 1
            assert "Move behaviour requested for agent" in str(w[0].message)

        assert self.agent.pos[0] == 0
        assert self.agent.pos[1] == 0

    def test_wait_step(self):
        """Test that a wait step where no behaviour is defined raises warning."""
        with warnings.catch_warnings(record=True) as w:
            self.agent.wait_step()
            assert len(w) == 1
            assert "Wait behaviour requested for agent" in str(w[0].message)

    def test_sense_step(self):
        """Test that a sense step where no behaviour is defined raises warning."""
        with warnings.catch_warnings(record=True) as w:
            self.agent.sense_step(sense_direction=90)
            assert len(w) == 1
            assert "Sense behaviour requested for agent" in str(w[0].message)

    def test_hide_step(self):
        """Test that a hide step where no behaviour is defined raises warning."""
        with warnings.catch_warnings(record=True) as w:
            self.agent.hide_step()
            assert len(w) == 1
            assert "Hide behaviour requested for agent" in str(w[0].message)

    def test_communicate_worldview_step(self):
        """Test that a communicate step where no behaviour is defined raises warning."""
        with warnings.catch_warnings(record=True) as w:
            self.agent.communicate_worldview_step(worldview_filter=None)
            assert len(w) == 1
            assert "Communicate Worldview behaviour requested for agent" in str(w[0].message)

        assert len(self.agent.perceived_world.get_perceived_agents()) == 1

    def test_communicate_orders_step(self):
        """Test that a communicate step where no behaviour is defined raises warning."""

        def num_orders() -> int:
            """Calculate number of orders, accounting for empty list being None.

            Returns:
                int: Number of orders
            """
            if not self.agent._orders:
                return 0
            else:
                return len(self.agent._orders)

        number_of_orders = num_orders()

        with warnings.catch_warnings(record=True) as w:
            self.agent.communicate_orders_step(orders=[])
            assert len(w) == 1
            assert "Communicate Orders behaviour requested for agent" in str(w[0].message)

        assert number_of_orders == num_orders()

    def test_disable_communication_step(self):
        """Test a disable comms step where no behaviour is defined raises warning."""
        with warnings.catch_warnings(record=True) as w:
            self.agent.disable_communication_step()
            assert len(w) == 1
            assert "Disable Communication behaviour requested for agent" in str(w[0].message)

    def test_fire_step(self):
        """Test a fire step where no behaviour is defined raises a warning."""
        with warnings.catch_warnings(record=True) as w:
            weapon = MockWeapon()
            self.agent.fire_step(rounds=1, weapon=weapon, target=(0, 0))
            assert len(w) == 1
            assert "Fire behaviour requested for agent" in str(w[0].message)

    def test_get_sense_schedule(self):
        """Test a sense schedule with no sense behaviour raises a warning."""
        with warnings.catch_warnings(record=True) as w:
            schedule = self.agent.get_sense_schedule(duration=timedelta(seconds=5))
            assert len(w) == 1  # type: ignore
            assert "Sense behaviour requested for agent" in str(w[0].message)  # type: ignore
            assert schedule.is_complete()

    def test_deploy_countermeasure_step(self):
        """Test a deploy countermeasure step with no behaviour raises a warning."""
        with warnings.catch_warnings(record=True) as w:
            self.agent.deploy_countermeasure_step()
            assert len(w) == 1
            assert "Deploy Countermeasure behaviour requested for agent" in str(w[0].message)

    def test_icon_paths(self):
        """Test that an agent can be given new icons."""
        icon_path = self.agent.icon_dict[Affiliation.HOSTILE]
        icon_path_str = str(icon_path)

        with open(icon_path) as i:
            expected = i.read()

        self.agent.set_icon_paths(icon_filepath=icon_path_str)
        assert self.agent.icon == expected

        killed_icon_path = self.agent.killed_icon_dict[Affiliation.HOSTILE]
        killed_icon_path_str = str(killed_icon_path)

        with open(killed_icon_path) as j:
            killed_expected = j.read()

        self.agent.set_icon_paths(killed_icon_filepath=killed_icon_path_str)
        self.agent.kill()
        assert self.agent.icon == killed_expected

    def test_killed_icon(self):
        """Test that the agent's icon changes when the agent is killed."""
        self.agent.kill()
        with open(self.agent.killed_icon_dict[Affiliation.FRIENDLY]) as i:
            expected = i.read()
        assert self.agent.icon == expected

    def test_wrong_icon_type(self):
        """Test that only SVG files can be provided as icons."""
        with self.assertRaises(TypeError) as e:
            self.agent.set_icon_paths(icon_filepath="__init__.py")
        assert "Icon file type incompatible. Please use SVG file type." in e.exception.args[0]

        with self.assertRaises(TypeError) as e:
            self.agent.set_icon_paths(killed_icon_filepath="__init__.py")
        assert "Icon file type incompatible. Please use SVG file type." in e.exception.args[0]


class RetAgentTemplate(RetAgentNoBehavioursTemplate):
    """Template test case for RetAgents.

    In order to run this test case for a bespoke Agent type, create a new
    test class which inherits from this class, and unittest.TestCase, and
    implement a `create_agent` method which returns the agent you wish to
    test.

    You can add any additional agent-specific test cases to the model should
    you wish.
    """

    communication_network: MockCommunicationNetwork

    def setUp(self):
        """Set up test cases."""
        super().setUp()
        self.agent.communication_network = self.communication_network

    def define_variables(self) -> None:
        """Define variables."""
        self.pos: Coordinate2d = (0, 0)
        self.name: str = "Agent"
        self.affiliation: Affiliation = Affiliation.FRIENDLY

        time_0 = datetime(2020, 1, 1, 0, 0)
        time_1 = time_0 + timedelta(minutes=1)
        time_2 = time_1 + timedelta(minutes=1)
        time_3 = time_2 + timedelta(minutes=1)
        time_4 = time_3 + timedelta(minutes=1)
        time_5 = time_4 + timedelta(minutes=1)

        self.orders = [
            Order(TimeTrigger(time_0, True), MoveTask((0, 50), 1), priority=0),
            Order(
                TimeTrigger(time_1, True),
                WaitTask(timedelta(minutes=2)),
                priority=0,
                persistent=True,
            ),
            Order(TimeTrigger(time_2, True), MoveTask((0, 200), 1), priority=0),
            Order(TimeTrigger(time_3, True), WaitTask(timedelta(minutes=2)), priority=1),
            Order(
                TimeTrigger(time_4, True),
                MoveTask((0, 100), 1),
                priority=2,
                persistent=True,
            ),
            Order(TimeTrigger(time_5, True), MoveTask((0, 200), 1), priority=3),
        ]
        # move to 50
        # wait for 3 time steps
        # move to 100
        # move to 200
        # infinite moves to 100

        self.move_behaviour = MockMoveBehaviour()
        self.hide_behaviour = MockHideBehaviour()
        self.wait_behaviour = MockWaitBehaviour()
        self.sense_behaviour = MockSenseBehaviour(
            time_before_first_sense=timedelta(seconds=0), time_between_senses=timedelta(seconds=5)
        )
        self.sensors = [MockSensor()]
        self.communicate_orders_behaviour = MockCommunicateOrdersBehaviour()
        self.communicate_worldview_behaviour = MockCommunicateWorldviewBehaviour()
        self.communicate_mission_message_behaviour = MockCommunicateMissionMessageBehaviour()
        self.fire_behaviour = MockFireBehaviour()
        self.disable_communication_behaviour = MockDisableCommunicationBehaviour()
        self.communication_receiver = CommunicationReceiver()
        self.refresh_technique = RemoveDuplicates()
        self.deploy_countermeasure_behaviour = MockDeployCountermeasureBehaviour()
        self.countermeasures = [
            MockCountermeasure(persist_beyond_deployer=True),
            MockCountermeasure(persist_beyond_deployer=False),
        ]

        self.communication_network = MockCommunicationNetwork()

    def test_model_step(self):
        """Test that the model can successfully step without error."""
        for _ in range(12):
            self.model.step()

    def test_model_running(self):
        """Test that the model can successfully be run without error."""
        self.model.run_model()

    def test_agent_step(self):
        """Test that the agent can successfully step without error."""
        self.agent.step()

    def test_check_for_new_active_order(self):
        """Test that agent can check for active orders."""
        assert self.agent.active_order is None
        assert len(self.agent._orders) == 6
        self.agent.check_for_new_active_order()
        assert self.agent.active_order._trigger._time == self.orders[0]._trigger._time
        assert len(self.agent._orders) == 6

    def test_agent_step_order_completed(self):
        """Test that the agent will activate their first order after the first step."""
        assert len(self.agent._orders) == 6
        assert self.agent.active_order is None
        self.agent.step()  # move0 - to 50
        assert self.agent.active_order._trigger._time == self.orders[0]._trigger._time
        assert self.agent.pos[0] == 0
        assert self.agent.pos[1] == 50
        assert len(self.agent._orders) == 6

    def test_model_step_1_order_completed(self):
        """Test the agent's state after completing it's first order."""
        self.model.step()  # move0 - to 50
        assert self.agent.active_order._trigger._time == self.orders[0]._trigger._time
        assert self.agent.pos[0] == 0
        assert self.agent.pos[1] == 50
        assert len(self.agent._orders) == 6

    def test_model_step_2_order_retained(self):
        """Test the agent's state after completing it's first and second order."""
        self.model.step()  # move0 - to 50
        self.model.step()  # wait0 - 1st half
        assert self.agent.active_order._trigger._time == self.orders[1]._trigger._time
        assert self.agent.pos[0] == 0
        assert self.agent.pos[1] == 50
        assert len(self.agent._orders) == 5

    def test_model_step_3_order_retained_favoured_new_order_same_priority(self):
        """Test that active orders are prioritised over new orders at same priority."""
        self.model.step()  # move0 - to 50
        self.model.step()  # wait0 - 1st half
        self.model.step()  # wait0 - completed
        assert self.agent.active_order._trigger._time == self.orders[1]._trigger._time
        assert self.agent.pos[0] == 0
        assert self.agent.pos[1] == 50
        assert len(self.agent._orders) == 5

    def test_model_step_4_order_high_priority_chosen_over_old_order(self):
        """Test that new higher priority orders supersede active order."""
        self.model.step()  # move0 - to 50
        self.model.step()  # wait0 - 1st half
        self.model.step()  # wait0 - completed
        self.model.step()  # wait1 - 1st half
        assert self.agent.active_order._trigger._time == self.orders[3]._trigger._time
        assert self.agent.pos[0] == 0
        assert self.agent.pos[1] == 50
        assert len(self.agent._orders) == 5

    def test_model_step_5_order_higher_priority_interrupts_retained_order(self):
        """Test that a higher priority order can interrupt an existing order."""
        self.model.step()  # move0 - to 50
        self.model.step()  # wait0 - 1st half
        self.model.step()  # wait0 - completed
        self.model.step()  # wait1 - 1st half
        self.model.step()  # move2 - to 100
        assert self.agent.active_order._trigger._time == self.orders[4]._trigger._time
        assert self.agent.pos[0] == 0
        assert self.agent.pos[1] == 100
        assert len(self.agent._orders) == 4

    def test_model_step_6_order_higher_priority_chosen_over_persistent(self):
        """Test that high priority orders are selected over persistent orders."""
        self.model.step()  # move0 - to 50
        self.model.step()  # wait0 - 1st half
        self.model.step()  # wait0 - completed
        self.model.step()  # wait1 - 1st half
        self.model.step()  # move2 - to 100
        self.model.step()  # move3 - to 200
        assert self.agent.active_order._trigger._time == self.orders[5]._trigger._time
        assert self.agent.pos[0] == 0
        assert self.agent.pos[1] == 200
        assert len(self.agent._orders) == 4

    def test_model_step_7_order_persistent_returned_to_after_higher_priority(self):
        """Test that persistent orders are returned to after high priority interrupt."""
        self.model.step()  # move0 - to 50
        self.model.step()  # wait0 - 1st half
        self.model.step()  # wait0 - completed
        self.model.step()  # wait1 - 1st half
        self.model.step()  # move2 - to 100
        self.model.step()  # move3 - to 200
        self.model.step()  # move2 - to 100
        assert self.agent.active_order._trigger._time == self.orders[4]._trigger._time
        assert self.agent.pos[0] == 0
        assert self.agent.pos[1] == 100
        assert len(self.agent._orders) == 3

    def test_model_step_8_order_repeats_highest_persistent_order_indefinitely(self):
        """Test model repeats highest persistent order indefinitely."""
        self.model.step()  # move0 - to 50
        self.model.step()  # wait0 - 1st half
        self.model.step()  # wait0 - completed
        self.model.step()  # wait1 - 1st half
        self.model.step()  # move2 - to 100
        self.model.step()  # move3 - to 200
        self.model.step()  # move2 - to 100
        self.model.step()  # move2 - to 100
        assert self.agent.active_order._trigger._time == self.orders[4]._trigger._time
        assert self.agent.pos[0] == 0
        assert self.agent.pos[1] == 100
        assert len(self.agent._orders) == 3

    def test_step_after_killed(self):
        """Test that agent can check for active orders."""
        assert self.agent.active_order is None
        self.agent.kill()
        self.agent.step()
        assert self.agent.active_order is None

    def test_move_step(self):
        """Test agent's position after performing a move step."""
        self.agent.move_step((0, 100))
        self.assertTrue(self.move_behaviour.stepped)

    def test_wait_step(self):
        """Test whether agent can perform a wait step."""
        self.agent.wait_step()
        self.assertTrue(self.wait_behaviour.stepped)

    def test_hide_step(self):
        """Test whether agent can perform a hide step."""
        self.agent.hide_step()
        self.assertTrue(self.hide_behaviour.stepped)

    def test_sense_step(self):
        """Test whether agent can perform a sense step."""
        self.agent.sense_step(sense_direction=90)
        self.assertTrue(self.sense_behaviour.stepped)

    def test_communicate_worldview_step(self):
        """Test whether agent can perform a communicate worldview step."""
        self.agent.communicate_worldview_step()
        self.assertTrue(self.communication_network.worldview_stepped)

    def test_communicate_orders_step(self):
        """Test whether agent can perform a communicate orders step."""
        self.agent.communicate_orders_step([])
        self.assertTrue(self.communication_network.orders_stepped)

    def test_disable_communication_step(self):
        """Test whether agent can perform a disable communicate step."""
        self.agent.disable_communication_step()
        self.assertTrue(self.disable_communication_behaviour.stepped)

    def test_fire_step(self):
        """Test whether agent can perform a fire step."""
        weapon = MockWeapon()
        self.agent.fire_step(rounds=1, weapon=weapon, target=(0, 0))
        assert self.fire_behaviour.stepped

    def test_get_sense_schedule(self):
        """Test whether an agent can generate a sense schedule."""
        schedule = self.agent.get_sense_schedule(duration=timedelta(seconds=5))
        assert isinstance(schedule, SenseSchedule)

    def test_clear_sense_behaviour(self):
        """Test whether an agent can clear a sense behaviour."""
        self.agent._existing_sense_behaviour = self.sense_behaviour
        self.agent.clear_sense_behaviour()
        assert self.agent._existing_sense_behaviour is None


class RetAgentTemplateWithCountermeasures(RetAgentTemplate):
    """Template test case for RetAgents with countermeasures.

    In order to run this test case for a bespoke Agent type, create a new
    test class which inherits from this class, and unittest.TestCase, and
    implement a `create_agent` method which returns the agent you wish to
    test.

    You can add any additional agent-specific test cases to the model should
    you wish.
    """

    def test_kill_countermeasures(self):
        """Test that countermeasures are killed when the agent is killed."""
        self.assertFalse(self.agent._countermeasures[0].killed)
        self.assertFalse(self.agent._countermeasures[1].killed)

        self.agent.kill()
        self.assertFalse(self.agent._countermeasures[0].killed)
        self.assertFalse(self.agent._countermeasures[1].killed)

        self.agent.deploy_countermeasure_step()
        self.agent.deploy_countermeasure_step()
        self.assertTrue(self.agent._countermeasures[0].deployed)
        self.assertTrue(self.agent._countermeasures[1].deployed)
        self.assertFalse(self.agent._countermeasures[0].killed)
        self.assertFalse(self.agent._countermeasures[1].killed)

        self.agent.kill()
        self.assertFalse(self.agent._countermeasures[0].killed)
        self.assertTrue(self.agent._countermeasures[1].killed)

    def test_kill_countermeasures_unique_to_agent(self):
        """Test that countermeasures assigned to two agents are killed correctly."""
        other_agent = RetAgent(
            model=self.model,
            pos=self.pos,
            name=self.name,
            affiliation=self.affiliation,
            behaviours=[self.deploy_countermeasure_behaviour],
            countermeasures=self.countermeasures,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        self.assertFalse(self.agent._countermeasures[0].killed)
        self.assertFalse(self.agent._countermeasures[1].killed)
        self.assertFalse(other_agent._countermeasures[0].killed)
        self.assertFalse(other_agent._countermeasures[1].killed)

        self.agent.deploy_countermeasure_step()
        self.agent.deploy_countermeasure_step()
        other_agent.deploy_countermeasure_step()
        other_agent.deploy_countermeasure_step()

        self.agent.kill()
        self.assertFalse(self.agent._countermeasures[0].killed)
        self.assertTrue(self.agent._countermeasures[1].killed)
        self.assertFalse(other_agent._countermeasures[0].killed)
        self.assertFalse(other_agent._countermeasures[1].killed)

        other_agent.kill()
        self.assertFalse(self.agent._countermeasures[0].killed)
        self.assertTrue(self.agent._countermeasures[1].killed)
        self.assertFalse(other_agent._countermeasures[0].killed)
        self.assertTrue(other_agent._countermeasures[1].killed)

    def test_deploy_countermeasure_step(self):
        """Test whether agent can perform a deploy countermeasure step."""
        self.agent.deploy_countermeasure_step()
        self.assertTrue(self.deploy_countermeasure_behaviour.stepped)
