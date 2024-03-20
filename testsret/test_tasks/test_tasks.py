"""Tests for tasks."""

from __future__ import annotations

import warnings
from collections import deque
from datetime import datetime, timedelta
from typing import TYPE_CHECKING
from unittest import TestCase

from ret.agents.affiliation import Affiliation
from ret.agents.agent import RetAgent
from ret.agents.agentfilter import FilterByAffiliation, FilterFriendly, FilterNeutral
from ret.behaviours.disablecommunication import DisableCommunicationBehaviour
from ret.behaviours.sense import SenseSchedule
from ret.orders.order import CompoundTask, Order
from ret.orders.tasks.communicate import (
    CommunicateMissionMessageTask,
    CommunicateOrderTask,
    CommunicateWorldviewTask,
)
from ret.orders.tasks.deploycountermeasure import DeployCountermeasureTask
from ret.orders.tasks.disablecommunication import DisableCommunicationTask
from ret.orders.tasks.fire import (
    DetermineTargetAndFireTask,
    FireAtAreaTask,
    FireAtTargetTask,
    RandomWeaponSelector,
    WeaponSelector,
)
from ret.orders.tasks.hide import HideTask
from ret.orders.tasks.move import MoveTask
from ret.orders.tasks.operate import OperateTask
from ret.orders.tasks.projectile import (
    DetonateOnImpactTask,
    DetonateTask,
    GuidedDetonateOnImpactTask,
)
from ret.orders.tasks.sense import SenseTask
from ret.orders.tasks.wait import WaitTask
from ret.orders.triggers.time import TimeTrigger
from ret.sensing.agentcasualtystate import AgentCasualtyState
from ret.sensing.perceivedworld import AffiliatedAgents, AgentById, Confidence, PerceivedAgent
from ret.space.feature import BoxFeature
from ret.testing.mocks import (
    MockCommunicateMissionMessageBehaviour,
    MockCommunicateOrdersBehaviour,
    MockCommunicateWorldviewBehaviour,
    MockCountermeasure,
    MockDeployCountermeasureBehaviour,
    MockFireBehaviour,
    MockHideBehaviour,
    MockModel2d,
    MockMoveBehaviour,
    MockMultipleShotsFiredLogger,
    MockOrderWithId,
    MockSenseBehaviour,
    MockWaitBehaviour,
    MockWeapon,
)
from parameterized import parameterized

if TYPE_CHECKING:
    from collections.abc import Sequence
    from random import Random
    from typing import Optional

    from ret.behaviours.sense import SenseBehaviour
    from ret.orders.order import Task
    from ret.space.clutter.countermeasure import Countermeasure
    from ret.template import Template
    from ret.weapons.weapon import Weapon


class MockDisableCommunicationBehaviour(DisableCommunicationBehaviour):
    """Disable Communication Behaviour.

    Disables all outgoing comms from an agent which is defined at initialisation
    """

    agent_to_disable: RetAgent

    def __init__(self, agent_to_disable: RetAgent):
        """Create a new DisableCommunicationBehaviour.

        Args:
            agent_to_disable (RetAgent): Agent to disable
        """
        super().__init__(log=False)
        self.agent_to_disable = agent_to_disable

    def identify_agents_to_disable(
        self, disabler: RetAgent
    ) -> list[tuple[RetAgent, Optional[list[RetAgent]]]]:
        """Identify agents to disable - No agents are identified.

        Args:
            disabler (RetAgent): The agent doing the disabling

        Returns:
            list[tuple[RetAgent, Optional[list[RetAgent]]]]: Agents to disable
        """
        return [(self.agent_to_disable, None)]


class TestTasks(TestCase):
    """Test cases for Tasks."""

    def setUp(self):
        """Set up test cases."""
        self.model = MockModel2d()

        self.second_agent = RetAgent(
            model=self.model,
            pos=(10, 10),
            name="Second agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        self.move_behaviour = MockMoveBehaviour()
        self.wait_behaviour = MockWaitBehaviour()
        self.communicate_orders_behaviour = MockCommunicateOrdersBehaviour()
        self.communicate_worldview_behaviour = MockCommunicateWorldviewBehaviour()
        self.communicate_mission_message_behaviour = MockCommunicateMissionMessageBehaviour()
        self.sense_behaviour = MockSenseBehaviour(
            time_before_first_sense=timedelta(seconds=0), time_between_senses=timedelta(seconds=5)
        )
        self.deploy_countermeasure_behaviour = MockDeployCountermeasureBehaviour()
        self.countermeasures: list[Template[Countermeasure]] = [MockCountermeasure()]
        self.fire_behaviour = MockFireBehaviour()
        self.disable_communication_behaviour = MockDisableCommunicationBehaviour(self.second_agent)
        self.hide_behaviour = MockHideBehaviour()

        behaviours = [
            self.move_behaviour,
            self.wait_behaviour,
            self.communicate_orders_behaviour,
            self.communicate_worldview_behaviour,
            self.communicate_mission_message_behaviour,
            self.sense_behaviour,
            self.deploy_countermeasure_behaviour,
            self.fire_behaviour,
            self.disable_communication_behaviour,
            self.hide_behaviour,
        ]

        self.agent = RetAgent(
            model=self.model,
            pos=(0, 0),
            name="Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            behaviours=behaviours,
            countermeasures=self.countermeasures,
            weapons=[
                MockWeapon(
                    radius=1,
                    time_between_rounds=timedelta(hours=1),
                    time_before_first_shot=timedelta(seconds=0),
                    kill_probability_per_round=1,
                )
            ],
        )

        self.third_agent = RetAgent(
            model=self.model,
            pos=(100, 100),
            name="Third agent",
            affiliation=Affiliation.NEUTRAL,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        self.hostile_agent = RetAgent(
            model=self.model,
            pos=(3, 3),
            name="Hostile",
            affiliation=Affiliation.HOSTILE,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            behaviours=[MockDisableCommunicationBehaviour(self.agent)],
        )

    def test_wait_short(self):
        """Test agent's state after short wait."""
        task = WaitTask(timedelta(hours=0.5))

        with self.assertWarns(Warning) as context:
            assert not task.is_task_complete(self.agent)

        self.assertTrue("Wait end time was not set" in str(context.warning))

        task.do_task_step(self.agent)
        assert task.is_task_complete(self.agent)

        self.model.schedule.steps += 1
        self.model.schedule.time += 1
        assert task.is_task_complete(self.agent)

    def test_wait_long(self):
        """Test agent's state after long wait which exceeds model duration."""
        task = WaitTask(timedelta(hours=1.5))

        with self.assertWarns(Warning) as context:
            assert not task.is_task_complete(self.agent)

        self.assertTrue("Wait end time was not set" in str(context.warning))

        task.do_task_step(self.agent)
        assert not task.is_task_complete(self.agent)

        self.model.schedule.steps += 1
        self.model.schedule.time += 1
        assert task.is_task_complete(self.agent)

    def test_wait_long_delayed_check(self):
        """Test agent's throws a warning if check is completed late."""
        task = WaitTask(timedelta(hours=1.5))

        task.do_task_step(self.agent)
        self.model.schedule.steps += 1
        self.model.schedule.time += 1
        self.model.schedule.steps += 1
        self.model.schedule.time += 1

        with self.assertWarns(Warning) as context:
            task.is_task_complete(self.agent)
        self.assertTrue("Task was completed before this time step" in str(context.warning))

    def test_compound_task(self):
        """Test state of agent after completing compound move task."""
        task1 = WaitTask(timedelta(hours=1.5))
        task2 = MoveTask((5000, 5000), 1)
        task3 = MoveTask((9000, 5000), 1)

        compound_task = CompoundTask([task1, task2, task3])

        compound_task.do_task_step(self.agent)
        self.model.schedule.steps += 1
        self.model.schedule.time += 1
        assert not compound_task.is_task_complete(self.agent)

        compound_task.do_task_step(self.agent)
        self.model.schedule.steps += 1
        self.model.schedule.time += 1
        assert not compound_task.is_task_complete(self.agent)

        compound_task.do_task_step(self.agent)
        self.model.schedule.steps += 1
        self.model.schedule.time += 1
        assert self.agent.pos[0] == 9000
        assert self.agent.pos[1] == 5000
        assert compound_task.is_task_complete(self.agent)

        compound_task = compound_task.get_new_instance()

        with self.assertWarns(Warning) as context:
            assert not compound_task.is_task_complete(self.agent)

        self.assertTrue("Wait end time was not set" in str(context.warning))

    def test_disable_communications(self):
        """Test agent's state after disable communication task."""
        disable_task = DisableCommunicationTask()

        disable_task.do_task_step(self.hostile_agent)

        assert len(self.agent.communication_network._modifiers) == 1
        self.assertTrue(disable_task.is_task_complete(self.hostile_agent))

    def test_sense_no_existing_schedule(self):
        """Test the agent's state after sense task, with no existing schedule."""
        sense_task = SenseTask(timedelta(seconds=1))

        assert sense_task.sense_schedule is None

        sense_task.do_task_step(self.agent)

        assert isinstance(sense_task.sense_schedule, SenseSchedule)

        behaviours: Sequence[SenseBehaviour] = self.agent.behaviour_pool.expose_behaviour(
            self.agent.behaviour_handlers.sense_handler,
            self.agent.behaviour_handlers.sense_type,
        )

        self.assertTrue(behaviours[0].stepped)  # type: ignore
        self.assertTrue(sense_task.is_task_complete(self.agent))

    def test_sense_existing_empty_schedule(self):
        """Test the agent's state after sense task, with an empty schedule."""
        sense_task = SenseTask(timedelta(seconds=1))
        sense_task.sense_schedule = SenseSchedule()

        assert isinstance(sense_task.sense_schedule, SenseSchedule)

        with warnings.catch_warnings(record=True) as w:
            sense_task.do_task_step(self.agent)
            assert len(w) == 1
            assert "Sense Task is already complete." == str(w[0].message)

        assert isinstance(sense_task.sense_schedule, SenseSchedule)

        behaviours: Sequence[SenseBehaviour] = self.agent.behaviour_pool.expose_behaviour(
            self.agent.behaviour_handlers.sense_handler,
            self.agent.behaviour_handlers.sense_type,
        )

        self.assertFalse(behaviours[0].stepped)  # type: ignore
        self.assertTrue(sense_task.is_task_complete(self.agent))

    def test_sense_existing_single_step_schedule(self):
        """Test the agent's state after sense task, with an existing single-step schedule."""
        sense_task = SenseTask(timedelta(seconds=1))
        sense_task.sense_schedule = SenseSchedule()
        sense_task.sense_schedule._steps = deque([True])

        assert isinstance(sense_task.sense_schedule, SenseSchedule)

        sense_task.do_task_step(self.agent)

        assert isinstance(sense_task.sense_schedule, SenseSchedule)

        behaviours: Sequence[SenseBehaviour] = self.agent.behaviour_pool.expose_behaviour(
            self.agent.behaviour_handlers.sense_handler,
            self.agent.behaviour_handlers.sense_type,
        )

        self.assertTrue(behaviours[0].stepped)  # type: ignore
        self.assertTrue(sense_task.is_task_complete(self.agent))

    def test_sense_existing_multiple_step_schedule(self):
        """Test the agent's state after sense task, with an existing multi-step schedule."""
        sense_task = SenseTask(timedelta(seconds=1))
        sense_task.sense_schedule = SenseSchedule()
        sense_task.sense_schedule._steps = deque([True, False, True])

        assert isinstance(sense_task.sense_schedule, SenseSchedule)

        sense_task.do_task_step(self.agent)

        assert isinstance(sense_task.sense_schedule, SenseSchedule)

        behaviours: Sequence[SenseBehaviour] = self.agent.behaviour_pool.expose_behaviour(
            self.agent.behaviour_handlers.sense_handler,
            self.agent.behaviour_handlers.sense_type,
        )

        self.assertTrue(behaviours[0].stepped)  # type: ignore
        self.assertFalse(sense_task.is_task_complete(self.agent))

        sense_task.do_task_step(self.agent)
        self.assertFalse(sense_task.is_task_complete(self.agent))

        sense_task.do_task_step(self.agent)
        self.assertTrue(sense_task.is_task_complete(self.agent))

    def test_sense_is_task_complete_no_schedule(self):
        """Test the _is_task_complete method for sense tasks."""
        task = SenseTask(timedelta(seconds=1))
        task.sense_schedule = None
        self.assertFalse(task.is_task_complete(self.agent))

    def test_deploy_countermeasure(self):
        """Test the agent's state after deploy countermeasure task."""
        deploy_countermeasure_task = DeployCountermeasureTask()

        deploy_countermeasure_task.do_task_step(self.agent)

        self.assertTrue(self.agent._countermeasures[0].deployed)
        self.assertTrue(deploy_countermeasure_task.is_task_complete(self.agent))

    def test_fire_at_target_step(self):
        """Test the agents state following a fire step at a defined target.

        Where an agent fires at a location close to the target, the target is killed.
        """
        assert not self.hostile_agent.killed

        off_target_task = FireAtTargetTask((20, 20))
        off_target_task.do_task_step(self.agent)

        assert not self.hostile_agent.killed

        # Within range of 1 of target
        on_target_task = FireAtTargetTask((3.25, 3.25))
        on_target_task.do_task_step(self.agent)

        assert self.hostile_agent.killed

    def test_fire_at_area_step(self):
        """Test that an agent fires at locations within a defined area.

        Confirms that all logged shots land inside the given area.
        """
        area = BoxFeature((20, 20), (25, 25), "Firing Area")
        self.model.logger = MockMultipleShotsFiredLogger(self.model)
        rounds = 50

        off_target_task = FireAtAreaTask(area, self.agent.model.random, rounds=rounds)

        for _ in range(0, 50):
            off_target_task.do_task_step(self.agent)

        test_set = set()

        for shot in self.model.logger.logged_shots_list:
            test_set.add(shot["aim location"])
            assert area.contains(shot["aim location"])

        assert len(test_set) == 50

    def test_fire_while_hiding(self):
        """Test the agents state following a fire step at a defined target, while hiding.

        The agent should not be able to fire, and the target should remain alive.
        """
        self.agent.hiding = True
        assert not self.hostile_agent.killed

        off_target_task = FireAtTargetTask((20, 20))
        with warnings.catch_warnings(record=True) as w:
            off_target_task.do_task_step(self.agent)
            assert len(w) == 1
            assert "Cannot fire while the agent is hiding." == str(w[0].message)

        assert not self.hostile_agent.killed

        # Within range of 1 of target
        on_target_task = FireAtTargetTask((3.25, 3.25))
        with warnings.catch_warnings(record=True) as w:
            on_target_task.do_task_step(self.agent)
            assert len(w) == 1
            assert "Cannot fire while the agent is hiding." == str(w[0].message)

        assert not self.hostile_agent.killed

    def test_determine_target_and_fire(self):
        """Test the agents state following a fire step without a defined target."""
        assert not self.hostile_agent.killed

        task = DetermineTargetAndFireTask(rounds=2)
        task.do_task_step(self.agent)

        assert not self.hostile_agent.killed

        self.agent.perceived_world.add_acquisitions(
            PerceivedAgent(
                unique_id=self.hostile_agent.unique_id,
                confidence=Confidence.IDENTIFY,
                location=self.hostile_agent.pos,
                affiliation=self.hostile_agent.affiliation,
                sense_time=self.model.get_time(),
                casualty_state=AgentCasualtyState.ALIVE,
            )
        )

        task.do_task_step(self.agent)

        assert self.hostile_agent.killed

    def test_handling_of_agent_with_no_weapons(self):
        """Test that warnings are raised if a task requests to fire with non-existent weapon."""

        class NullWeaponSelector(WeaponSelector):
            """Weapon Selector which never returns a weapon."""

            def __init__(self):
                """Create a new NullWeaponSelector."""
                pass

            def select_weapon(self, candidates: list[Weapon], rand: Random) -> Optional[Weapon]:
                """Attempts to select a weapon, always returning None.

                Regardless of the weapons in the candidates list, None are selected.

                Args:
                    candidates (list[Weapon]): Candidates for selection
                    rand (Random): Randomisation

                Returns:
                    Optional[Weapon]: Selected weapon - Always None
                """
                return None

        agent = RetAgent(
            model=self.model,
            pos=(0, 0),
            name="Test agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=1.0,
            reflectivity=1.0,
            temperature=1.0,
            weapons=[MockWeapon()],
        )

        # This just confirms that the agent has weapons - The purpose of this test is to check
        # the warning if none of the weapons are suitable. A different warning is raised if the
        # agent has no weapons at all, and this is tested explicitly where testing the
        # WeaponSelector implementation
        assert len(agent.weapons) != 0

        task = FireAtTargetTask(target=(10, 10), weapon_selector=NullWeaponSelector(), rounds=1)

        with warnings.catch_warnings(record=True) as w:
            task.do_task_step(agent)

        assert w is not None
        assert str(w[0].message) == "The firer has no suitable weapons for this Task."

    def test_communicate_orders(self):
        """Test communicating orders task."""
        self.agent.communication_network.add_recipient([self.second_agent, self.third_agent])

        task = CommunicateOrderTask([MockOrderWithId(1), MockOrderWithId(2)])
        task.do_task_step(self.agent)

        orders2: list[MockOrderWithId] = self.second_agent._orders  # type: ignore
        orders3: list[MockOrderWithId] = self.third_agent._orders  # type: ignore

        assert True in (o.id == 1 for o in orders2)
        assert True in (o.id == 2 for o in orders2)

        assert True in (o.id == 1 for o in orders3)
        assert True in (o.id == 2 for o in orders3)

        assert len(orders2) == 2
        assert len(orders3) == 2

        assert task.is_task_complete(self.agent)

    def test_communicate_worldview(self):
        """Test communicating worldview task."""
        self.agent.communication_network.add_recipient(self.second_agent)
        self.agent.communication_network.add_recipient(self.third_agent)

        view_id = 1000

        self.agent.perceived_world.add_acquisitions(
            PerceivedAgent(
                unique_id=view_id,
                confidence=Confidence.KNOWN,
                sense_time=self.model.get_time(),
                location=(150, 150),
                casualty_state=AgentCasualtyState.ALIVE,
            )
        )

        task = CommunicateWorldviewTask()
        task.do_task_step(self.agent)

        # Views contain the new perceived agent, self.agent and either self.second_agent
        # or self.third_agent

        a2views = self.second_agent.perceived_world.get_perceived_agents()
        a3views = self.third_agent.perceived_world.get_perceived_agents()

        assert len(a2views) == 3
        assert len(a3views) == 3

        assert True in (o.unique_id == view_id for o in a2views)
        assert True in (o.unique_id == view_id for o in a3views)
        assert True in (o.unique_id == self.agent.unique_id for o in a2views)
        assert True in (o.unique_id == self.agent.unique_id for o in a3views)
        assert True in (o.unique_id == self.second_agent.unique_id for o in a2views)
        assert True in (o.unique_id == self.third_agent.unique_id for o in a3views)

        assert task.is_task_complete(self.agent)

    def test_communicate_filtered_worldview(self):
        """Test that the perceived agent filters are applied where sending views."""
        task_1_id = 1000
        task_2_id = 2000

        self.agent.communication_network.add_recipient(self.second_agent)
        self.agent.perceived_world.add_acquisitions(
            [
                PerceivedAgent(
                    sense_time=self.model.get_time(),
                    confidence=Confidence.KNOWN,
                    location=(100, 100),
                    unique_id=task_1_id,
                    casualty_state=AgentCasualtyState.ALIVE,
                ),
                PerceivedAgent(
                    sense_time=self.model.get_time(),
                    confidence=Confidence.KNOWN,
                    location=(200, 200),
                    unique_id=task_2_id,
                    casualty_state=AgentCasualtyState.ALIVE,
                ),
            ]
        )

        task_1_filter = AgentById(task_1_id)
        task_2_filter = AgentById(task_2_id)

        task_1 = CommunicateWorldviewTask(worldview_filter=task_1_filter)
        task_2 = CommunicateWorldviewTask(worldview_filter=task_2_filter)

        task_1.do_task_step(self.agent)

        post_task_1_views = self.second_agent.perceived_world.get_perceived_agents()

        assert len(task_1_filter.run(post_task_1_views)) == 1
        assert len(task_2_filter.run(post_task_1_views)) == 0

        task_2.do_task_step(self.agent)

        post_task_2_views = self.second_agent.perceived_world.get_perceived_agents()

        assert len(task_2_filter.run(post_task_2_views)) == 1

    def test_communicate_orders_to_filtered_recipients(self):
        """Test communicating orders to subset of network."""
        self.agent.communication_network.add_recipient(self.second_agent)
        self.agent.communication_network.add_recipient(self.third_agent)

        send_to_neutral_task = CommunicateOrderTask(
            [MockOrderWithId(1), MockOrderWithId(2)],
            recipient_filter=FilterNeutral(),
        )
        send_to_neutral_task.do_task_step(self.agent)

        orders2: list[MockOrderWithId] = self.second_agent._orders  # type: ignore
        orders3: list[MockOrderWithId] = self.third_agent._orders  # type: ignore

        assert True not in (o.id == 1 for o in orders2)
        assert True not in (o.id == 2 for o in orders2)

        assert True in (o.id == 1 for o in orders3)
        assert True in (o.id == 2 for o in orders3)

        assert len(orders2) == 0
        assert len(orders3) == 2

        assert send_to_neutral_task.is_task_complete(self.agent)

    def test_communicate_worldview_to_filtered_recipients(self):
        """Test communicating worldview to subset of network."""
        self.agent.communication_network.add_recipient(self.second_agent)
        self.agent.communication_network.add_recipient(self.third_agent)

        view_id = 1000

        self.agent.perceived_world.add_acquisitions(
            PerceivedAgent(
                unique_id=view_id,
                confidence=Confidence.KNOWN,
                sense_time=self.model.get_time(),
                location=(150, 150),
                casualty_state=AgentCasualtyState.ALIVE,
            )
        )

        send_to_friendly_task = CommunicateWorldviewTask(recipient_filter=FilterFriendly())
        send_to_friendly_task.do_task_step(self.agent)

        # Views contain the new perceived agent, self.agent and either self.second_agent
        # or self.third_agent

        a2views = self.second_agent.perceived_world.get_perceived_agents()
        a3views = self.third_agent.perceived_world.get_perceived_agents()

        assert len(a2views) == 3
        assert len(a3views) == 1

        assert True in (o.unique_id == view_id for o in a2views)
        assert True not in (o.unique_id == view_id for o in a3views)
        assert True in (o.unique_id == self.agent.unique_id for o in a2views)
        assert True not in (o.unique_id == self.agent.unique_id for o in a3views)
        assert True in (o.unique_id == self.second_agent.unique_id for o in a2views)
        assert True in (o.unique_id == self.third_agent.unique_id for o in a3views)

        assert send_to_friendly_task.is_task_complete(self.agent)

    def test_communicate_mission_message(self):
        """Test communicating mission message task."""
        self.agent.communication_network.add_recipient(self.second_agent)
        self.agent.communication_network.add_recipient(self.third_agent)

        task = CommunicateMissionMessageTask(message="test", recipient_filter=FilterFriendly())
        task.do_task_step(self.agent)

        assert len(self.second_agent.mission_messages) == 1
        assert "test" in self.second_agent.mission_messages
        assert len(self.third_agent.mission_messages) == 0
        assert "test" not in self.third_agent.mission_messages

        assert task.is_task_complete(self.agent)

    def test_move_str_method(self):
        """Test the move task string method."""
        task = MoveTask((5000, 5000), 1)
        assert str(task) == "Move Task"

    def test_wait_str_method(self):
        """Test the wait task string method."""
        task = WaitTask(timedelta(hours=0.5))
        assert str(task) == "Wait Task"

    def test_disable_communication_str_method(self):
        """Test the disable communication task string method."""
        task = DisableCommunicationTask()
        assert str(task) == "Disable Communication Task"

    def test_communicate_order_str_method(self):
        """Test the communicate task string method."""
        orders: list[Template[Order]] = [
            Order(TimeTrigger(datetime(2020, 1, 1, 0, 1)), MoveTask((5000, 5000), 1))
        ]
        task = CommunicateOrderTask(orders)
        assert str(task) == "Communicate Order Task"

    def test_communicate_world_view_str_method(self):
        """Test the communicate world view task string method."""
        task = CommunicateWorldviewTask()
        assert str(task) == "Communicate World View Task"

    def test_sense_str_method(self):
        """Test the sense task string method."""
        task = SenseTask(timedelta(seconds=1))
        assert str(task) == "Sense Task"

    def test_deploy_countermeasure_str_method(self):
        """Test the deploy countermeasure task string method."""
        task = DeployCountermeasureTask()
        assert str(task) == "Deploy Countermeasure Task"

    def test_compound_3_task_str_method(self):
        """Test the compound task string method."""
        task1 = WaitTask(timedelta(hours=1.5))
        task2 = MoveTask((5000, 5000), 1)
        task3 = MoveTask((9000, 5000), 1)
        task = CompoundTask([task1, task2, task3])
        assert (
            str(task)
            == "Compound Task: (" + str(task1) + ", " + str(task2) + ", " + str(task3) + ")"
        )

    def test_compound_1_task_str_method(self):
        """Test the compound task string method."""
        task1 = WaitTask(timedelta(hours=1.5))
        task = CompoundTask([task1])
        assert str(task) == "Compound Task: (" + str(task1) + ")"

    def test_fire_at_target_task_str_method(self):
        """Test the fire at target string method."""
        task = FireAtTargetTask((0, 0))
        assert str(task) == "Fire at Target Task"

    def test_determine_target_and_fire_task_str_method(self):
        """Test the determine target and fire string method."""
        task = DetermineTargetAndFireTask()
        assert str(task) == "Determine Target and Fire Task"

    def test_move_get_new_instance(self):
        """Test the move task get_new_instance method."""
        task = MoveTask(destination=(5000, 5000), tolerance=1, log=False)
        clone = task.get_new_instance()

        assert task is not clone
        assert isinstance(clone, MoveTask)
        assert task._destination == clone._destination
        assert task._tolerance == clone._tolerance
        assert task._log == clone._log

    def test_wait_get_new_instance(self):
        """Test the wait task get_new_instance method."""
        task = WaitTask(duration=timedelta(hours=0.5), log=False)
        clone = task.get_new_instance()

        assert task is not clone
        assert isinstance(clone, WaitTask)
        assert task._duration == clone._duration
        assert task._log == clone._log

    def test_disable_communication_get_new_instance(self):
        """Test the disable communication task get_new_instance method."""
        task = DisableCommunicationTask(log=False)
        clone = task.get_new_instance()

        assert task is not clone
        assert isinstance(clone, DisableCommunicationTask)
        assert task._log == clone._log

    def test_communicate_order_get_new_instance(self):
        """Test the communicate task get_new_instance method."""
        orders: list[Template[Order]] = [
            Order(TimeTrigger(datetime(2020, 1, 1, 0, 1)), MoveTask((5000, 5000), 1))
        ]
        task = CommunicateOrderTask(
            orders=orders, log=False, recipient_filter=FilterByAffiliation(Affiliation.FRIENDLY)
        )
        clone = task.get_new_instance()

        assert task is not clone
        assert isinstance(clone, CommunicateOrderTask)
        assert task._orders == clone._orders
        assert task._log == clone._log
        assert task._recipient_filter == clone._recipient_filter

    def test_communicate_world_view_get_new_instance(self):
        """Test the communicate world view task get_new_instance method."""
        task = CommunicateWorldviewTask(
            worldview_filter=AffiliatedAgents(Affiliation.FRIENDLY),
            log=False,
            recipient_filter=FilterByAffiliation(Affiliation.FRIENDLY),
        )
        clone = task.get_new_instance()

        assert task is not clone
        assert isinstance(clone, CommunicateWorldviewTask)
        assert task._worldview_filter == clone._worldview_filter
        assert task._log == clone._log
        assert task._recipient_filter == clone._recipient_filter

    def test_sense_get_new_instance(self):
        """Test the sense task get_new_instance method."""
        task = SenseTask(duration=timedelta(seconds=1), direction=45, log=False)
        clone = task.get_new_instance()

        assert task is not clone
        assert isinstance(clone, SenseTask)
        assert task.duration == clone.duration
        assert task._direction == clone._direction
        assert task._log == clone._log

    def test_deploy_countermeasure_get_new_instance(self):
        """Test the deploy countermeasure task get_new_instance method."""
        task = DeployCountermeasureTask(log=False)
        clone = task.get_new_instance()

        assert task is not clone
        assert isinstance(clone, DeployCountermeasureTask)
        assert task._log == clone._log

    def test_compound_task_get_new_instance(self):
        """Test the compound task get_new_instance method."""
        task1 = WaitTask(timedelta(hours=1.5))
        task2 = MoveTask((5000, 5000), 1)
        task3 = MoveTask((9000, 5000), 1)
        task = CompoundTask(tasks=[task1, task2, task3], log=False)
        clone = task.get_new_instance()

        assert task is not clone
        assert isinstance(clone, CompoundTask)
        assert task._task_templates == clone._task_templates
        assert task._remaining_tasks != clone._remaining_tasks
        assert task._log == clone._log

    def test_fire_at_target_task_new_instance(self):
        """Test the fire at target `get_new_instance` method."""
        task = FireAtTargetTask(
            target=(10, 10), rounds=3, weapon_selector=RandomWeaponSelector(), log=False
        )
        clone = task.get_new_instance()

        assert task is not clone
        assert task.target == clone.target
        assert task.rounds_to_fire == clone.rounds_to_fire
        assert task.weapon_selector == clone.weapon_selector
        assert task._log == clone._log

        task.target = (20, 20)
        assert clone.target == (10, 10)

    def test_fire_at_area_task_new_instance(self):
        """Test the fire at area `get_new_instance` method."""
        task = FireAtAreaTask(
            BoxFeature((0, 0), (10, 10), "Box area 1"), rounds=4, random=self.model.random
        )
        clone = task.get_new_instance()

        assert task is not clone
        assert task.target_area == clone.target_area
        assert task.rounds_to_fire == clone.rounds_to_fire
        assert task.weapon_selector == clone.weapon_selector
        assert task._log == clone._log

    def test_determine_target_and_fire_task_new_instance(self):
        """Test the determine target and fire `get_new_instance` method."""
        task = DetermineTargetAndFireTask(
            rounds=3, weapon_selector=RandomWeaponSelector(), log=False
        )
        clone = task.get_new_instance()

        assert task is not clone
        assert task.rounds_to_fire == clone.rounds_to_fire
        assert task.weapon_selector == clone.weapon_selector
        assert task._log == clone._log

    def test_fire_at_target_task_completion_default(self):
        """Test the completion status of a fire at target task."""
        task = FireAtTargetTask((0, 0))
        self.assertFalse(task.is_task_complete(self.agent))
        task.do_task_step(self.agent)
        self.assertTrue(task.is_task_complete(self.agent))

    def test_determine_target_and_fire_completion_default(self):
        """Test the completion status of a determine target and fire task."""
        task = DetermineTargetAndFireTask()
        self.assertFalse(task.is_task_complete(self.agent))
        task.do_task_step(self.agent)
        self.assertTrue(task.is_task_complete(self.agent))

    @parameterized.expand([[1], [2], [3], [4], [5]])
    def test_fire_at_target_task_completion_specified_time_steps(self, n_steps: int):
        """Test the completion status of a fire at target task.

        For cases where the number of time steps that the task runs for is user-defined.

        Args:
            n_steps (int): Number of steps
        """
        task = FireAtTargetTask(target=(0, 0), rounds=n_steps)

        self.assertFalse(task.is_task_complete(self.agent))
        for _ in range(1, n_steps):
            task.do_task_step(self.agent)
            self.assertFalse(task.is_task_complete(self.agent))

        task.do_task_step(self.agent)
        self.assertTrue(task.is_task_complete(self.agent))

    @parameterized.expand([[1], [2], [3], [4], [5]])
    def test_determine_target_and_fire_completion_specified_time_steps(self, n_steps: int):
        """Test the completion status of a determine target and fire task.

        For cases where the number of time steps that the task runs for is user-defined.

        Args:
            n_steps (int): Number of steps
        """
        task = DetermineTargetAndFireTask(rounds=n_steps)

        for _ in range(1, n_steps):
            task.do_task_step(self.agent)
            self.assertFalse(task.is_task_complete(self.agent))

        task.do_task_step(self.agent)
        self.assertTrue(task.is_task_complete(self.agent))

    @parameterized.expand([[0], [-1], [-2], [-3], [-4], [-5]])
    def test_invalid_determine_target_and_fire_task_creation(self, n_steps: int):
        """Test invalid creation of DetermineTargetAndFireTask.

        Args:
            n_steps (int): Number of steps
        """
        with self.assertRaises(ValueError) as e:
            DetermineTargetAndFireTask(rounds=n_steps)
        self.assertEqual(f"Number of rounds must be > 0. {n_steps} provided.", str(e.exception))

    @parameterized.expand([[0], [-1], [-2], [-3], [-4], [-5]])
    def test_invalid_fire_at_target_test(self, n_steps: int):
        """Test invalid creation of FireAtTargetTask.

        Args:
            n_steps (int): Number of steps
        """
        with self.assertRaises(ValueError) as e:
            FireAtTargetTask(target=(0, 0), rounds=n_steps)
        self.assertEqual(f"Number of rounds must be > 0. {n_steps} provided.", str(e.exception))

    def test_warning_on_fire_at_target_step(self):
        """Test that warning is raised if FireAtTargetTask is stepped too far.

        Checks contents of warning to ensure it's the correct warning.
        """
        task = FireAtTargetTask(target=(0, 0), rounds=2)
        task.do_task_step(self.agent)
        task.do_task_step(self.agent)

        with warnings.catch_warnings(record=True) as w:
            task.do_task_step(self.agent)
            assert len(w) == 1
            assert "is already complete" in str(w[0].message)

    def test_warning_on_determine_target_and_fire(self):
        """Test that warning is raised if DetermineTargetAndFireTask is stepped too far.

        Checks contents of warning to ensure it's the correct warning.
        """
        task = DetermineTargetAndFireTask(rounds=2)
        task.do_task_step(self.agent)
        task.do_task_step(self.agent)

        with warnings.catch_warnings(record=True) as w:
            task.do_task_step(self.agent)
        assert len(w) == 1
        assert "is already complete" in str(w[0].message)

    @parameterized.expand(
        [
            [WaitTask(timedelta(hours=0.5))],
            [SenseTask(timedelta(hours=0.5))],
            [DeployCountermeasureTask()],
            [MoveTask((5000, 5000), 1)],
            [DisableCommunicationTask()],
            [FireAtTargetTask((0, 0))],
        ]
    )
    def test_operate(self, default_task: Task):
        """Test operate for a selection of default_tasks.

        Args:
            default_task (Task): The default task to perform while operating
        """
        task = OperateTask(default_task)

        task.do_task_step(self.agent)
        assert task.is_task_complete(self.agent)

        self.model.schedule.steps += 1
        self.model.schedule.time += 1
        assert task.is_task_complete(self.agent)

    def test_operate_str_method(self):
        """Test the operate task string method."""
        default_task = SenseTask(timedelta(hours=0.5))
        task = OperateTask(default_task)
        assert str(task) == "Operate Task"

    @parameterized.expand(
        [
            [WaitTask(timedelta(hours=0.5))],
            [SenseTask(timedelta(hours=0.5))],
            [DeployCountermeasureTask()],
            [MoveTask((5000, 5000), 1)],
            [DisableCommunicationTask()],
            [FireAtTargetTask((0, 0))],
        ]
    )
    def test_operate_get_new_instance(self, default_task: Task):
        """Test operate get_new_instance for a selection of default_tasks.

        Args:
            default_task (Task): The default task to operate.
        """
        task = OperateTask(default_task=default_task, log=False)
        clone = task.get_new_instance()

        assert task is not clone
        assert type(task) is type(clone)
        assert task._default_task == clone._default_task
        assert task._log == clone._log

    def test_hide_short(self):
        """Test hide task."""
        task = HideTask(duration=timedelta(seconds=60))

        with self.assertWarns(Warning) as context:
            assert not task.is_task_complete(self.agent)

        self.assertTrue("Hide end time was not set" in str(context.warning))

        task.do_task_step(self.agent)
        assert task.is_task_complete(self.agent)

        self.model.schedule.steps += 1
        self.model.schedule.time += 1
        assert task.is_task_complete(self.agent)

    def test_hide_long(self):
        """Test agent's state after long hide which exceeds model duration."""
        task = HideTask(timedelta(hours=1.5))

        with self.assertWarns(Warning) as context:
            assert not task.is_task_complete(self.agent)

        self.assertTrue("Hide end time was not set" in str(context.warning))

        task.do_task_step(self.agent)
        assert not task.is_task_complete(self.agent)

        self.model.schedule.steps += 1
        self.model.schedule.time += 1
        assert task.is_task_complete(self.agent)

    def test_hide_long_delayed_check(self):
        """Test agent's throws a warning if check is completed late."""
        task = HideTask(timedelta(hours=1.5))

        task.do_task_step(self.agent)
        self.model.schedule.steps += 1
        self.model.schedule.time += 1
        self.model.schedule.steps += 1
        self.model.schedule.time += 1

        with self.assertWarns(Warning) as context:
            task.is_task_complete(self.agent)
        self.assertTrue("Task was completed before this time step" in str(context.warning))

    def test_hide_str_method(self):
        """Test the hide task string method."""
        task = HideTask(duration=timedelta(seconds=60))
        assert str(task) == "Hide Task"

    def test_hide_get_new_instance(self):
        """Test hide get_new_instance method."""
        task = HideTask(duration=timedelta(seconds=60), log=False)
        clone = task.get_new_instance()

        assert task is not clone
        assert isinstance(clone, HideTask)
        assert task._duration == clone._duration
        assert task._log == clone._log

    def test_detonate(self):
        """Test Detonate task."""
        task = DetonateTask()

        assert not task.is_task_complete(self.agent)

        task.do_task_step(self.agent)

        assert task.is_task_complete(self.agent)
        assert task.get_target() == self.agent.pos
        assert task.__str__() == "Detonate Task"

        clone = task.get_new_instance()

        assert task is not clone
        assert isinstance(clone, DetonateTask)
        assert clone._log == task._log
        assert not clone.is_task_complete(self.agent)

    def test_detonateonimpact(self):
        """Test DetonateOnImpact task."""
        task = DetonateOnImpactTask(self.hostile_agent.pos)

        assert not task.is_task_complete(self.agent)

        clone = task.get_new_instance()

        assert task is not clone
        assert isinstance(clone, DetonateOnImpactTask)
        assert clone._log == task._log
        assert not clone.is_task_complete(self.agent)

    def test_guided_detonateonimpact(self):
        """Test GuidedDetonateOnImpact task."""
        task = GuidedDetonateOnImpactTask(self.hostile_agent.pos, target=[1], max_weapon_steps=50)

        assert not task.is_task_complete(self.agent)

        clone = task.get_new_instance()

        assert task is not clone
        assert isinstance(clone, GuidedDetonateOnImpactTask)
        assert clone._log == task._log
        assert not clone.is_task_complete(self.agent)
