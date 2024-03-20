"""Tests for AI Gym mesa environment."""

from __future__ import annotations

import random
import unittest
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
from gym_ret.envs.ret_env import RetEnv, TaskIds

from ret.agents.affiliation import Affiliation
from ret.agents.agent import RetAgent
from ret.agents.agenttype import AgentType
from ret.behaviours.behaviourpool import NoEqualityAdder
from ret.orders.order import Order
from ret.orders.tasks.communicate import CommunicateWorldviewTask
from ret.orders.tasks.deploycountermeasure import DeployCountermeasureTask
from ret.orders.tasks.disablecommunication import DisableCommunicationTask
from ret.orders.tasks.fire import DetermineTargetAndFireTask, FireAtTargetTask
from ret.orders.tasks.move import MoveInBandTask, MoveTask
from ret.orders.tasks.sense import SenseTask
from ret.orders.tasks.wait import WaitTask
from ret.orders.triggers.immediate import ImmediateTrigger
from ret.sensing.agentcasualtystate import AgentCasualtyState
from ret.sensing.perceivedworld import Confidence, PerceivedAgent
from ret.space.heightband import AbsoluteHeightBand
from ret.testing.mocks import (
    MockCommunicateMissionMessageBehaviour,
    MockCommunicateWorldviewBehaviour,
    MockDeployCountermeasureBehaviour,
    MockDisableCommunicationBehaviour,
    MockFireBehaviour,
    MockModel3d,
    MockMoveBehaviour,
    MockMoveInBandBehaviour,
    MockSenseBehaviour,
    MockTask,
    MockTrigger,
    MockWaitBehaviour,
    MockWeapon,
)

if TYPE_CHECKING:
    from typing import Any, Optional

    from gymnasium import spaces


class MockGymModel(MockModel3d):
    """Extension of MockModel3d to contain AI Gym relevant components."""

    def __init__(self):
        """Create a new MockGymModel."""
        super().__init__()

        behaviours = [
            MockMoveBehaviour(),
            MockWaitBehaviour(),
            MockCommunicateWorldviewBehaviour(),
            MockCommunicateMissionMessageBehaviour(),
            MockDisableCommunicationBehaviour(),
            MockDeployCountermeasureBehaviour(),
            MockSenseBehaviour(
                time_before_first_sense=timedelta(seconds=0),
                time_between_senses=timedelta(seconds=5),
            ),
            MockFireBehaviour(),
        ]

        self.agent_1 = RetAgent(
            model=self,
            pos=(0, 0, 0),
            name="Agent 1",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            behaviours=behaviours,
            behaviour_adder=NoEqualityAdder,
            weapons=[MockWeapon()],
        )

        self.agent_2 = RetAgent(
            model=self,
            pos=(10, 10, 10),
            name="Agent 2",
            affiliation=Affiliation.HOSTILE,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            behaviours=[MockMoveInBandBehaviour([AbsoluteHeightBand("test", 1)])],
            behaviour_adder=NoEqualityAdder,
            weapons=[MockWeapon()],
        )


class MockGymModelWithKwargs(MockGymModel):
    """Extension of the MockGymModel which supports keyword arguments."""

    def __init__(self, arg_1: int):
        """Create a new MockGymModelWithKwargs.

        Args:
            arg_1 (int): Data to store within model for testing kwargs
        """
        super().__init__()
        self.arg_1 = arg_1


class TestGym(unittest.TestCase):
    """Tests for AI Gym mesa env."""

    def setUp(self):
        """Test case set up."""
        self.model = MockGymModel()
        self.env = RetEnv(MockGymModel)

        self.agent_1_id = 1
        self.agent_2_id = 2

    def test_init(self):
        """Test the initialisation of the AI Gym mesa environment."""
        assert self.env._model_cls is MockGymModel
        assert self.env.model is not self.model
        assert self.env.model is not None
        assert self.env.action_space is not None
        assert self.env.observation_space is not None
        assert self.env._n_impossible_tasks_requested == 0

    def test_init_with_kwargs(self):
        """Test initialisation of AI Gym mesa environment with Keyword Arguments."""
        env = RetEnv(MockGymModelWithKwargs, {"arg_1": 1})

        assert env._model_cls is MockGymModelWithKwargs
        assert hasattr(env.model, "arg_1")
        assert env.model.arg_1 == 1  # type: ignore

    def test_update_action_space(self):
        """Test the update methodology for the action space."""
        self.env._update_action_space()
        action_space: spaces.Dict = self.env.action_space
        assert action_space[self.agent_1_id] is not None
        assert action_space[self.agent_2_id] is not None

    def test_update_observation_space(self):
        """Test the update methodology for the observation space."""
        self.env._update_observation_space()
        observation_space: spaces.Dict = self.env.observation_space
        assert observation_space[self.agent_1_id] is not None
        assert observation_space[self.agent_2_id] is not None

    def test_step_observation_do_nothing(self):
        """Test the observations in the environment after a do nothing action."""
        action: dict = {}

        action[self.agent_1_id] = self.do_none()
        action[self.agent_2_id] = self.do_none()
        observation, _, _, _ = self.env.step(action)
        assert len(observation[self.agent_1_id]["perceived_agents"]) == 1
        assert observation[self.agent_1_id]["perceived_agents"][0]["pos"][0] == 0
        assert observation[self.agent_1_id]["perceived_agents"][0]["pos"][1] == 0
        assert observation[self.agent_1_id]["perceived_agents"][0]["pos"][2] == 0
        assert observation[self.agent_1_id]["current_task"] == 0
        assert observation[self.agent_1_id]["current_task_parameters"] == {}
        assert observation[self.agent_1_id]["killed"] == 0

    def test_step_observation_do_something(self):
        """Test the observations in the environment after performing an action."""
        action: dict = {}

        action[self.agent_1_id] = self.do_move(destination=[1, 1, 1], tolerance=1)
        action[self.agent_2_id] = self.do_move_in_band(destination_xy=[1, 1], tolerance=1)
        observation, _, _, _ = self.env.step(action)
        assert len(observation[self.agent_1_id]["perceived_agents"]) == 1
        assert observation[self.agent_1_id]["perceived_agents"][0]["pos"][0] == 1
        assert observation[self.agent_1_id]["perceived_agents"][0]["pos"][1] == 1
        assert observation[self.agent_1_id]["perceived_agents"][0]["pos"][2] == 1
        assert observation[self.agent_1_id]["current_task"] == 2
        assert (
            observation[self.agent_1_id]["current_task_parameters"]["move"]["destination"][0] == 1
        )
        assert (
            observation[self.agent_1_id]["current_task_parameters"]["move"]["destination"][1] == 1
        )
        assert (
            observation[self.agent_1_id]["current_task_parameters"]["move"]["destination"][0] == 1
        )
        assert observation[self.agent_1_id]["current_task_parameters"]["move"]["tolerance"] == 1
        assert observation[self.agent_1_id]["killed"] == 0

    def test_step_reward_after_valid_action(self):
        """Test that the reward remains zero after completing a valid action."""
        action: dict = {}
        action[self.agent_1_id] = self.do_none()
        action[self.agent_2_id] = self.do_none()

        _, reward, _, _ = self.env.step(action)
        assert reward == 0

    def test_step_reward_after_invalid_action(self):
        """Test that the reward is reduced after performing an invalid action."""
        action: dict = {}

        action[self.agent_1_id] = self.do_move_in_band(destination_xy=[1, 1], tolerance=1)
        action[self.agent_2_id] = self.do_move(destination=[1, 1, 1], tolerance=1)

        _, reward, _, _ = self.env.step(action)
        assert reward == -2

    def test_step_done(self):
        """Test that the model is marked as done after completing all steps."""
        action: dict = {}
        action[self.agent_1_id] = self.do_none()
        action[self.agent_2_id] = self.do_none()

        for _ in range(24):  # model is set up to run for 24 hours in 1 hour time steps
            _, _, done, _ = self.env.step(action)
            self.assertFalse(done)

        _, _, done, _ = self.env.step(action)
        self.assertTrue(done)

    def test_step_info(self):
        """Test information collation following an action step."""
        action: dict = {}
        action[self.agent_1_id] = self.do_none()
        action[self.agent_2_id] = self.do_none()

        _, _, _, info = self.env.step(action)

        assert info["n_agents"] == 2
        assert info["current_time"] == self.model.start_time + self.model.time_step
        assert info["current_time"] == self.env.model.get_time()
        assert info["end_time"] == self.model.end_time

    def test_reset_position(self):
        """Test that reset reset's the agent's position."""
        agent_1 = self.get_agent_by_id(1)
        assert agent_1.pos[0] == 0
        assert agent_1.pos[1] == 0
        assert agent_1.pos[2] == 0

        payload = {"orders": Order(ImmediateTrigger(), MoveTask((20, 20, 20), 1), priority=1)}

        agent_1.communication_network.receiver.receive(agent_1, payload)

        self.env.model.step()
        assert agent_1.pos[0] == 20
        assert agent_1.pos[1] == 20
        assert agent_1.pos[2] == 20

        self.env.reset()

        agent_1 = self.get_agent_by_id(1)
        assert agent_1.pos[0] == 0
        assert agent_1.pos[1] == 0
        assert agent_1.pos[2] == 0

    def test_reset_killed_state(self):
        """Test that reset reset's the agent's killed status."""
        agent_1 = self.get_agent_by_id(1)

        self.env.model.step()
        agent_1.kill()

        self.assertTrue(agent_1.killed)

        observation = self.env.reset()

        agent_1 = self.get_agent_by_id(1)
        self.assertFalse(agent_1.killed)
        assert observation[self.agent_1_id]["killed"] == 0

    def test_reset_random_state(self):
        """Test that reset does not reset the model's randomiser."""
        self.env.model.reset_randomizer(1)
        state_1 = self.env.model.random.getstate()

        self.env.model.step()
        self.env.model.reset_randomizer(2)
        state_2 = self.env.model.random.getstate()

        self.env.reset()

        state_3 = self.env.model.random.getstate()

        assert state_1 != state_3
        assert state_2 == state_3

    def test_close(self):
        """Test that the environment can be closed cleanly."""
        # no assert statements as environment does not currently need to clean up
        # anything
        self.env.close()

    def test_seed(self):
        """Test the model's seed."""
        original_seed = self.env.model._seed
        assert self.env.seed(None) == [original_seed]
        assert self.env.seed(1) == [1]
        assert self.env.model._seed == 1

    def test_do_action_none(self):
        """Test agent's active orders following a do nothing action."""
        action: dict = {}

        agent_1 = self.get_agent_by_id(1)
        agent_2 = self.get_agent_by_id(2)

        assert agent_1.active_order is None
        assert agent_2.active_order is None

        action[self.agent_1_id] = self.do_none()
        action[self.agent_2_id] = self.do_none()
        self.env._do_action(action)
        self.env.model.step()
        assert agent_1.active_order is None
        assert agent_2.active_order is None

    def test_do_action_wait(self):
        """Test agent's active orders following a do wait action."""
        action: dict = {}

        agent_1 = self.get_agent_by_id(1)
        agent_2 = self.get_agent_by_id(2)

        assert agent_1.active_order is None
        assert agent_2.active_order is None

        action[self.agent_1_id] = self.do_wait(duration_seconds=1)
        action[self.agent_2_id] = self.do_none()
        self.env._do_action(action)
        self.env.model.step()
        assert agent_1.active_order is not None
        assert isinstance(agent_1.active_order._task, WaitTask)
        assert agent_1.active_order._task._duration == timedelta(seconds=1)
        assert agent_2.active_order is None

    def test_do_action_move(self):
        """Test agents positions following a move action."""
        action: dict = {}

        agent_1 = self.get_agent_by_id(1)
        agent_2 = self.get_agent_by_id(2)

        assert agent_1.active_order is None
        assert agent_2.active_order is None

        action[self.agent_1_id] = self.do_move(destination=[1, 1, 1], tolerance=1)
        action[self.agent_2_id] = self.do_none()
        self.env._do_action(action)
        self.env.model.step()
        assert agent_1.active_order is not None
        assert isinstance(agent_1.active_order._task, MoveTask)
        assert agent_1.active_order._task._destination[0] == 1
        assert agent_1.active_order._task._destination[1] == 1
        assert agent_1.active_order._task._destination[2] == 1
        assert agent_1.active_order._task._tolerance == 1
        assert agent_2.active_order is None

    def test_do_action_move_in_band(self):
        """Test agents positions following a move in band action."""
        action: dict = {}

        agent_1 = self.get_agent_by_id(1)
        agent_2 = self.get_agent_by_id(2)

        assert agent_1.active_order is None
        assert agent_2.active_order is None

        action[self.agent_1_id] = self.do_none()
        action[self.agent_2_id] = self.do_move_in_band(destination_xy=[1, 1], tolerance=1)
        self.env._do_action(action)
        self.env.model.step()
        assert agent_1.active_order is None
        assert agent_2.active_order is not None
        assert isinstance(agent_2.active_order._task, MoveInBandTask)
        assert agent_2.active_order._task._destination[0] == 1
        assert agent_2.active_order._task._destination[1] == 1
        assert agent_2.active_order._task._destination[2] == "test"
        assert agent_2.active_order._task._tolerance == 1
        assert agent_2.active_order._task._real_destination[0] == 1
        assert agent_2.active_order._task._real_destination[1] == 1
        assert agent_2.active_order._task._real_destination[2] == 1

    def test_do_action_communicate_worldview(self):
        """Test agents perceived world after a communicate worldview action."""
        action: dict = {}

        agent_1 = self.get_agent_by_id(1)
        agent_2 = self.get_agent_by_id(2)

        assert agent_1.active_order is None
        assert agent_2.active_order is None

        action[self.agent_1_id] = self.do_communicate_worldview()
        action[self.agent_2_id] = self.do_none()
        self.env._do_action(action)
        self.env.model.step()
        assert agent_1.active_order is not None
        assert isinstance(agent_1.active_order._task, CommunicateWorldviewTask)
        assert agent_2.active_order is None

    def test_do_action_disable_communication(self):
        """Test agents active orders following a disable communications action."""
        action: dict = {}

        agent_1 = self.get_agent_by_id(1)
        agent_2 = self.get_agent_by_id(2)

        assert agent_1.active_order is None
        assert agent_2.active_order is None

        action[self.agent_1_id] = self.do_disable_communication()
        action[self.agent_2_id] = self.do_none()
        self.env._do_action(action)
        self.env.model.step()
        assert agent_1.active_order is not None
        assert isinstance(agent_1.active_order._task, DisableCommunicationTask)
        assert agent_2.active_order is None

    def test_do_action_deploy_countermeasure(self):
        """Test agents active orders following a deploy countermeasure action."""
        action: dict = {}

        agent_1 = self.get_agent_by_id(1)
        agent_2 = self.get_agent_by_id(2)

        assert agent_1.active_order is None
        assert agent_2.active_order is None

        action[self.agent_1_id] = self.do_deploy_countermeasure()
        action[self.agent_2_id] = self.do_none()
        self.env._do_action(action)
        self.env.model.step()
        assert agent_1.active_order is not None
        assert isinstance(agent_1.active_order._task, DeployCountermeasureTask)
        assert agent_2.active_order is None

    def test_do_action_sense(self):
        """Test agents active orders following a sense action."""
        action: dict = {}

        agent_1 = self.get_agent_by_id(1)
        agent_2 = self.get_agent_by_id(2)

        assert agent_1.active_order is None
        assert agent_2.active_order is None

        action[self.agent_1_id] = self.do_sense()
        action[self.agent_2_id] = self.do_none()
        self.env._do_action(action)
        self.env.model.step()
        assert agent_1.active_order is not None
        assert isinstance(agent_1.active_order._task, SenseTask)
        assert agent_2.active_order is None

    def test_do_action_fire_at_target(self):
        """Test agents active orders following a fire at target action."""
        action: dict = {}

        agent_1 = self.get_agent_by_id(1)
        agent_2 = self.get_agent_by_id(2)

        assert agent_1.active_order is None
        assert agent_2.active_order is None

        action[self.agent_1_id] = self.do_fire_at_target()
        action[self.agent_2_id] = self.do_none()
        self.env._do_action(action)
        self.env.model.step()
        assert agent_1.active_order is not None
        assert isinstance(agent_1.active_order._task, FireAtTargetTask)
        assert agent_2.active_order is None

    def test_do_action_determine_target_and_fire(self):
        """Test agents active orders following a determine target and fire action."""
        action: dict = {}

        agent_1 = self.get_agent_by_id(1)
        agent_2 = self.get_agent_by_id(2)

        assert agent_1.active_order is None
        assert agent_2.active_order is None

        action[self.agent_1_id] = self.do_determine_target_and_fire()
        action[self.agent_2_id] = self.do_none()
        self.env._do_action(action)
        self.env.model.step()
        assert agent_1.active_order is not None
        assert isinstance(agent_1.active_order._task, DetermineTargetAndFireTask)
        assert agent_2.active_order is None

    def test_do_action_with_existing_order(self):
        """Test new orders are given the highest priority."""
        action: dict = {}

        agent_1 = self.get_agent_by_id(1)
        agent_2 = self.get_agent_by_id(2)

        assert len(agent_1._orders) == 0
        assert len(agent_2._orders) == 0

        agent_1.add_orders(Order(MockTrigger(), MockTask(), priority=10))

        assert len(agent_1._orders) == 1
        assert len(agent_2._orders) == 0

        action[self.agent_1_id] = self.do_wait(duration_seconds=1)
        action[self.agent_2_id] = self.do_none()
        self.env._do_action(action)

        assert len(agent_1._orders) == 2
        assert len(agent_2._orders) == 0
        assert agent_1._orders[1].priority == 11

    def test_do_action_invalid(self):
        """Test agents state after attempting to do an invalid action."""
        action: dict = {}

        agent_1 = self.get_agent_by_id(1)
        agent_2 = self.get_agent_by_id(2)

        assert agent_1.active_order is None
        assert agent_2.active_order is None

        action[self.agent_1_id] = self.do_none()
        action[self.agent_2_id] = self.do_disable_communication()
        self.env._do_action(action)
        self.env.model.step()
        assert agent_1.active_order is None
        assert agent_2.active_order is None
        assert self.env._n_impossible_tasks_requested == 1

    def test_do_action_not_all_supplied(self):
        """Test environment's handling of cases where actions are not defined."""
        action: dict = {}

        agent_1 = self.get_agent_by_id(1)
        agent_2 = self.get_agent_by_id(2)

        assert agent_1.active_order is None
        assert agent_2.active_order is None

        action[self.agent_1_id] = self.do_none()
        with self.assertRaises(KeyError) as e:
            self.env._do_action(action)
        self.assertEqual("2", str(e.exception))

    def test_get_observation(self):
        """Test environment's collection of observations."""
        obs = self.env._get_observation()

        assert len(obs) == 2

        assert len(obs[self.agent_1_id]["perceived_agents"]) == 1
        assert obs[self.agent_1_id]["perceived_agents"][0]["confidence"] == Confidence.KNOWN.value
        assert obs[self.agent_1_id]["current_task"] == 0
        assert obs[self.agent_1_id]["current_task_parameters"] == {}
        assert obs[self.agent_1_id]["possible_tasks"] == [
            True,
            True,
            True,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
        ]
        assert obs[self.agent_1_id]["killed"] == 0

        assert len(obs[self.agent_2_id]["perceived_agents"]) == 1
        assert obs[self.agent_2_id]["perceived_agents"][0]["confidence"] == Confidence.KNOWN.value
        assert obs[self.agent_2_id]["current_task"] == 0
        assert obs[self.agent_2_id]["current_task_parameters"] == {}
        assert obs[self.agent_2_id]["possible_tasks"] == [
            True,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
        ]
        assert obs[self.agent_2_id]["killed"] == 0

        agent_1 = self.get_agent_by_id(1)
        agent_1.perceived_world.add_acquisitions(
            PerceivedAgent(
                datetime(2020, 1, 1),
                Confidence.DETECT,
                (1, 1, 1),
                3,
                AgentCasualtyState.ALIVE,
                Affiliation.UNKNOWN,
                AgentType.AIR,
            )
        )
        obs = self.env._get_observation()
        assert len(obs[self.agent_1_id]["perceived_agents"]) == 2
        assert obs[self.agent_1_id]["perceived_agents"][0]["confidence"] == Confidence.KNOWN.value
        assert obs[self.agent_1_id]["perceived_agents"][1]["confidence"] == Confidence.DETECT.value

        agent_2 = self.get_agent_by_id(2)
        agent_2.kill()
        agent_2.active_order = Order(ImmediateTrigger(), WaitTask(timedelta(1)))
        obs = self.env._get_observation()
        assert obs[self.agent_2_id]["current_task"] == 1
        assert obs[self.agent_2_id]["killed"] == 1

    def test_get_perceived_agent_observation(self):
        """Test environment's collection of perceived agent observations."""
        perceived_agent_detect = PerceivedAgent(
            datetime(2020, 1, 1),
            Confidence.DETECT,
            (1, 1, 1),
            1,
            AgentCasualtyState.ALIVE,
            Affiliation.UNKNOWN,
            AgentType.AIR,
        )
        perceived_agent_recognise = PerceivedAgent(
            datetime(2020, 1, 2),
            Confidence.RECOGNISE,
            (2, 2, 2),
            2,
            AgentCasualtyState.ALIVE,
            Affiliation.HOSTILE,
            AgentType.AIR_DEFENCE,
        )
        perceived_agent_identify = PerceivedAgent(
            datetime(2020, 1, 3),
            Confidence.IDENTIFY,
            (3, 3, 3),
            3,
            AgentCasualtyState.ALIVE,
            Affiliation.NEUTRAL,
            AgentType.ARMOUR,
        )
        perceived_agent_known = PerceivedAgent(
            datetime(2020, 1, 4),
            Confidence.KNOWN,
            (4, 4, 4),
            4,
            AgentCasualtyState.ALIVE,
            Affiliation.FRIENDLY,
            AgentType.INFANTRY,
        )

        obs_detect = self.env._get_perceived_agent_observation(perceived_agent_detect)
        obs_recognise = self.env._get_perceived_agent_observation(perceived_agent_recognise)
        obs_identify = self.env._get_perceived_agent_observation(perceived_agent_identify)
        obs_known = self.env._get_perceived_agent_observation(perceived_agent_known)

        assert obs_detect["confidence"] == Confidence.DETECT.value
        assert obs_detect["pos"][0] == 1
        assert obs_detect["pos"][1] == 1
        assert obs_detect["pos"][2] == 1
        assert obs_detect["unique_id"] is None
        assert obs_detect["affiliation"] == Affiliation.UNKNOWN.value
        assert obs_detect["agent_type"] == AgentType.UNKNOWN.value

        assert obs_recognise["confidence"] == Confidence.RECOGNISE.value
        assert obs_recognise["pos"][0] == 2
        assert obs_recognise["pos"][1] == 2
        assert obs_recognise["pos"][2] == 2
        assert obs_recognise["unique_id"] is None
        assert obs_recognise["affiliation"] == Affiliation.UNKNOWN.value
        assert obs_recognise["agent_type"] == AgentType.AIR_DEFENCE.value

        assert obs_identify["confidence"] == Confidence.IDENTIFY.value
        assert obs_identify["pos"][0] == 3
        assert obs_identify["pos"][1] == 3
        assert obs_identify["pos"][2] == 3
        assert obs_identify["unique_id"] == 3
        assert obs_identify["affiliation"] == Affiliation.NEUTRAL.value
        assert obs_identify["agent_type"] == AgentType.ARMOUR.value

        assert obs_known["confidence"] == Confidence.KNOWN.value
        assert obs_known["pos"][0] == 4
        assert obs_known["pos"][1] == 4
        assert obs_known["pos"][2] == 4
        assert obs_known["unique_id"] == 4
        assert obs_known["affiliation"] == Affiliation.FRIENDLY.value
        assert obs_known["agent_type"] == AgentType.INFANTRY.value

    def test_get_task_id(self):
        """Test calculation of task IDs from task types."""
        assert self.env._get_task_id(WaitTask(timedelta(1))) == TaskIds.WAIT.value
        assert self.env._get_task_id(MoveTask((0, 0, 0), 1)) == TaskIds.MOVE.value
        assert (
            self.env._get_task_id(
                MoveInBandTask(
                    (0, 0, "test"),
                    [AbsoluteHeightBand("test", 1)],
                    self.model.space,  # type: ignore
                    1,
                )
            )
            == TaskIds.MOVE_IN_BAND.value
        )
        assert (
            self.env._get_task_id(CommunicateWorldviewTask())
            == TaskIds.COMMUNICATE_WORLD_VIEW.value
        )
        assert (
            self.env._get_task_id(DisableCommunicationTask()) == TaskIds.DISABLE_COMMUNICATION.value
        )

    def test_get_task_parameters(self):
        """Test extraction of task parameters."""
        agent = RetAgent(
            self.model,
            (0, 0, 0),
            "Agent 1",
            Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        agent.active_order = Order(MockTrigger(), MockTask())
        assert self.env._get_task_parameters(agent) == {}

        agent.active_order = Order(MockTrigger(), MoveTask((1, 1, 1), 1))
        assert self.env._get_task_parameters(agent) == {
            "move": {"destination": (1, 1, 1), "tolerance": 1}
        }

        agent.active_order = Order(
            MockTrigger(),
            MoveInBandTask((1, 1, "test"), [AbsoluteHeightBand("test", 1)], self.model.space, 1),
        )
        assert self.env._get_task_parameters(agent) == {
            "moveInBand": {"band": "test", "destination_xy": (1, 1), "tolerance": 1}
        }

        agent.active_order = Order(MockTrigger(), CommunicateWorldviewTask())
        assert self.env._get_task_parameters(agent) == {}

        agent.active_order = Order(MockTrigger(), DisableCommunicationTask())
        assert self.env._get_task_parameters(agent) == {}

        agent.active_order = Order(MockTrigger(), DeployCountermeasureTask())
        assert self.env._get_task_parameters(agent) == {}

        agent.active_order = Order(MockTrigger(), SenseTask(timedelta(seconds=1)))
        assert self.env._get_task_parameters(agent) == {}

        agent.active_order = Order(MockTrigger(), FireAtTargetTask((1, 1, 1), 1))
        assert self.env._get_task_parameters(agent) == {
            "fireAtTarget": {"target": (1, 1, 1), "rounds": 1}
        }

        agent.active_order = Order(MockTrigger(), DetermineTargetAndFireTask(1))
        assert self.env._get_task_parameters(agent) == {"determineTargetAndFire": {"rounds": 1}}

    def test_get_possible_tasks(self):
        """Test calculation of an agent's possible tasks, based on defined behaviour."""
        agent = RetAgent(
            self.model,
            (0, 0, 0),
            "Agent 1",
            Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )
        assert self.env._get_possible_tasks(agent) == [
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]

        agent.behaviour_pool.add_behaviour(MockWaitBehaviour())
        assert self.env._get_possible_tasks(agent) == [
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]

        agent.behaviour_pool.add_behaviour(MockMoveBehaviour())
        assert self.env._get_possible_tasks(agent) == [
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]

        agent.behaviour_pool.add_behaviour(MockMoveInBandBehaviour())
        assert self.env._get_possible_tasks(agent) == [
            True,
            True,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
        ]

        agent.behaviour_pool.add_behaviour(MockCommunicateWorldviewBehaviour())
        assert self.env._get_possible_tasks(agent) == [
            True,
            True,
            False,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
        ]

        agent.behaviour_pool.add_behaviour(MockDisableCommunicationBehaviour())
        assert self.env._get_possible_tasks(agent) == [
            True,
            True,
            False,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
        ]

        agent.behaviour_pool.add_behaviour(MockDeployCountermeasureBehaviour())
        assert self.env._get_possible_tasks(agent) == [
            True,
            True,
            False,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
        ]

        agent.behaviour_pool.add_behaviour(
            MockSenseBehaviour(
                time_before_first_sense=timedelta(seconds=0),
                time_between_senses=timedelta(seconds=5),
            )
        )
        assert self.env._get_possible_tasks(agent) == [
            True,
            True,
            False,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
        ]

        agent.behaviour_pool.add_behaviour(MockFireBehaviour())
        assert self.env._get_possible_tasks(agent) == [
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
        ]

    def test_get_reward(self):
        """Test calculation of reward based on number of impossible tasks."""
        assert self.env._get_reward() == 0
        n_impossible_tasks_requested = random.randint(1, 100)
        self.env._n_impossible_tasks_requested = n_impossible_tasks_requested
        assert self.env._get_reward() == -n_impossible_tasks_requested

    def test_get_info(self):
        """Test calculation of environment information."""
        info = self.env._get_info()
        assert info["n_agents"] == 2
        assert info["current_time"] == self.model.start_time
        assert info["current_time"] == self.env.model.get_time()
        assert info["end_time"] == self.model.end_time

        self.env.model.step()

        info = self.env._get_info()
        assert info["n_agents"] == 2
        assert info["current_time"] == self.model.start_time + self.model.time_step
        assert info["current_time"] == self.env.model.get_time()
        assert info["end_time"] == self.model.end_time

    def test_reset_counters(self):
        """Test that number of impossible tasks resets on counter reset."""
        assert self.env._n_impossible_tasks_requested == 0
        self.env._n_impossible_tasks_requested = 1
        self.env._reset_counters()
        assert self.env._n_impossible_tasks_requested == 0

    def test_invalid_move_behaviour_bands(self):
        """Test exception handling where agent has conflicting move behaviour."""
        model: MockGymModel = self.env.model  # type: ignore
        model.agent_1.behaviour_pool.add_behaviour(
            MockMoveInBandBehaviour([AbsoluteHeightBand("B", 20)])
        )

        model.agent_1.behaviour_pool.add_behaviour(
            MockMoveInBandBehaviour([AbsoluteHeightBand("A", 10)])
        )

        with self.assertRaises(ValueError) as context:
            self.env._get_task_parameters_space(model.agent_1)

        msg = (
            "Height bands across all MoveInBandBehaviour behaviours must have "
            + f"a consistent set of height bands for Agent {model.agent_1.unique_id}"
        )

        self.assertTrue(msg in str(context.exception))

    def get_agent_by_id(self, id: int) -> RetAgent:
        """Return agent in schedule matching a given ID.

        This assumes that the agent is present in the schedule, and is only present a
        single time.

        Args:
            id (int): ID to match

        Returns:
            RetAgent: Agent matching ID.
        """
        agent: RetAgent = [a for a in self.env.model.schedule.agents if a.unique_id == id][
            0
        ]  # type: ignore

        return agent

    def task_parameters(
        self,
        duration_seconds: int = 1,
        destination: Optional[list[float]] = None,
        tolerance: float = 1,
        destination_xy: Optional[list[float]] = None,
        band: int = 0,
        target: Optional[list[float]] = None,
        rounds: int = 1,
    ) -> dict[str, Any]:
        """Return an appropriately formatted task parameters dictionary.

        Args:
            duration_seconds (int): Duration. Defaults to 1.
            destination (Optional[list[float]]): 3D destination. If None or unspecified defaults to
                [1, 1, 1].
            tolerance (float): Tolerance. Defaults to 1.
            destination_xy (Optional[list[float]]): 2D destination. If None or unspecified, defaults
                to [1, 1].
            band (int): Band. Defaults to 0.
            target (Optional[list[float]]): Target of the task. If None or unspecified, defaults to
                (1, 1, 1).
            rounds (int): Number of rounds to fire. Defaults to 1.

        Returns:
            dict[str, Any]: Task parametrisation
        """
        if destination is None:
            destination = [1, 1, 1]

        if destination_xy is None:
            destination_xy = [1, 1]

        if target is None:
            target = [1, 1, 1]

        return {
            "wait": {"duration_seconds": np.array(duration_seconds)},
            "move": {
                "destination": np.array(destination),
                "tolerance": np.array(tolerance),
            },
            "moveInBand": {
                "destination_xy": np.array(destination_xy),
                "band": band,
                "tolerance": np.array(tolerance),
            },
            "fireAtTarget": {
                "target": np.array(target),
                "rounds": rounds,
            },
            "determineTargetAndFire": {"rounds": rounds},
        }

    def do_none(self) -> dict[str, Any]:
        """Return a do nothing parametrisation.

        Returns:
            dict[str, Any]: AI Gym representation of do nothing
        """
        return {
            "new_task": TaskIds.NONE.value,
            "new_task_parameters": self.task_parameters(),
        }

    def do_wait(self, duration_seconds: int) -> dict[str, Any]:
        """Return a do wait parametrisation.

        Args:
            duration_seconds (int): Wait duration

        Returns:
            dict[str, Any]: AI Gym representation of waiting
        """
        return {
            "new_task": TaskIds.WAIT.value,
            "new_task_parameters": self.task_parameters(duration_seconds=duration_seconds),
        }

    def do_move(self, destination: list[float], tolerance: float) -> dict[str, Any]:
        """Return a move parametrisation.

        Args:
            destination (list[float]): 3D destination.
            tolerance (float): tolerance

        Returns:
            dict[str, Any]: AI Gym representation of moving.
        """
        return {
            "new_task": TaskIds.MOVE.value,
            "new_task_parameters": self.task_parameters(
                destination=destination, tolerance=tolerance
            ),
        }

    def do_move_in_band(self, destination_xy: list[float], tolerance: float) -> dict[str, Any]:
        """Return a move in band parametrisation.

        Args:
            destination_xy (list[float]): 2D destination
            tolerance (float): tolerance

        Returns:
            dict[str, Any]: AI Gym representation of moving in a band.
        """
        return {
            "new_task": TaskIds.MOVE_IN_BAND.value,
            "new_task_parameters": self.task_parameters(
                destination_xy=destination_xy, tolerance=tolerance
            ),
        }

    def do_communicate_worldview(self) -> dict[str, Any]:
        """Return a communicate worldview parametrisation.

        Returns:
            dict[str, Any]: AI Gym representation of communicating worldview
        """
        return {
            "new_task": TaskIds.COMMUNICATE_WORLD_VIEW.value,
            "new_task_parameters": self.task_parameters(),
        }

    def do_disable_communication(self) -> dict[str, Any]:
        """Return a disable communication parametrisation.

        Returns:
            dict[str, Any]: AI Gym representation of disabling communication
        """
        return {
            "new_task": TaskIds.DISABLE_COMMUNICATION.value,
            "new_task_parameters": self.task_parameters(),
        }

    def do_deploy_countermeasure(self) -> dict[str, Any]:
        """Return a deploy countermeasure parametrisation.

        Returns:
            dict[str, Any]: AI Gym representation of deploy countermeasure
        """
        return {
            "new_task": TaskIds.DEPLOY_COUNTERMEASURE.value,
            "new_task_parameters": self.task_parameters(),
        }

    def do_sense(self) -> dict[str, Any]:
        """Return a sense parametrisation.

        Returns:
            dict[str, Any]: AI Gym representation of sense
        """
        return {
            "new_task": TaskIds.SENSE.value,
            "new_task_parameters": self.task_parameters(),
        }

    def do_fire_at_target(self) -> dict[str, Any]:
        """Return a fire at target parametrisation.

        Returns:
            dict[str, Any]: AI Gym representation of fire at target
        """
        return {
            "new_task": TaskIds.FIRE_AT_TARGET.value,
            "new_task_parameters": self.task_parameters(),
        }

    def do_determine_target_and_fire(self) -> dict[str, Any]:
        """Return a determine target and fire parametrisation.

        Returns:
            dict[str, Any]: AI Gym representation of determine target and fire
        """
        return {
            "new_task": TaskIds.DETERMINE_TARGET_AND_FIRE.value,
            "new_task_parameters": self.task_parameters(),
        }
