"""AI Gym wrapper for ret."""
from __future__ import annotations

from datetime import timedelta
from enum import Enum
from operator import attrgetter
from typing import TYPE_CHECKING

import gym
import numpy as np
from gym import spaces
from mesa_ret.agents.affiliation import Affiliation
from mesa_ret.agents.agenttype import AgentType
from mesa_ret.behaviours.behaviourpool import BehaviourHandlers
from mesa_ret.behaviours.move import MoveInBandBehaviour
from mesa_ret.orders.order import Order
from mesa_ret.orders.tasks.communicate import CommunicateWorldviewTask
from mesa_ret.orders.tasks.deploycountermeasure import DeployCountermeasureTask
from mesa_ret.orders.tasks.disablecommunication import DisableCommunicationTask
from mesa_ret.orders.tasks.fire import DetermineTargetAndFireTask, FireAtTargetTask
from mesa_ret.orders.tasks.move import MoveInBandTask, MoveTask
from mesa_ret.orders.tasks.sense import SenseTask
from mesa_ret.orders.tasks.wait import WaitTask
from mesa_ret.orders.triggers.immediate import ImmediateTrigger
from mesa_ret.sensing.perceivedworld import Confidence

if TYPE_CHECKING:
    from typing import Any, Callable, Optional, Type

    from mesa_ret.agents.agent import RetAgent
    from mesa_ret.model import RetModel
    from mesa_ret.orders.order import Task
    from mesa_ret.sensing.perceivedworld import PerceivedAgent
    from mesa_ret.space.heightband import HeightBand


class TaskIds(Enum):
    """Task-type specific keys in action dictionary."""

    NONE = 0
    WAIT = 1
    MOVE = 2
    MOVE_IN_BAND = 3
    COMMUNICATE_WORLD_VIEW = 4
    DISABLE_COMMUNICATION = 5
    DEPLOY_COUNTERMEASURE = 6
    SENSE = 7
    FIRE_AT_TARGET = 8
    DETERMINE_TARGET_AND_FIRE = 9


class RetEnv(gym.Env):
    """An OpenAI Gym for a RetModel."""

    metadata = {"render.modes": ["human"]}

    # this controls the indices of the true/false conditions returned from
    # _get_possible_tasks
    _task_type_id_map: dict[Type, int] = {
        WaitTask: TaskIds.WAIT.value,
        MoveTask: TaskIds.MOVE.value,
        MoveInBandTask: TaskIds.MOVE_IN_BAND.value,
        CommunicateWorldviewTask: TaskIds.COMMUNICATE_WORLD_VIEW.value,
        DisableCommunicationTask: TaskIds.DISABLE_COMMUNICATION.value,
        DeployCountermeasureTask: TaskIds.DEPLOY_COUNTERMEASURE.value,
        SenseTask: TaskIds.SENSE.value,
        FireAtTargetTask: TaskIds.FIRE_AT_TARGET.value,
        DetermineTargetAndFireTask: TaskIds.DETERMINE_TARGET_AND_FIRE.value,
    }

    _n_impossible_tasks_requested: int

    def __init__(
        self, model_cls: Callable[..., RetModel], model_params: Optional[dict[str, Any]] = None
    ):
        """Initialise the gym with a given RetModel.

        Args:
            model_cls (Callable[..., RetModel]): The model to use for the gym
            model_params (Optional[dict[str, Any]]): Model input parameters. If None, defaults to an
                empty dictionary. Defaults to None
        """
        super().__init__()
        self._model_cls = model_cls
        if model_params is not None:
            self._model_kwargs = model_params
        else:
            self._model_kwargs = {}

        self.model = self._model_cls(**self._model_kwargs)
        self.behaviour_handlers = BehaviourHandlers()

        self._update_action_space()
        self._update_observation_space()

        self._n_impossible_tasks_requested = 0

    def _update_action_space(self) -> None:
        """Update the action space based on the current state of the model."""
        space: dict = {}

        agent: RetAgent
        for agent in self.model.schedule.agents:  # type:ignore

            space[agent.unique_id] = spaces.Dict(
                {
                    "new_task": self._get_task_space(),
                    "new_task_parameters": self._get_task_parameters_space(agent),
                }
            )

        self.action_space = spaces.Dict(space)

    def _update_observation_space(self) -> None:
        """Update observation space based on the current state of the model.

        This is currently only called on initialisation but can change shape during a
        model run (e.g. the number of perceived agents can change). If an updated
        observation space is needed then this method can be called again. This is not
        called after each model step as it is an expensive operation and slows the
        model significantly.
        """
        space: dict = {}

        perceived_agent_space = spaces.Dict(
            {
                "confidence": spaces.Discrete(len(Confidence)),
                "pos": spaces.Box(
                    low=np.array(  # type: ignore
                        [self.model.space.x_min, self.model.space.y_min, -np.inf],
                        dtype=np.float32,
                    ),
                    high=np.array(  # type: ignore
                        [self.model.space.x_max, self.model.space.y_max, np.inf],
                        dtype=np.float32,
                    ),
                ),
                "unique_id": spaces.Discrete(self.model.current_id),
                "affiliation": spaces.Discrete(len(Affiliation)),
                "agent_type": spaces.Discrete(len(AgentType)),
            }
        )

        agent: RetAgent
        for agent in self.model.schedule.agents:  # type:ignore
            n_perceived_agents = len(agent.perceived_world.get_perceived_agents())
            space[agent.unique_id] = spaces.Dict(
                {
                    "perceived_agents": spaces.Tuple(
                        (perceived_agent_space for _ in range(n_perceived_agents))
                    ),
                    "current_task": self._get_task_space(),
                    "current_task_parameters": self._get_task_parameters_space(agent),
                    "possible_tasks": spaces.MultiBinary(len(self._task_type_id_map) + 1),
                    "killed": spaces.Discrete(2),
                }
            )

        self.observation_space = spaces.Dict(space)

    def _get_task_space(self) -> spaces.Discrete:
        """Get the task space.

        Returns:
            spaces.Discrete: the task space
        """
        return spaces.Discrete(len(self._task_type_id_map) + 1)

    def _get_task_parameters_space(self, agent: RetAgent) -> spaces.Dict:
        """Get the task parameter space for a given agent.

        Args:
            agent (RetAgent): the agent

        Returns:
            spaces.Dict: the task parameter space

        Raises:
            ValueError: Inconsistent height bands within MoveInBandBehaviour
        """
        move_in_band_behaviours: list[MoveInBandBehaviour] = agent.behaviour_pool.expose_behaviour(
            "step", MoveInBandBehaviour
        )  # type: ignore

        bands = [sorted(b.height_bands, key=attrgetter("height")) for b in move_in_band_behaviours]

        # lists are not hashable but tuples are - hence the bands are converted to a
        # tuple to facilitate checking whether there is a single, consistent set of
        # bands across all MoveInBandBehaviour behaviours for the given agent
        if len(set(tuple(i) for i in bands)) > 1:
            msg = (
                "Height bands across all MoveInBandBehaviour behaviours must have "
                + f"a consistent set of height bands for Agent {agent.unique_id}"
            )

            raise ValueError(msg)

        n_bands = max([len(b) for b in bands] + [1])

        max_steps = int((self.model.end_time - self.model.start_time) / self.model.time_step)

        return spaces.Dict(
            {
                "wait": spaces.Dict({"duration_seconds": spaces.Box(low=0, high=np.inf, shape=())}),
                "sense": spaces.Dict(
                    {"duration_seconds": spaces.Box(low=0, high=np.inf, shape=())}
                ),
                "move": spaces.Dict(
                    {
                        "destination": spaces.Box(
                            low=np.array(  # type: ignore
                                [
                                    self.model.space.x_min,
                                    self.model.space.y_min,
                                    -np.inf,
                                ],
                                dtype=np.float32,
                            ),
                            high=np.array(  # type: ignore
                                [
                                    self.model.space.x_max,
                                    self.model.space.y_max,
                                    np.inf,
                                ],
                                dtype=np.float32,
                            ),
                        ),
                        "tolerance": spaces.Box(low=0, high=np.inf, shape=()),
                    }
                ),
                "moveInBand": spaces.Dict(
                    {
                        "destination_xy": spaces.Box(
                            low=np.array(  # type: ignore
                                [self.model.space.x_min, self.model.space.y_min],
                                dtype=np.float32,
                            ),
                            high=np.array(  # type: ignore
                                [self.model.space.x_max, self.model.space.y_max],
                                dtype=np.float32,
                            ),
                        ),
                        "band": spaces.Discrete(n_bands),
                        "tolerance": spaces.Box(
                            low=0,
                            high=np.inf,
                            shape=(),
                        ),
                    }
                ),
                "fireAtTarget": spaces.Dict(
                    {
                        "target": spaces.Box(
                            low=np.array(  # type: ignore
                                [
                                    self.model.space.x_min,
                                    self.model.space.y_min,
                                    -np.inf,
                                ],
                                dtype=np.float32,
                            ),
                            high=np.array(  # type: ignore
                                [
                                    self.model.space.x_max,
                                    self.model.space.y_max,
                                    np.inf,
                                ],
                                dtype=np.float32,
                            ),
                        ),
                        "rounds": spaces.Discrete(max_steps),
                    }
                ),
                "determineTargetAndFire": spaces.Dict({"rounds": spaces.Discrete(max_steps)}),
            }
        )

    def step(self, action) -> tuple[object, float, bool, dict[str, Any]]:
        """Run one time step of the environment's dynamics.

        When end of episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Args:
            action (object): an action provided by the agent

        Returns:
            output (tuple[object, float, bool, dict[str, Any]]): observations, reward,
                completion status and info messages from the step:
                    - observations: agent's observation of the current environment
                    - reward: amount of reward returned after previous action
                    - done: whether the episode has ended, in which case further step()
                        calls will return undefined results
                    - info: contains auxiliary diagnostic information (helpful for
                        debugging, and sometimes learning)
        """
        self._do_action(action)

        self.model.step()

        observation = self._get_observation()
        reward = self._get_reward()
        done = not self.model.running
        info = self._get_info()

        self._reset_counters()

        return observation, reward, done, info

    def reset(self):
        """Reset the environment to an initial state and returns an initial observation.

        Note that this function does not reset the environment's random number generator
        so that each call of `reset()` yields an environment suitable for a new episode,
        independent of previous episodes.

        Returns:
            observation (object): the initial observation.
        """
        state = self.model.random.getstate()
        self.model = self._model_cls(**self._model_kwargs)
        self.model.random.setstate(state)
        return self._get_observation()

    def render(self, mode: str = "human"):
        """Render the environment.

        Args:
            mode (str): Rendering mode
        """
        # TODO: fill in this method for human render
        super().render(mode=mode)

    def close(self):  # pragma: no cover
        """Perform any necessary cleanup."""
        pass

    def seed(self, seed: Optional[int] = None) -> list[int]:
        """Set the seed for this env's random number generator.

        Args:
            seed (Optional[int]): Seed for random number generator. Defaults to None,
                where run cannot be repeated.

        Returns:
            list[int]: The seed used in this env's random number generator (in a list
                as per parent class).
        """
        self.model.reset_randomizer(seed)

        return [self.model._seed]

    def _do_action(self, action) -> None:  # noqa: C901
        """Add the tasks defined in the action to the agents orders.

        Args:
            action (object): The action to convert into tasks/orders, this has the form
                as defined in the action space
        """
        agent: RetAgent
        for agent in self.model.schedule.agents:  # type:ignore
            new_task_id = action[agent.unique_id]["new_task"]

            if not self._get_possible_tasks(agent)[new_task_id]:
                self._n_impossible_tasks_requested += 1
                continue

            parameters = action[agent.unique_id]["new_task_parameters"]

            new_task: Optional[Task] = None
            if new_task_id == TaskIds.WAIT.value:
                duration = timedelta(seconds=parameters["wait"]["duration_seconds"].item())
                new_task = WaitTask(duration)

            elif new_task_id == TaskIds.MOVE.value:
                destination = (
                    parameters["move"]["destination"][0].item(),
                    parameters["move"]["destination"][1].item(),
                    parameters["move"]["destination"][2].item(),
                )
                tolerance = parameters["move"]["tolerance"].item()
                new_task = MoveTask(destination, tolerance)

            elif new_task_id == TaskIds.MOVE_IN_BAND.value:
                # It is safe to do this indexing, as we have previously checked that
                # there is at least one MoveInBandBehaviour, and that all
                # MoveInBandBehaviours for a given agent have a consistent set of bands

                behaviour: MoveInBandBehaviour = agent.behaviour_pool.expose_behaviour(
                    "step", MoveInBandBehaviour
                )[
                    0
                ]  # type: ignore

                bands: list[HeightBand] = behaviour.height_bands

                z = bands[parameters["moveInBand"]["band"]].name
                destination = (
                    parameters["moveInBand"]["destination_xy"][0].item(),
                    parameters["moveInBand"]["destination_xy"][1].item(),
                    z,
                )
                tolerance = parameters["moveInBand"]["tolerance"].item()
                new_task = MoveInBandTask(
                    destination, bands, self.model.space, tolerance  # type:ignore
                )

            elif new_task_id == TaskIds.COMMUNICATE_WORLD_VIEW.value:
                new_task = CommunicateWorldviewTask()

            elif new_task_id == TaskIds.DISABLE_COMMUNICATION.value:
                new_task = DisableCommunicationTask()

            elif new_task_id == TaskIds.DEPLOY_COUNTERMEASURE.value:
                new_task = DeployCountermeasureTask()

            elif new_task_id == TaskIds.SENSE.value:
                duration = timedelta(seconds=parameters["wait"]["duration_seconds"].item())
                new_task = SenseTask(duration)

            elif new_task_id == TaskIds.FIRE_AT_TARGET.value:
                target = (
                    parameters["fireAtTarget"]["target"][0].item(),
                    parameters["fireAtTarget"]["target"][1].item(),
                    parameters["fireAtTarget"]["target"][2].item(),
                )
                rounds = parameters["fireAtTarget"]["rounds"]
                new_task = FireAtTargetTask(target, rounds)

            elif new_task_id == TaskIds.DETERMINE_TARGET_AND_FIRE.value:
                rounds = parameters["determineTargetAndFire"]["rounds"]
                new_task = DetermineTargetAndFireTask(rounds)

            if new_task is not None:
                priority = 0
                if len(agent._orders) > 0:
                    priority = max(o.priority for o in agent._orders) + 1

                new_order = Order(ImmediateTrigger(), new_task, priority=priority)

                payload = {"orders": new_order}

                agent.communication_network.receiver.receive(agent, payload)

    def _get_observation(self) -> dict[int, Any]:
        """Get the observation based on the current state of the model.

        Returns:
            dict[int, Any]: The observation of the current state of the model, has the
                form as defined in the observation space
        """
        observation: dict = {}
        agent: RetAgent
        for agent in self.model.schedule.agents:  # type:ignore

            observation[agent.unique_id] = {
                "perceived_agents": tuple(
                    self._get_perceived_agent_observation(pa)
                    for pa in agent.perceived_world.get_perceived_agents()
                ),
                "current_task": self._get_task_id(agent.active_order._task)
                if agent.active_order
                else 0,
                "current_task_parameters": self._get_task_parameters(agent),
                "possible_tasks": self._get_possible_tasks(agent),
                "killed": int(agent.killed),
            }

        return observation

    def _get_perceived_agent_observation(self, perceived_agent: PerceivedAgent) -> dict[str, Any]:
        """Get the observation of a given PerceivedAgent.

        Args:
            perceived_agent (PerceivedAgent): The PerceivedAgent to observe

        Returns:
            dict[str, Any]: The observation of the PerceivedAgent
        """
        observation: dict = {
            "confidence": perceived_agent.confidence.value,
            "pos": perceived_agent.location,
            "unique_id": perceived_agent.unique_id,
            "affiliation": perceived_agent.affiliation.value,
            "agent_type": perceived_agent.agent_type.value,
        }

        return observation

    def _get_task_id(self, task: Task) -> int:
        """Return the task id for a given task.

        Args:
            task (Task): The task

        Returns:
            int: The id of the task in the observation/action spaces
        """
        return self._task_type_id_map.get(type(task), 0)

    def _get_task_parameters(self, agent: RetAgent) -> dict[str, Any]:  # noqa: C901
        task_parameters = {}
        task_id = 0
        if agent.active_order:
            task = agent.active_order._task
            task_id = self._get_task_id(task)

            if task_id == TaskIds.WAIT.value:
                task_parameters = {
                    "wait": {"duration_seconds": task._duration.seconds}  # type: ignore
                }

            elif task_id == TaskIds.MOVE.value:
                task_parameters = {
                    "move": {
                        "destination": task._destination,  # type: ignore
                        "tolerance": task._tolerance,  # type: ignore
                    }
                }

            elif task_id == TaskIds.MOVE_IN_BAND.value:
                task_parameters = {
                    "moveInBand": {
                        "band": task._destination[2],  # type: ignore
                        "destination_xy": (
                            task._destination[0],  # type: ignore
                            task._destination[1],  # type: ignore
                        ),
                        "tolerance": task._tolerance,  # type: ignore
                    }
                }

            elif task_id == TaskIds.COMMUNICATE_WORLD_VIEW.value:
                task_parameters = {}

            elif task_id == TaskIds.DISABLE_COMMUNICATION.value:
                task_parameters = {}

            elif task_id == TaskIds.DEPLOY_COUNTERMEASURE.value:
                task_parameters = {}

            elif task_id == TaskIds.SENSE.value:
                task_parameters = {}

            elif task_id == TaskIds.FIRE_AT_TARGET.value:
                task_parameters = {
                    "fireAtTarget": {
                        "target": task.target,  # type: ignore
                        "rounds": task.rounds_to_fire,  # type: ignore
                    }
                }

            elif task_id == TaskIds.DETERMINE_TARGET_AND_FIRE.value:
                task_parameters = {
                    "determineTargetAndFire": {"rounds": task.rounds_to_fire}  # type: ignore
                }

        return task_parameters

    def _get_possible_tasks(self, agent: RetAgent) -> list[bool]:
        """Get the possible tasks for a given agent.

        Args:
            agent (RetAgent): The agent

        Returns:
            list[bool]: A list of bools where each element specifies if the given agent
                can perfrom the task associated with that index position
        """
        can_none: bool = True
        can_wait: bool = (
            len(
                agent.behaviour_pool.expose_behaviour(
                    agent.behaviour_handlers.wait_handler,
                    agent.behaviour_handlers.wait_type,
                )
            )
            > 0
        )
        can_move_in_band: bool = (
            len(
                agent.behaviour_pool.expose_behaviour(
                    agent.behaviour_handlers.move_handler, MoveInBandBehaviour
                )
            )
            > 0
        )
        can_move: bool = (
            len(
                agent.behaviour_pool.expose_behaviour(
                    agent.behaviour_handlers.move_handler,
                    agent.behaviour_handlers.move_type,
                )
            )
            > 0
            and not can_move_in_band
        )

        can_communicate_world_view: bool = (
            len(
                agent.behaviour_pool.expose_behaviour(
                    agent.behaviour_handlers.communicate_worldview_handler,
                    agent.behaviour_handlers.communicate_worldview_type,
                )
            )
            > 0
        )

        can_disable_communication: bool = (
            len(
                agent.behaviour_pool.expose_behaviour(
                    agent.behaviour_handlers.disable_communication_handler,
                    agent.behaviour_handlers.disable_communication_type,
                )
            )
            > 0
        )

        can_deploy_countermeasure: bool = (
            len(
                agent.behaviour_pool.expose_behaviour(
                    agent.behaviour_handlers.deploy_countermeasure_handler,
                    agent.behaviour_handlers.deploy_countermeasure_type,
                )
            )
            > 0
        )

        can_sense: bool = (
            len(
                agent.behaviour_pool.expose_behaviour(
                    agent.behaviour_handlers.sense_handler,
                    agent.behaviour_handlers.sense_type,
                )
            )
            > 0
        )

        can_fire_at_target: bool = (
            len(
                agent.behaviour_pool.expose_behaviour(
                    agent.behaviour_handlers.fire_handler,
                    agent.behaviour_handlers.fire_type,
                )
            )
            > 0
        )

        can_determine_target_and_fire: bool = can_fire_at_target

        available_behaviours: list[bool] = list()
        available_behaviours.append(can_none)
        available_behaviours.append(can_wait)
        available_behaviours.append(can_move)
        available_behaviours.append(can_move_in_band)
        available_behaviours.append(can_communicate_world_view)
        available_behaviours.append(can_disable_communication)
        available_behaviours.append(can_deploy_countermeasure)
        available_behaviours.append(can_sense)
        available_behaviours.append(can_fire_at_target)
        available_behaviours.append(can_determine_target_and_fire)

        return available_behaviours

    def _get_reward(self) -> float:
        """Return the reward for the given step.

        This is currently a placeholder and will be part of developing the RL agent

        Returns:
            float: The reward
        """
        return -self._n_impossible_tasks_requested

    def _get_info(self) -> dict[str, Any]:
        """Return the info.

        Returns:
            dict[str, Any]: The info
        """
        return {
            "n_agents": len(self.model.schedule.agents),
            "current_time": self.model.get_time(),
            "end_time": self.model.end_time,
        }

    def _reset_counters(self) -> None:
        """Reset all counters that have been used."""
        self._n_impossible_tasks_requested = 0
