"""Move task."""

from __future__ import annotations

import cmath
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, cast

import numpy as np

from ret.agents.groupagent import GroupAgent
from ret.orders.order import Order, Task
from ret.orders.triggers.immediate import ImmediateTrigger
from ret.types import Coordinate3dBand  # noqa: TC001

if TYPE_CHECKING:
    from random import Random
    from typing import Optional, Tuple

    from ret.agents.agent import RetAgent
    from ret.formations import Formation
    from ret.orders.order import Trigger
    from ret.sensing.perceivedworld import PerceivedAgentFilter
    from ret.space.feature import Area
    from ret.space.heightband import HeightBand
    from ret.space.space import ContinuousSpaceWithTerrainAndCulture3d
    from ret.types import Coordinate, Coordinate2d, Coordinate2dOr3d, Coordinate3d


class MoveTask(Task):
    """Task for an agent to move to a given location."""

    _destination: Coordinate
    _tolerance: float

    def __init__(self, destination: Coordinate, tolerance: float, log: bool = True) -> None:
        """Create task.

        Args:
            destination (Coordinate): The destination
            tolerance (float): The tolerance
            log (bool): whether to log or not. Defaults to True.
        """
        super().__init__(log)
        self._destination = destination
        self._tolerance = tolerance

    def __str__(self):
        """Output a human readable name for the task.

        Returns:
            string: brief description of the task
        """
        return "Move Task"

    def _do_task_step(self, doer: RetAgent) -> None:
        """Move one time steps worth towards the destination.

        Args:
            doer (RetAgent): the agent doing the task
        """
        doer.move_step(self._destination)

    def _is_task_complete(self, doer: RetAgent) -> bool:
        """Check if the movement is complete.

        Args:
            doer (RetAgent): the agent doing the task

        Returns:
            bool: true when the agent is within the tolerance of the destination, false
                otherwise
        """
        doer_pos = doer.perceived_world.get_agent_pos(doer.unique_id)
        if doer_pos is None:
            warnings.warn(
                f"Agent {doer.unique_id} cannot find it's own position in it's perceived world so "
                "move task cannot be checked or completed.",
                stacklevel=2,
            )
            return False
        distance = doer.perceived_world.get_distance(doer_pos, self._destination)
        return distance <= self._tolerance

    def _get_log_message(self):
        """Get the log message.

        Returns:
            string: The log message
        """
        return f"Destination: {self._destination}"

    def get_new_instance(self) -> MoveTask:
        """Return a new instance of a functionally identical task.

        Returns:
            MoveTask: New instance of the task
        """
        return MoveTask(destination=self._destination, tolerance=self._tolerance, log=self._log)

    def update_destination(self, x_update: float = 0, y_update: float = 0) -> None:
        """Update the move destination in 2D.

        Args:
            x_update (float): Amount to increase the x value of the
                destination coordinate by. Defaults to 0.
            y_update (float): Amount to increase the y value of the
                destination coordinate by. Defaults to 0.
        """
        self._destination = self._offset_destination(self._destination, x_update, y_update)

    @staticmethod
    def _offset_destination(
        initial_destination: Coordinate,
        x_increment: float,
        y_increment: float,
    ) -> Coordinate:
        new_x_destination = initial_destination[0] + x_increment
        new_y_destination = initial_destination[1] + y_increment
        if len(initial_destination) == 2:
            return (new_x_destination, new_y_destination)
        return (new_x_destination, new_y_destination, initial_destination[2])  # type: ignore


class MoveAlongPathTask(MoveTask):
    """Task for an agent to move to a given location along a path.

    This assumes that the agent moves from one point along the path to the next, until it reaches
    its destination. Once it reaches a point along the path, it will pause until the next time
    step before moving on to the next point. Hence the time step should be small enough so that
    this is appropriate (or the spatial resolution of the pathfinder low enough that this is not
    an issue).
    """

    def __init__(self, destination: Coordinate2dOr3d, tolerance: float, log: bool = True) -> None:
        """Create task.

        Args:
            destination (Coordinate2dOr3d): The destination
            tolerance (float): The tolerance
            log (bool): whether to log or not. Defaults to True.
        """
        super().__init__(destination, tolerance, log)
        self._path: list[Coordinate3d] = []
        self._destination: Coordinate2dOr3d = destination
        self._current_path_target_index = 0

    def __str__(self):
        """Output a human readable name for the task.

        Returns:
            string: brief description of the task
        """
        return "Move Along Path Task"

    def _do_task_step(self, doer: RetAgent) -> None:
        """Move one time steps worth towards the destination.

        Args:
            doer (RetAgent): the agent doing the task
        """
        # Check that a task has a path, if not, generate one.
        if len(self._path) == 0:
            if doer.pathfinder is None:
                raise ValueError("Pathfinding task has been given to agent without a pathfinder.")
            start_coordinate = doer.perceived_world.get_agent_pos(doer.unique_id)
            if start_coordinate is None:
                warnings.warn(
                    f"Agent {doer.unique_id} cannot find it's own position in it's perceived world "
                    "so move along path task cannot be checked or completed.",
                    stacklevel=2,
                )
                return
            self._path = doer.pathfinder.find_path(
                start_coordinate,
                self._destination,
            )

        if self._current_path_target_index != (len(self._path) - 1):
            current_path_target = doer.model.space.get_coordinate_in_correct_dimension(
                self._path[self._current_path_target_index]
            )

            doer_pos = doer.perceived_world.get_agent_pos(doer.unique_id)
            if doer_pos is None:
                warnings.warn(
                    f"Agent {doer.unique_id} cannot find it's own position in it's perceived world "
                    "so move along path task cannot be checked or completed.",
                    stacklevel=2,
                )
                return
            doer_current_location = doer.model.space.get_coordinate_in_correct_dimension(doer_pos)

            distance_to_current_path_target = doer.perceived_world.get_distance(
                doer_current_location,
                current_path_target,
            )

            if distance_to_current_path_target <= self._tolerance:
                self._current_path_target_index += 1

            doer.move_step(self._path[self._current_path_target_index])

        else:
            # if agent has reached end of path, move to destination directly
            doer.move_step(self._destination)

    def get_new_instance(self) -> MoveAlongPathTask:
        """Return a new instance of a functionally identical task.

        Returns:
            MoveTask: New instance of the task
        """
        return MoveAlongPathTask(
            destination=self._destination, tolerance=self._tolerance, log=self._log
        )


class MoveInBandTask(MoveTask):
    """Task for an agent to move to a given location within a HeightBand."""

    _bands: list[HeightBand]
    _space: ContinuousSpaceWithTerrainAndCulture3d

    _real_destination: Coordinate3d

    def __init__(
        self,
        destination: Coordinate3dBand,
        bands: list[HeightBand],
        space: ContinuousSpaceWithTerrainAndCulture3d,
        tolerance: float,
        log: bool = True,
    ) -> None:
        """Create task.

        Args:
            destination (Coordinate3dBand): The destination
            bands (list[HeightBand]): List of HeightBands
            space (ContinuousSpaceWithTerrainAndCulture3d): The space the bands are in
            tolerance (float): The tolerance
            log (bool): Whether to log or not. Defaults to True.
        """
        super().__init__(destination, tolerance, log)
        self._bands = bands
        self._space = space
        self._real_destination = space.get_coordinate_3d(destination, bands)

    def __str__(self):
        """Output a human readable name for the task.

        Returns:
            string: brief description of the task
        """
        return "Move in Band Task"

    def _is_task_complete(self, doer: RetAgent) -> bool:
        """Check if the movement is complete.

        Args:
            doer (RetAgent): the agent doing the task.

        Returns:
            bool: true when the agent is within the tolerance of the destination, false
                otherwise.
        """
        doer_pos = doer.perceived_world.get_agent_pos(doer.unique_id)
        if doer_pos is None:
            warnings.warn(
                f"Agent {doer.unique_id} cannot find it's own position in it's perceived world "
                "so move in band task cannot be checked or completed.",
                stacklevel=2,
            )
            return False
        distance = doer.perceived_world.get_distance(doer_pos, self._real_destination)

        return distance <= self._tolerance

    def get_new_instance(self) -> MoveInBandTask:
        """Return a new instance of a functionally identical task.

        Returns:
            MoveInBandTask: New instance of the task
        """
        return MoveInBandTask(
            destination=cast("Coordinate3dBand", self._destination),
            bands=self._bands,
            space=self._space,
            tolerance=self._tolerance,
            log=self._log,
        )


class MoveToTargetTask(MoveTask):
    """Task for an agent to move to the location of a given agent."""

    def __init__(
        self,
        destination: Coordinate2dOr3d,
        tolerance: float,
        target: int,
        max_weapon_steps: int,
        log: bool = True,
    ) -> None:
        """Create task.

        Args:
            destination (Coordinate2dOr3d): The destination
            tolerance (float): The tolerance
            target (int): The unique ID of the target to aim at
            log (bool): whether to log or not. Defaults to True.
            max_weapon_steps (int): The maximum number of steps the weapon can move for
                before reaching max range
        """
        super().__init__(destination, tolerance, log)
        self._destination: Coordinate2dOr3d = destination
        self._target = target
        self._step_counter = 0
        self._max_weapon_steps = max_weapon_steps

    def __str__(self):
        """Output a human readable name for the task.

        Returns:
            string: brief description of the task
        """
        return "Move To Target Task"

    def _do_task_step(self, doer: RetAgent) -> None:
        """Move one time steps worth towards the destination.

        Args:
            doer (RetAgent): the agent doing the task
        """
        self._step_counter += 1
        target_agent = [a for a in doer.model.get_all_agents() if int(a.unique_id) == self._target][
            0
        ]
        if self._step_counter < self._max_weapon_steps:
            self._destination = doer.model.space.get_coordinate_2d(target_agent.pos)
        else:
            self._destination = doer.model.space.get_coordinate_2d(doer.pos)
        doer.move_step(self._destination)

    def get_new_instance(self) -> MoveToTargetTask:
        """Return a new instance of a functionally identical task.

        Returns:
            MoveTask: New instance of the task
        """
        return MoveToTargetTask(
            destination=self._destination,
            tolerance=self._tolerance,
            target=self._target,
            log=self._log,
            max_weapon_steps=self._max_weapon_steps,
        )


class RandomLocationPicker:
    """Utility class for selecting a random location."""

    def __init__(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        search_depth: int = 1000,
    ):
        """Create a new RandomLocationPicker.

        Args:
            x_min (float): Minimum allowable x position of the chosen coordinate
            x_max (float): Maximum allowable x position of the chosen coordinate
            y_min (float): Minimum allowable y position of the chosen coordinate
            y_max (float): Maximum allowable y position of the chosen coordinate
            search_depth (int): Maximum number of searches that can be made for picking a single
                coordinate. Defaults to 1000

        Raises:
            ValueError: Coordinates contain a minimum larger than the maximum.
            ValueError: A search depth of zero or less was given and is therefore is invalid.
        """
        if x_min > x_max or y_min > y_max:
            raise ValueError(f"Invalid coordinate bounds: {x_min=}, {x_max=}, {y_min=}, {y_max=}")

        if search_depth <= 0:
            raise ValueError(f"Invalid search depth: {search_depth=}")

        self._x_min = x_min
        self._x_max = x_max
        self._y_min = y_min
        self._y_max = y_max
        self._search_depth = search_depth

    def pick(
        self,
        coordinate: Coordinate,
        min_displacement: float,
        max_displacement: float,
        randomiser: Random,
    ) -> Coordinate2d:
        """Pick a random location within the parameters allowed by the picker.

        If the max search depth is reaches, the original coordinate will be returned

        Args:
            coordinate (Coordinate): The originating coordinate to deviate from.
            min_displacement (float): The minimum displacement from the originating coordinate
            max_displacement (float): The maximum displacement from the originating coordinate
            randomiser (Random): Mechanism for random number generation

        Returns:
            Coordinate2d: The chosen coordinate.

        Raises:
            ValueError: The minimum displacement is greater than the maximum displacement.
        """
        iterations = 0

        if min_displacement > max_displacement:
            raise ValueError(
                f"'min_displacement' ({min_displacement}) "
                + f"is greater than 'max_displacement' ({max_displacement})."
            )

        while iterations <= self._search_depth:
            new_coordinate = self._pick_one(
                coordinate, min_displacement, max_displacement, randomiser
            )
            if self._check_condition(new_coordinate):
                return new_coordinate
            iterations += 1

        warnings.warn(
            "Unable to pick a new random location within the allowable "
            + f"search depth. Returning initial position ({coordinate})",
            stacklevel=2,
        )
        return (coordinate[0], coordinate[1])

    def _check_condition(self, new_coordinate: Coordinate2d) -> bool:
        """Check whether the chosen point meets criteria of the new coordinate.

        Args:
            new_coordinate (Coordinate2d): Newly selected point

        Returns:
            bool: Whether or not the new point is valid
        """
        return (
            new_coordinate[0] >= self._x_min
            and new_coordinate[0] <= self._x_max
            and new_coordinate[1] >= self._y_min
            and new_coordinate[1] <= self._y_max
        )

    def _pick_one(
        self,
        coordinate: Coordinate,
        min_displacement: float,
        max_displacement: float,
        randomiser: Random,
    ) -> Coordinate2d:
        """Pick a 2D coordinate, a random displacement from the initial coordinate.

        Note that this will randomly pick the bearing towards and the distance to the random
        coordinate, therefore meaning that points towards the bound of the annulus will be selected
        with a greater frequency.

        Args:
            coordinate (Coordinate): The originating coordinate to deviate from.
            min_displacement (float): The minimum displacement from the originating coordinate
            max_displacement (float): The maximum displacement from the originating coordinate
            randomiser (Random): Mechanism for random number generation

        Returns:
            Coordinate2d: The chosen coordinate.
        """
        displacement = randomiser.uniform(min_displacement, max_displacement)
        bearing = randomiser.uniform(-np.pi, np.pi)

        start = complex(coordinate[0], coordinate[1])
        movement = cmath.rect(displacement, bearing)

        end = start + movement

        return (end.real, end.imag)


class InAreaRandomLocationPicker(RandomLocationPicker):
    """Location Picker which only chooses new destinations within an area."""

    def __init__(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        area: Area,
        search_depth: int = 1000,
    ):
        """Create a new RandomLocationPicker.

        Args:
            x_min (float): Minimum allowable x position of the chosen coordinate
            x_max (float): Maximum allowable x position of the chosen coordinate
            y_min (float): Minimum allowable y position of the chosen coordinate
            y_max (float): Maximum allowable y position of the chosen coordinate
            area (Area): The area in which the new coordinate must be chosen
            search_depth (int): Maximum number of searches that can be made for picking a single
                coordinate. Defaults to 1000
        """
        super().__init__(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            search_depth=search_depth,
        )
        self._area = area

    def _check_condition(self, new_coordinate: Coordinate2d) -> bool:
        """Check whether the chosen point meets criteria of the new coordinate.

        Args:
            new_coordinate (Coordinate2d): Newly selected point

        Returns:
            bool: Whether or not the new point is valid
        """
        in_space = super()._check_condition(new_coordinate)
        in_area = self._area.contains(new_coordinate)

        return in_space and in_area


class DisplacementCalculator(ABC):
    """Abstract representation of displacement calculator utility."""

    @abstractmethod
    def get_min_displacement(self, agent: RetAgent) -> float:  # pragma: no cover
        """Get minimum displacement.

        Args:
            agent (RetAgent): The agent which is determining how far it needs to move.

        Returns:
            float: Minimum displacement distance
        """
        raise NotImplementedError()

    @abstractmethod
    def get_max_displacement(self, agent: RetAgent) -> float:  # pragma: no cover
        """Get maximum displacement.

        Args:
            agent (RetAgent): The agent which is determining how far it needs to move.

        Returns:
            float: Maximum displacement distance
        """
        raise NotImplementedError()

    @abstractmethod
    def get_new_instance(self) -> DisplacementCalculator:  # pragma: no cover
        """Create new instance of displacement calculator.

        Returns:
            DisplacementCalculator: Copy of displacement calculator
        """
        raise NotImplementedError()


class FixedDisplacementCalculator(DisplacementCalculator):
    """Displacement calculator for cases where agent always has to move a pre-definable distance.

    This distance can be variable, i.e., within a min and max value, but cannot be determined based
    on environment.
    """

    def __init__(self, min_displacement: float, max_displacement: float):
        """Create a new FixedDisplacementCalculator.

        Args:
            min_displacement (float): Minimum displacement
            max_displacement (float): Maximum displacement
        """
        self._min_displacement = min_displacement
        self._max_displacement = max_displacement

    def get_min_displacement(self, agent: RetAgent) -> float:
        """Get minimum displacement.

        Args:
            agent (RetAgent): The agent which is determining how far it needs to move.

        Returns:
            float: Minimum displacement distance
        """
        return self._min_displacement

    def get_max_displacement(self, agent: RetAgent) -> float:
        """Get maximum displacement.

        Args:
            agent (RetAgent): The agent which is determining how far it needs to move.

        Returns:
            float: Maximum displacement distance
        """
        return self._max_displacement

    def get_new_instance(self) -> FixedDisplacementCalculator:
        """Create new instance of displacement calculator.

        Returns:
            FixedDisplacementCalculator: Copy of displacement calculator
        """
        return FixedDisplacementCalculator(
            min_displacement=self._min_displacement, max_displacement=self._max_displacement
        )


class ThreatBasedDisplacementCalculator(DisplacementCalculator):
    """Threat-Based displacement calculator.

    The distance moved is based on a risk value, which can be elicited from the model's perceived
    world.
    """

    def __init__(
        self,
        high_risk: tuple[float, float],
        low_risk: tuple[float, float],
        filter: Optional[PerceivedAgentFilter],
    ):
        """Create a new ThreatBasedDisplacementCalculator.

        Args:
            high_risk (tuple[float, float]): Min and max displacement for high risk cases.
            low_risk (tuple[float, float]): Min and max displacement for low risk cases.
            filter (Optional[PerceivedAgentFilter]): Filter for perceived world. If any agents match
                this filter, the high risk bounds will be used. Otherwise, the low risk ones will be
                used.
        """
        self._high_risk = high_risk
        self._low_risk = low_risk
        self._filter = filter

    def get_min_displacement(self, agent: RetAgent) -> float:
        """Get minimum displacement.

        Args:
            agent (RetAgent): The agent which is determining how far it needs to move.

        Returns:
            float: Minimum displacement distance
        """
        if len(agent.perceived_world.get_perceived_agents(self._filter)) > 0:
            return self._high_risk[0]
        return self._low_risk[0]

    def get_max_displacement(self, agent: RetAgent) -> float:
        """Get maximum displacement.

        Args:
            agent (RetAgent): The agent which is determining how far it needs to move.

        Returns:
            float: Maximum displacement distance
        """
        if len(agent.perceived_world.get_perceived_agents(self._filter)) > 0:
            return self._high_risk[1]
        return self._low_risk[1]

    def get_new_instance(self) -> ThreatBasedDisplacementCalculator:
        """Create new instance of displacement calculator.

        Returns:
            ThreatBasedDisplacementCalculator: Copy of displacement calculator
        """
        return ThreatBasedDisplacementCalculator(
            high_risk=self._high_risk, low_risk=self._low_risk, filter=self._filter
        )


class RandomMoveTask(Task):
    """Move task which makes the agent move in a random direction by a given distance.

    The distance is to move is dictated by the DisplacementCalculator, which facilitates threat
    based calculations.
    """

    def __init__(
        self,
        displacement_calculator: DisplacementCalculator,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        tolerance: float,
        area: Optional[Area] = None,
        log: bool = True,
    ):
        """Create a new RandomMoveTask.

        Args:
            displacement_calculator (DisplacementCalculator): Utility for calculating how far the
                agent should move.
            x_min (float): Minimum allowable x position of the chosen coordinate
            x_max (float): Maximum allowable x position of the chosen coordinate
            y_min (float): Minimum allowable y position of the chosen coordinate
            y_max (float): Maximum allowable y position of the chosen coordinate
            tolerance (float): Tolerance for the move reaching destination
            area (Optional[Area]): Area in which the random move must occur. Defaults to None, in
                which case the entire space is considered
            log (bool): Whether or not the movement should be logged. Defaults to True

        """
        super().__init__(log)
        self._displacement_calculator = displacement_calculator
        self._tolerance = tolerance
        self._log = log
        self._x_min = x_min
        self._x_max = x_max
        self._y_min = y_min
        self._y_max = y_max
        self._area = area

        if area is None:
            self._random_location_picker = RandomLocationPicker(x_min, x_max, y_min, y_max)
        else:
            self._random_location_picker = InAreaRandomLocationPicker(
                x_min, x_max, y_min, y_max, area
            )

        self._derived_move_task: Optional[MoveTask] = None

    def _is_task_complete(self, doer: RetAgent) -> bool:
        """Check if the movement is complete.

        Movement is complete if the agent has reached it's destination.

        Args:
            doer (RetAgent): [description]

        Returns:
            bool: Whether the task is complete or not.
        """
        if self._derived_move_task is None:
            return False

        return self._derived_move_task._is_task_complete(doer)

    def get_new_instance(self) -> RandomMoveTask:
        """Return a new instance of a functionally identical task.

        Returns:
            RandomMoveTask: New instance of the task
        """
        return RandomMoveTask(
            displacement_calculator=self._displacement_calculator.get_new_instance(),
            x_min=self._x_min,
            x_max=self._x_max,
            y_min=self._y_min,
            y_max=self._y_max,
            tolerance=self._tolerance,
            area=self._area,
            log=self._log,
        )

    def _do_task_step(self, doer: RetAgent) -> None:
        """Do a step in the move task.

        Args:
            doer (RetAgent): Agent doing the move task
        """
        if self._derived_move_task is None:
            min_displacement = self._displacement_calculator.get_min_displacement(doer)
            max_displacement = self._displacement_calculator.get_max_displacement(doer)
            destination = self._random_location_picker.pick(
                doer.pos, min_displacement, max_displacement, doer.model.random
            )
            # Create the move task to execute the move. Note that this will never log, as logging
            # is handled by this parent task
            self._derived_move_task = MoveTask(
                destination=destination, tolerance=self._tolerance, log=False
            )

        self._derived_move_task._do_task_step(doer)

    def __str__(self):
        """Output a human-readable name for the task.

        Returns:
            string: Name for the task
        """
        return "Random Move Task"


class FixedRandomMoveTask(RandomMoveTask):
    """Move task which makes the agent move in a random direction by a given distance."""

    def __init__(
        self,
        min_displacement: float,
        max_displacement: float,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        tolerance,
        area: Optional[Area] = None,
        log: bool = True,
    ):
        """Create a new RandomMoveTask.

        Args:
            min_displacement (float): The minimum displacement from the originating coordinate
            max_displacement (float): The maximum displacement from the originating coordinate
            x_min (float): Minimum allowable x position of the chosen coordinate
            x_max (float): Maximum allowable x position of the chosen coordinate
            y_min (float): Minimum allowable y position of the chosen coordinate
            y_max (float): Maximum allowable y position of the chosen coordinate
            tolerance (float): Tolerance for the move reaching destination
            area (Optional[Area]): Area in which the random move must occur. Defaults to None, in
                which case the entire space is considered
            log (bool): Whether or not the movement should be logged. Defaults to True

        """
        super().__init__(
            displacement_calculator=FixedDisplacementCalculator(
                min_displacement=min_displacement, max_displacement=max_displacement
            ),
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            tolerance=tolerance,
            area=area,  # area is not stateful, so this is safe to do
            log=log,
        )


class GroupFormationMoveTask(Task):
    """Move task which passes individual move orders from a group agent to subordinate agents."""

    def __init__(
        self,
        move_task: MoveTask,
        formation: Formation,
        subordinate_move_trigger: Trigger = None,
        subordinate_move_priority: int = 0,
        log: bool = True,
    ):
        """Create a new GroupFormationMoveTask.

        Args:
            move_task (MoveTask): A move task to be modified to include destination offsets
                representing a formation.
            formation (Formation): A formation object holding the formation configuration.
            subordinate_move_trigger (Optional[Trigger]): The trigger of the task to be passed on to
                the subordinate agents. Defaults to ImmediateTrigger.
            subordinate_move_priority (Optional[int]): The priority of the task to be passed on to
                the subordinate agents. Defaults to 0.
            log (bool): Whether or not the movement should be logged. Defaults to True
        """
        super().__init__(log=log)
        self.move_task = move_task
        self.formation = formation
        self.subordinate_move_priority = subordinate_move_priority
        self.subordinate_move_trigger = (
            subordinate_move_trigger if subordinate_move_trigger is not None else ImmediateTrigger()
        )

    def get_move_in_formation_task(
        self, move_task: MoveTask, offset: Tuple[float, float]
    ) -> MoveTask:
        """Modify a move task to include a formation offset.

        Args:
            move_task (MoveTask): The initial movement task to have it's destination modified.
            offset (Tuple[float, float]): The offset to apply to the move task.

        Returns:
            MoveTask: The modified movement task with the formation offset.
        """
        new_task = move_task.get_new_instance()
        new_task.update_destination(offset[0], offset[1])
        return new_task

    def generate_formation_offsets(self, doer: GroupAgent) -> dict[RetAgent, Tuple[float, float]]:
        """Modify a move task to include a formation offset.

        Args:
            doer (GroupAgent): The group agent which contains the subordinate agents.

        Returns:
            dict[int, Tuple[float, float]]: A dictionary of agent's unique IDs to their position
                modifiers. Giving these position modifiers to the corresponding agents creates a
                formation around the central point.
        """
        agents_to_receive_orders = [
            a for a in doer.agents if not a.killed or self.formation.include_killed
        ]
        offsets = self.formation.get_formation(len(agents_to_receive_orders))

        formation_dictionary = dict(zip([a for a in agents_to_receive_orders], offsets))
        return formation_dictionary

    def _is_task_complete(self, doer: RetAgent) -> bool:
        """Check if the giving of orders is complete.

        Args:
            doer (RetAgent): The agent doing the task.

        Returns:
            bool: True - Communication only ever takes one time step.
        """
        return True

    def _do_task_step(self, doer: RetAgent) -> None:
        """Do a step in the move task.

        Args:
            doer (RetAgent): Group Agent doing the move task

        Raises:
            ValueError: A non-GroupAgent was given a group agent formation task.
        """
        if not isinstance(doer, GroupAgent):
            raise ValueError(
                "Formation move task not given to a GroupAgent, "
                + f"receiving agent was of type: '{doer.agent_type.name}'"
            )

        if not doer.agents:
            warnings.warn("GroupAgent has no agents to move.", stacklevel=2)
            return

        for agent, offset in self.generate_formation_offsets(doer).items():
            new_move_task = self.get_move_in_formation_task(self.move_task, offset)
            agent.add_orders(
                Order(
                    trigger=self.subordinate_move_trigger,
                    task=new_move_task,
                    priority=self.subordinate_move_priority,
                )
            )

    def __str__(self):
        """Output a human-readable name for the task.

        Returns:
            string: Name for the task
        """
        return "Group Agent Formation Move Task"

    def get_new_instance(self) -> GroupFormationMoveTask:
        """Return a new instance of a functionally identical task.

        Returns:
            GroupFormationMoveTask: New instance of the task
        """
        return GroupFormationMoveTask(
            move_task=self.move_task,
            formation=self.formation,
            subordinate_move_priority=self.subordinate_move_priority,
            subordinate_move_trigger=self.subordinate_move_trigger,
            log=self._log,
        )
