"""Tests for move tasks."""
from __future__ import annotations

import random
import statistics
from typing import TYPE_CHECKING
from warnings import catch_warnings

import numpy as np
import pytest
from ret.agents.affiliation import Affiliation
from ret.agents.agent import RetAgent
from ret.behaviours.move import GroundBasedMoveBehaviour
from ret.orders.tasks.move import (
    FixedRandomMoveTask,
    MoveInBandTask,
    MoveTask,
    RandomLocationPicker,
    RandomMoveTask,
    ThreatBasedDisplacementCalculator,
)
from ret.sensing.agentcasualtystate import AgentCasualtyState
from ret.sensing.perceivedworld import (
    And,
    Confidence,
    HostileAgents,
    NearbyAgents,
    PerceivedAgent,
)
from ret.space.feature import BoxFeature
from ret.space.heightband import AbsoluteHeightBand
from ret.testing.mocks import (
    MockModel2d,
    MockModel3d,
    MockMoveBehaviour,
    MockMoveInBandBehaviour,
)
from pytest import raises

if TYPE_CHECKING:
    from ret.behaviours import Behaviour
    from ret.space.heightband import HeightBand


def test_basic_move_task():
    """Test basic 2D ground movement task."""
    model = MockModel2d()

    behaviours: list[Behaviour] = [MockMoveBehaviour()]

    agent = RetAgent(
        model=model,
        pos=(0, 0),
        name="Agent",
        affiliation=Affiliation.FRIENDLY,
        critical_dimension=2.0,
        reflectivity=0.1,
        temperature=20.0,
        behaviours=behaviours,
    )

    task = MoveTask((5000, 5000), 1)

    task.do_task_step(agent)
    assert task.is_task_complete(agent)
    assert agent.pos is not None
    assert agent.pos[0] == 5000
    assert agent.pos[1] == 5000


def test_update_destination_2d_with_3d_initial_destination():
    """Test the update_destination method with a 2D update to a 3D initial destination."""
    task = MoveTask((5000, 5000, 1000), 1)
    task.update_destination(1, 1)
    assert task._destination == (5001, 5001, 1000)


def test_move_in_band_task():
    """Test 3D move in band task."""
    model = MockModel3d()

    height_bands: list[HeightBand] = [
        AbsoluteHeightBand("0m", 0),
        AbsoluteHeightBand("1000m", 1000),
        AbsoluteHeightBand("2000m", 2000),
    ]

    behaviours: list[Behaviour] = [MockMoveInBandBehaviour(height_bands)]

    agent = RetAgent(
        model=model,
        pos=(0, 0, 0),
        name="Agent",
        affiliation=Affiliation.FRIENDLY,
        critical_dimension=2.0,
        reflectivity=0.1,
        temperature=20.0,
        behaviours=behaviours,
    )

    task = MoveInBandTask(
        (5000, 5000, "2000m"),
        height_bands,
        model.space,  # type: ignore
        1,
    )

    task.do_task_step(agent)
    assert task.is_task_complete(agent)
    assert agent.pos is not None
    assert agent.pos[0] == 5000
    assert agent.pos[1] == 5000
    assert agent.pos[2] == 2000


def test_non_existent_move_in_band_task():
    """Test error handling for non-existing band specification in move in band task."""
    model = MockModel3d()

    height_bands: list[HeightBand] = [
        AbsoluteHeightBand("0m", 0),
        AbsoluteHeightBand("1000m", 1000),
        AbsoluteHeightBand("2000m", 2000),
    ]

    with raises(ValueError) as e:
        MoveInBandTask(
            (5000, 5000, "non-existant"),
            height_bands,
            model.space,  # type: ignore
            1,
        )
    assert e.value.args[0] == 'Destination band with name "non-existant" does not exist'


def test_duplicate_move_in_band_task():
    """Test error handling for duplicate band specification in move in band task."""
    model = MockModel3d()

    height_bands: list[HeightBand] = [
        AbsoluteHeightBand("0m", 0),
        AbsoluteHeightBand("1000m", 1000),
        AbsoluteHeightBand("2000m", 2000),
        AbsoluteHeightBand("duplicate", 2000),
        AbsoluteHeightBand("duplicate", 2000),
    ]

    with raises(ValueError) as e:
        MoveInBandTask(
            (5000, 5000, "duplicate"),
            height_bands,
            model.space,  # type: ignore
            1,
        )
    assert e.value.args[0] == 'More than one band with name "duplicate" exists'


def test_move_in_band_str_method():
    """Test the move in band task string method."""
    model = MockModel3d()

    height_bands: list[HeightBand] = [
        AbsoluteHeightBand("0m", 0),
        AbsoluteHeightBand("1000m", 1000),
        AbsoluteHeightBand("2000m", 2000),
    ]

    task = MoveInBandTask(
        (5000, 5000, "2000m"),
        height_bands,
        model.space,  # type: ignore
        1,
    )
    assert str(task) == "Move in Band Task"


def test_move_in_band_get_new_instance():
    """Test the move in band task get_new_instance method."""
    model = MockModel3d()

    height_bands: list[HeightBand] = [
        AbsoluteHeightBand("0m", 0),
        AbsoluteHeightBand("1000m", 1000),
        AbsoluteHeightBand("2000m", 2000),
    ]

    task = MoveInBandTask(
        destination=(5000, 5000, "2000m"),
        bands=height_bands,
        space=model.space,  # type: ignore
        tolerance=1,
        log=False,
    )
    clone = task.get_new_instance()

    assert task is not clone
    assert isinstance(clone, MoveInBandTask)
    assert task._destination == clone._destination
    assert task._tolerance == clone._tolerance
    assert task._real_destination == clone._real_destination
    assert task._log == clone._log


def test_random_move_task():
    """Test random move task, for moves where the entire move can be completed in a single step."""
    model = MockModel2d()

    behaviours: list[Behaviour] = [MockMoveBehaviour()]

    agent = RetAgent(
        model=model,
        pos=(0, 0),
        name="Agent",
        affiliation=Affiliation.FRIENDLY,
        critical_dimension=2.0,
        reflectivity=0.1,
        temperature=20.0,
        behaviours=behaviours,
    )

    min_distance = 10
    max_distance = 20

    task = FixedRandomMoveTask(
        min_displacement=min_distance,
        max_displacement=max_distance,
        tolerance=0.5,
        x_min=model.space.x_min,
        x_max=model.space.x_max,
        y_min=model.space.y_min,
        y_max=model.space.y_max,
    )
    task.do_task_step(agent)

    assert agent.pos is not None
    displacement = np.linalg.norm(agent.pos)

    assert displacement >= min_distance
    assert displacement <= max_distance


def test_multi_step_move_behaviour():
    """Test random move task, for moves where the entire move takes multiple steps."""
    model = MockModel2d()

    behaviours: list[Behaviour] = [
        GroundBasedMoveBehaviour(base_speed=3, gradient_speed_modifiers=[((-np.inf, np.inf), 1)])
    ]

    agent = RetAgent(
        model=model,
        pos=(0, 0),
        name="Agent",
        affiliation=Affiliation.FRIENDLY,
        critical_dimension=2.0,
        reflectivity=0.1,
        temperature=20.0,
        behaviours=behaviours,
    )

    min_distance = 10
    max_distance = 20

    task = FixedRandomMoveTask(
        min_displacement=min_distance,
        max_displacement=max_distance,
        tolerance=0.5,
        x_min=model.space.x_min,
        x_max=model.space.x_max,
        y_min=model.space.y_min,
        y_max=model.space.y_max,
    )

    for _ in range(0, 7):
        task.do_task_step(agent)

    displacement = np.linalg.norm(agent.pos)
    assert displacement >= min_distance
    assert displacement <= max_distance


def test_multiple_moves_have_different_outcomes():
    """Test that multiple random moves end up in different locations."""
    model = MockModel2d()

    behaviours: list[Behaviour] = [MockMoveBehaviour()]

    new_positions = []

    min_distance = 10
    max_distance = 20

    for _ in range(0, 100):
        agent = RetAgent(
            model=model,
            pos=(0, 0),
            name="Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            behaviours=behaviours,
        )

        task = FixedRandomMoveTask(
            min_displacement=min_distance,
            max_displacement=max_distance,
            tolerance=0.5,
            x_min=model.space.x_min,
            x_max=model.space.x_max,
            y_min=model.space.y_min,
            y_max=model.space.y_max,
        )
        task.do_task_step(agent)

        new_positions.append(agent.pos)

    displacements = [np.linalg.norm(new_position) for new_position in new_positions]

    # rough check that the random movements are variable within the range 10-20. In very rare cases,
    # these could fail. However the statistical chance of this is very low
    assert statistics.mean(displacements) > 13
    assert statistics.mean(displacements) < 17
    assert max(displacements) > 18
    assert min(displacements) < 12

    # Check that new positions have variability in both X and Y coordinates
    assert len(set(p[0] for p in new_positions)) > 1
    assert len(set(p[1] for p in new_positions)) > 1


def test_random_move_is_in_defined_area():
    """Test that random moves always end in a defined area, if supplied."""
    model = MockModel2d()

    behaviours: list[Behaviour] = [MockMoveBehaviour()]

    min_distance = 10
    max_distance = 20

    area = BoxFeature((5, 4), (19, 5), "box")

    for _ in range(0, 100):
        agent = RetAgent(
            model=model,
            pos=(0, 0),
            name="Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            behaviours=behaviours,
        )

        task = FixedRandomMoveTask(
            min_displacement=min_distance,
            max_displacement=max_distance,
            tolerance=0.5,
            x_min=model.space.x_min,
            x_max=model.space.x_max,
            y_min=model.space.y_min,
            y_max=model.space.y_max,
            area=area,
        )
        task.do_task_step(agent)

        assert agent.pos is not None
        assert area.contains(agent.pos)


def test_move_picker_search_depth():
    """Test the search depth for move pickers.

    A move picker with greater search depth finds more candidates over a suitably large number of
    selections.
    """
    move_picker_1_change = RandomLocationPicker(
        x_min=0,
        x_max=100,
        y_min=0,
        y_max=100,
        search_depth=1,
    )
    move_picker_100_change = RandomLocationPicker(
        x_min=0,
        x_max=100,
        y_min=0,
        y_max=100,
        search_depth=100,
    )

    not_found_count_for_search_depth_1 = 0
    not_found_count_for_search_depth_100 = 0
    r = random.Random()

    with catch_warnings(record=True):
        for _ in range(0, 100):
            coord_1 = move_picker_1_change.pick(
                (0, 0), min_displacement=5, max_displacement=10, randomiser=r
            )
            if coord_1 == (0, 0):
                not_found_count_for_search_depth_1 += 1

            coord_2 = move_picker_100_change.pick(
                (0, 0), min_displacement=5, max_displacement=10, randomiser=r
            )
            if coord_2 == (0, 0):
                not_found_count_for_search_depth_100 += 1

    assert not_found_count_for_search_depth_1 > not_found_count_for_search_depth_100


def test_not_found_warning():
    """Test warnings where a point cannot be found."""
    # This random location picker can never pick a destination where moving from (0, 0), as the
    # minimum distance will always take it out of the space.
    picker = RandomLocationPicker(
        x_min=0,
        x_max=1,
        y_min=0,
        y_max=1,
        search_depth=10,
    )

    with catch_warnings(record=True) as w:
        coord = picker.pick(
            (0, 0), min_displacement=10, max_displacement=20, randomiser=random.Random()
        )

    assert coord == (0, 0)
    assert len(w) == 1
    assert "Unable to pick a new random location within the allowable search depth" in str(
        w[0].message
    )


def test_threat_based_displacement():
    """Test the calculation of threat-based displacement distances."""
    model = MockModel2d()
    agent = RetAgent(
        model=model,
        pos=(0, 0),
        name="Agent",
        affiliation=Affiliation.FRIENDLY,
        critical_dimension=2.0,
        reflectivity=0.1,
        temperature=20.0,
    )

    filter = And([HostileAgents(), NearbyAgents(model.space.get_distance, agent, 10)])

    threat_based_displacement_calculator = ThreatBasedDisplacementCalculator(
        high_risk=(20, 30), low_risk=(5, 10), filter=filter
    )

    assert len(agent.perceived_world.get_perceived_agents(HostileAgents())) == 0  # No hostile
    assert threat_based_displacement_calculator.get_max_displacement(agent) == 10
    assert threat_based_displacement_calculator.get_min_displacement(agent) == 5

    # Hostile, but too far away to trigger the nearly filter
    agent.perceived_world.add_acquisitions(
        PerceivedAgent(
            sense_time=model.get_time(),
            confidence=Confidence.IDENTIFY,
            location=(10, 10),
            unique_id=2,
            casualty_state=AgentCasualtyState.ALIVE,
            affiliation=Affiliation.HOSTILE,
        )
    )

    assert len(agent.perceived_world.get_perceived_agents(HostileAgents())) == 1  # Hostile far away
    assert threat_based_displacement_calculator.get_max_displacement(agent) == 10
    assert threat_based_displacement_calculator.get_min_displacement(agent) == 5

    # Triggers the nearby filter
    agent.perceived_world.add_acquisitions(
        PerceivedAgent(
            sense_time=model.get_time(),
            confidence=Confidence.IDENTIFY,
            location=(2, 2),
            unique_id=3,
            casualty_state=AgentCasualtyState.ALIVE,
            affiliation=Affiliation.HOSTILE,
        )
    )

    assert len(agent.perceived_world.get_perceived_agents(HostileAgents())) == 2
    assert threat_based_displacement_calculator.get_max_displacement(agent) == 30
    assert threat_based_displacement_calculator.get_min_displacement(agent) == 20


def test_threat_based_random_move_task():
    """Test random threat-based move task."""
    model = MockModel2d()

    behaviours: list[Behaviour] = [MockMoveBehaviour()]

    agent = RetAgent(
        model=model,
        pos=(0, 0),
        name="Agent",
        affiliation=Affiliation.FRIENDLY,
        critical_dimension=2.0,
        reflectivity=0.1,
        temperature=20.0,
        behaviours=behaviours,
    )

    min_distance = 10
    max_distance = 20

    # The distance of the move will be zero in low risk cases, meaning the agent will stay at (0, 0)
    task = RandomMoveTask(
        displacement_calculator=ThreatBasedDisplacementCalculator(
            high_risk=(min_distance, max_distance), low_risk=(0, 0), filter=HostileAgents()
        ),
        tolerance=0.5,
        x_min=model.space.x_min,
        x_max=model.space.x_max,
        y_min=model.space.y_min,
        y_max=model.space.y_max,
    )

    assert agent.pos is not None
    assert agent.pos[0] == 0
    assert agent.pos[1] == 0

    task.do_task_step(agent)

    assert agent.pos is not None
    assert agent.pos[0] == 0
    assert agent.pos[1] == 0

    agent.perceived_world.add_acquisitions(
        PerceivedAgent(
            sense_time=model.get_time(),
            confidence=Confidence.IDENTIFY,
            location=(10, 10),
            unique_id=2,
            casualty_state=AgentCasualtyState.ALIVE,
            affiliation=Affiliation.HOSTILE,
        )
    )

    # Task has to be copied because it is 'complete' in the previous step (even though zero move)
    task.get_new_instance().do_task_step(agent)

    assert agent.pos is not None
    displacement = np.linalg.norm(agent.pos)

    assert displacement >= min_distance
    assert displacement <= max_distance


@pytest.mark.parametrize("x_min,x_max,y_min,y_max", [[-1, -5, 3, 5], [3, 7, 4, 3]])
def test_location_picker_initialisation(x_min: float, x_max: float, y_min: float, y_max: float):
    """Test validation of RandomLocationPicker creation.

    Args:
        x_min (float): Min x coordinate
        x_max (float): Max x coordinate
        y_min (float): Min y coordinate
        y_max (float): Max y coordinate
    """
    with pytest.raises(ValueError) as e:
        RandomLocationPicker(x_min, x_max, y_min, y_max, search_depth=1)

    assert e.value.args[0] == f"Invalid coordinate bounds: {x_min=}, {x_max=}, {y_min=}, {y_max=}"


def test_search_depth_location_picker_initialisation():
    """Test validation of search depth during RandomLocationPicker creation."""
    with pytest.raises(ValueError) as e:
        RandomLocationPicker(0, 100, 0, 100, 0)

    assert e.value.args[0] == "Invalid search depth: search_depth=0"


def test_min_max_displacement_validation():
    """Test validation for RandomLocationPicker min and max displacement."""
    rlp = RandomLocationPicker(0, 100, 0, 100, 1)

    with pytest.raises(ValueError) as e:
        rlp.pick(
            coordinate=(0, 0), min_displacement=10, max_displacement=9, randomiser=random.Random()
        )

    assert e.value.args[0] == "'min_displacement' (10) is greater than 'max_displacement' (9)."


def test_random_move_new_instance_references():
    """Test getting new instance for random move."""
    task = RandomMoveTask(
        displacement_calculator=ThreatBasedDisplacementCalculator(
            high_risk=(0, 10), low_risk=(0, 0), filter=HostileAgents()
        ),
        tolerance=0.5,
        x_min=0,
        x_max=50,
        y_min=0,
        y_max=50,
    )

    clone = task.get_new_instance()

    assert clone is not task
    assert clone._displacement_calculator is not task._displacement_calculator


@pytest.mark.parametrize(
    "attr",
    [
        "_x_min",
        "_x_max",
        "_y_min",
        "_y_max",
        "_tolerance",
        "_area",
        "_log",
        "_displacement_calculator",
    ],
)
def test_random_move_new_instance_fields(attr: str):
    """Test that modifying fields in cloned RandomMoveTask does not influence original.

    Args:
        attr (str): The attribute to test modification.
    """
    task = RandomMoveTask(
        displacement_calculator=ThreatBasedDisplacementCalculator(
            high_risk=(0, 10), low_risk=(0, 0), filter=HostileAgents()
        ),
        tolerance=0.5,
        x_min=0,
        x_max=50,
        y_min=0,
        y_max=50,
    )

    clone = task.get_new_instance()

    setattr(task, attr, random.random())
    t_attr = getattr(task, attr)
    c_attr = getattr(clone, attr)
    assert t_attr != c_attr


def test_random_move_str():
    """Test random move string."""
    task = RandomMoveTask(
        displacement_calculator=ThreatBasedDisplacementCalculator(
            high_risk=(0, 10), low_risk=(0, 0), filter=HostileAgents()
        ),
        tolerance=0.5,
        x_min=0,
        x_max=50,
        y_min=0,
        y_max=50,
    )

    assert str(task) == "Random Move Task"
