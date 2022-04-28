"""Tests for sensor field of view."""
from __future__ import annotations

import unittest
from typing import TYPE_CHECKING
from unittest.mock import Mock, create_autospec, patch, sentinel
from warnings import catch_warnings

import mesa_ret.utilities.geometric_utilities as gu
from mesa_ret.agents.affiliation import Affiliation
from mesa_ret.agents.agent import RetAgent
from mesa_ret.space.space import ContinuousSpaceWithTerrainAndCulture2d
from mesa_ret.testing.mocks import (
    MockAgent,
    MockFieldOfViewSensor,
    MockModel,
    MockModel2dWithBoxFeature,
    MockModel3dWithBoxFeature,
    MockSensor,
)
from parameterized.parameterized import parameterized

if TYPE_CHECKING:
    from typing import Optional, Union

    from mesa_ret.sensing.sensor import ArcOfRegard
    from mesa_ret.types import Coordinate2d, Coordinate2dOr3d, Coordinate3d


class TestFieldOfView(unittest.TestCase):
    """Test field of view calculations."""

    def setUp(self):
        """Set up test cases."""
        # 2D Model with Box Feature
        self.model_2d = MockModel2dWithBoxFeature(
            box_name="Mock Box Feature", box_min_coord=(0.0, 100.0), box_max_coord=(1.0, 101.0)
        )

        self.sensor_agent_2d = RetAgent(
            model=self.model_2d,
            pos=(100.0, 100.0),
            name="Sensor Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            arc_of_regard={(350, 10): 1},
        )

        sensor_agent_location: Coordinate2d = self.sensor_agent_2d.pos  # type: ignore
        # Layout of target agents around sensor agent where x is the sensor agent and
        # target agents are numbered
        #     1  2
        #  6  x  3
        #     5  4
        for location_offset in [
            (0, 1),  # 1
            (1, 1),  # 2
            (1, 0),  # 3
            (1, -1),  # 4
            (0, -1),  # 5
            (-1, 0),  # 6
        ]:
            location = (
                sensor_agent_location[0] + location_offset[0],
                sensor_agent_location[1] + location_offset[1],
            )
            RetAgent(
                model=self.model_2d,
                pos=location,
                name="Target Agent",
                affiliation=Affiliation.FRIENDLY,
                critical_dimension=2.0,
                reflectivity=0.1,
                temperature=20.0,
            )

            # 3D Model
            self.model_3d = MockModel3dWithBoxFeature(
                box_name="Mock Box Feature",
                box_min_coord=(100.0, 200.0, 0.0),
                box_max_coord=(100.0, 201.0, 1.0),
            )

            self.sensor_agent_3d = RetAgent(
                model=self.model_3d,
                pos=(100.0, 100.0, 100.0),
                name="Sensor Agent",
                affiliation=Affiliation.FRIENDLY,
                critical_dimension=2.0,
                reflectivity=0.1,
                temperature=20.0,
                arc_of_regard={(350, 10): 1},
            )
        sensor_agent_location: Coordinate3d = self.sensor_agent_3d.pos  # type: ignore
        for location_offset in [
            (0, 1, -50),  # North and below
            (1, 1, 50),  # North East and above
            (1, 0, 0),  # East and at same height
        ]:
            location = (
                sensor_agent_location[0] + location_offset[0],
                sensor_agent_location[1] + location_offset[1],
                sensor_agent_location[2] + location_offset[2],
            )
            RetAgent(
                model=self.model_3d,
                pos=location,
                name="Target Agent",
                affiliation=Affiliation.FRIENDLY,
                critical_dimension=2.0,
                reflectivity=0.1,
                temperature=20.0,
            )

    @parameterized.expand(
        [
            [(1, 1), 0, 45],
            [(1, 0), 0, 90],
            [(0, 1), 0, 0],
            [(-1, 1), 0, 45],
            [(-1, 0), 0, 90],
            [(-1, -1), 0, 135],
            [(0, -1), 0, 180],
            [(1, 1), 90, 45],
            [(1, 0), 90, 0],
            [(-1, 1), 90, 135],
            [(0, -1), 90, 90],
            [(-1, 0), 90, 180],
            [(1, 1), 180, 135],
            [(1, 0), 180, 90],
            [(-1, 1), 180, 135],
            [(0, 1), 180, 180],
            [(1, 1), 270, 135],
            [(1, 0), 270, 180],
            [(-1, 1), 270, 45],
            [(0, 1), 270, 90],
        ]
    )
    def test_get_abs_angle_to_target(
        self, location_offset: tuple[float, float], sense_direction: float, expected_angle: float
    ):
        """Test get angle to target method.

        Args:
            location_offset (tuple[float, float]): The offset of the target agent location from the
                sensor agent.
            sense_direction (float): The direction to sense in (clockwise from the y-axis in
                degrees).
            expected_angle (float): The expected angle.
        """
        sensor_agent_location: Coordinate2d = self.sensor_agent_2d.pos  # type: ignore
        target_location = (
            sensor_agent_location[0] + location_offset[0],
            sensor_agent_location[1] + location_offset[1],
        )
        target_agent = RetAgent(
            model=self.model_2d,
            pos=target_location,
            name="Target Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        sensor = MockSensor(view_angle=90)
        angle = sensor._get_angle_size_to_target(
            self.sensor_agent_2d, target_agent, sense_direction
        )
        assert angle == expected_angle

    @parameterized.expand(
        [
            [(1, 1, 0), 0, 45],
            [(1, 1, 50), 0, 45],
            [(1, 1, -50), 0, 45],
            [(0, 1, 0), 0, 0],
            [(0, 1, 50), 0, 0],
            [(0, 1, -50), 0, 0],
            [(1, 1, 0), 90, 45],
            [(1, 1, 50), 90, 45],
            [(1, 1, -50), 90, 45],
            [(1, 0, 0), 180, 90],
            [(1, 0, 50), 180, 90],
            [(1, 0, -50), 180, 90],
            [(1, 1, 0), 270, 135],
            [(1, 1, 50), 270, 135],
            [(1, 1, -50), 270, 135],
        ]
    )
    def test_get_abs_angle_to_target_3d(
        self,
        location_offset: tuple[float, float, float],
        sense_direction: float,
        expected_angle: float,
    ):
        """Test get angle to target method with 3D coordinates.

        Args:
            location_offset (tuple[float, float, float]): The offset of the target agent location
                from the sensor agent.
            sense_direction (float): The direction to sense in (clockwise from the y-axis in
                degrees).
            expected_angle (float): The expected angle.
        """
        sensor_agent_location: Coordinate3d = self.sensor_agent_3d.pos  # type: ignore
        target_location = (
            sensor_agent_location[0] + location_offset[0],
            sensor_agent_location[1] + location_offset[1],
            sensor_agent_location[2] + location_offset[2],
        )
        target_agent = RetAgent(
            model=self.model_3d,
            pos=target_location,
            name="Target Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        sensor = MockSensor(view_angle=90)
        angle = sensor._get_angle_size_to_target(
            self.sensor_agent_3d, target_agent, sense_direction
        )
        assert angle == expected_angle

    @parameterized.expand(
        [
            [10, [0, 0.001, -1, -4.999, 5], [-5.001, 179, -179, 180]],
            [359, [0, 45, 179.5, -90], [180]],
            [180, [0, 90, -90], [-90.001, 180]],
        ]
    )
    @patch("mesa_ret.utilities.geometric_utilities._heading_to_clockise_angle_degrees")
    def test_reduce_to_field_of_view(
        self,
        view_angle: float,
        headings_viewable: list[float],
        headings_nonviewable: list[float],
        mock_heading_to_clockise_angle_degrees: Mock,
    ):
        """Parameterised unit test of the Sensor.reduce_to_field_of_view method.

        Args:
            view_angle (float): Angular width supplied for the sensor's field of view.
            headings_viewable (list[float]): List of some heading angles (clockwise
                from +y) that should be in view.
            headings_nonviewable (list[float]): List of some heading angles (clockwise
                from +y) that should not be in view.
            mock_heading_to_clockise_angle_degrees (Mock): Mock of
                _heading_to_clockise_angle_degrees that will return predefined values so
                reduce_to_field_of_view is tested in isolation.
        """
        viewable_agents = [
            MockAgent(str(uid), sentinel.pos)  # type:ignore
            for uid in headings_viewable
        ]

        nonviewable_agents = [
            MockAgent(str(uid), sentinel.pos)  # type:ignore
            for uid in headings_nonviewable
        ]

        all_agents: list[RetAgent] = viewable_agents + nonviewable_agents  # type:ignore

        sensor = MockSensor()
        sensor.view_angle = view_angle
        gu._heading_to_clockise_angle_degrees.side_effect = [  # type:ignore
            angle for angle in headings_viewable + headings_nonviewable
        ]

        senser = MockAgent(sentinel.uid, (0, 0))
        senser.model = MockModel(create_autospec(ContinuousSpaceWithTerrainAndCulture2d))
        senser.model.space.get_heading.side_effect = [
            (0, 0)  # Heading must be supplied but is overridden by mock
            for _ in range(len(headings_viewable) + len(headings_nonviewable))
        ]

        viewed_agents = sensor._reduce_to_field_of_view(senser, all_agents, 0)
        viewed_uids = [agent.unique_id for agent in viewed_agents]

        for agent in viewable_agents:
            assert (str(agent.unique_id), agent) in zip(
                viewed_uids, viewed_agents  # type:ignore
            )

        for agent in nonviewable_agents:
            assert (str(agent.unique_id), agent) not in zip(
                viewed_uids, viewed_agents  # type:ignore
            )

    @parameterized.expand(
        [
            [
                90,
                45,
                [(0.001, 0), (1, 1), (1.0, 0.0), (0, 1), (1000, 100), (4.673, 2.001)],
                [
                    (-0.001, 0),
                    (1, -1),
                    (-1.0, -0.0),
                    (0, -1),
                    (-1000, 100),
                    (4.673, -2.001),
                    (-1, -1),
                    (-1, 1),
                ],
            ],
            [
                269.999,
                225,
                [
                    (-0.001, 0),
                    (1, -1),
                    (-1.0, -0.0),
                    (0, -1),
                    (-1000, 100),
                    (4.673, -2.001),
                    (-1, -1),
                    (-1, 1),
                ],
                [(0.001, 0), (1, 1), (1.0, 0.0), (0, 1), (1000, 100), (4.673, 2.001)],
            ],
        ]
    )
    def test_reduce_to_field_of_view_integration(
        self,
        view_angle: float,
        sense_direction: float,
        positions_viewable: list[tuple[float, float]],
        positions_nonviewable: list[tuple[float, float]],
    ):
        """Integration test for Sensor.reduce_to_field_of_view.

        Args:
            view_angle (float): Angular width supplied for the sensor's field of view.
            sense_direction (float): The direction to sense in (clockwise from the y-axis in
                degrees).
            positions_viewable (list[tuple[float, float]]): List of some agent positions that
                should be in view.
            positions_nonviewable (list[tuple[float, float]]): List of some agent positions that
                should not be in view.
        """
        senser = MockAgent(sentinel.uid, (0, 0))
        senser.model = MockModel(ContinuousSpaceWithTerrainAndCulture2d(1000, 1000))

        sensor = MockSensor(view_angle=view_angle)

        viewable_agents = [
            MockAgent(uid, position) for uid, position in enumerate(positions_viewable)
        ]

        nonviewable_agents = [
            MockAgent(uid, position)
            for uid, position in enumerate(positions_nonviewable, start=len(positions_viewable))
        ]

        all_agents: list[RetAgent] = viewable_agents + nonviewable_agents  # type: ignore

        viewed_agents = sensor._reduce_to_field_of_view(senser, all_agents, sense_direction)
        viewed_uids = [agent.unique_id for agent in viewed_agents]

        for agent in viewable_agents:
            assert (agent.unique_id, agent) in zip(viewed_uids, viewed_agents)  # type:ignore

        for agent in nonviewable_agents:
            assert (agent.unique_id, agent) not in zip(viewed_uids, viewed_agents)  # type:ignore

    def test_choose_arc_of_regard_sector(self):
        """Test choosing of sector from arc of regard."""
        sensor = MockSensor()
        # Arc of regard must be normalised
        arc_of_regard: ArcOfRegard = {(340, 20): 0.7, (20, 90): 0.3}
        sector = sensor._choose_arc_of_regard_sector(arc_of_regard, self.model_2d)
        assert sector in arc_of_regard

        n_total = 1000
        n_middle_sector = 0
        n_offset_sector = 0
        for _ in range(n_total):
            sector = sensor._choose_arc_of_regard_sector(arc_of_regard, self.model_2d)
            if sector == (340, 20):
                n_middle_sector += 1
            elif sector == (20, 90):
                n_offset_sector += 1

        assert n_middle_sector + n_offset_sector == n_total
        assert n_middle_sector > n_offset_sector
        assert n_middle_sector > 0
        assert n_offset_sector > 0

    @parameterized.expand(
        [
            [(30, 60), 45],
            [(60, 30), 225],
            [(300, 360), 330],
            [(360, 300), 150],
            [(180, 360), 270],
            [(360, 180), 90],
            [(340, 20), 0],
            [(20, 340), 180],
            [(350, 20), 5],
            [(20, 350), 185],
            [(330, 20), 355],
            [(20, 330), 175],
            [(30, 30), 30],
            [(0, 360), 180],
            [(360, 0), 0],
        ]
    )
    def test_get_sector_midpoint_angle(
        self, sector: tuple[float, float], expected_midpoint_angle: float
    ):
        """Test get sector midpoint angle.

        All angles are in degrees between 0 and 360.

        Args:
            sector (tuple[float, float]): The sector to find the midpoint of
            expected_midpoint_angle (float): The expected midpoint
        """
        sensor = MockSensor()
        midpoint_angle = sensor._get_sector_midpoint_angle(sector)
        assert midpoint_angle == expected_midpoint_angle

    @parameterized.expand(
        [
            [
                90,  # view angle is 90 so can sense +-45 degrees
                90,  # try to try to look straight down the positive x-axis
                (-20, 20),  # look straight at sense direction
                3,
            ],
            [
                180,  # view angle is 180 so can sense +-90 degrees
                90,  # try to look straight down the positive x-axis
                (-20, 20),  # look straight at sense direction
                5,
            ],
            [
                None,  # view angle is None so can sense all around
                90,  # try to look straight down the positive x-axis
                (-20, 20),  # look straight at sense direction
                6,  # all agents
            ],
            [
                40,  # view angle is 40 so can sense +-20 degrees
                0,  # try to look straight down the positive y-axis
                (-20, 20),  # look straight at sense direction
                1,
            ],
            [
                40,  # view angle is 40 so can sense +-20 degrees
                0,  # try to look straight down the positive y-axis
                (-20, 20),  # look straight at sense direction
                1,
            ],
            [
                90,  # view angle is 90 so can sense +-45 degrees
                0,  # try to look straight down the positive y-axis
                (20, 40),  # offset sense direction by +30 degrees
                2,
            ],
            [
                90,  # view angle is 90 so can sense +-45 degrees
                0,  # try to look straight down the positive y-axis
                (180, 360),  # offset sense direction by +270 degrees
                1,  # only sense agent 6
            ],
            [
                90,  # view angle is 90 so can sense +-45 degrees
                45,  # try to look straight down (1, 1)
                (-5, 5),  # look straight at sense direction
                3,
            ],
        ]
    )
    def test_filter_target_agents_by_view_angle(
        self,
        view_angle: Optional[float],
        sense_direction: float,
        sector: tuple[float, float],
        expected_number_of_target_agents: int,
    ):
        """Test filtering of target agents using the view angle.

        Args:
            view_angle (Optional[float]): View angle
            sense_direction (float): sense direction
            sector (tuple[float, float]): Sector of angles to consider
            expected_number_of_target_agents (int): Expected number of agents to be detected
        """
        sensor = MockSensor(view_angle=view_angle)

        # Add type: ignore to suppress black 'Cannot assign to a method' error
        sensor._choose_arc_of_regard_sector = Mock(return_value=sector)  # type: ignore

        all_target_agents: list[
            RetAgent
        ] = self.sensor_agent_2d.model.schedule.agents  # type: ignore
        all_target_agents.remove(self.sensor_agent_2d)
        filtered_agents = sensor._filter_target_agents_by_view_angle(
            self.sensor_agent_2d,
            sense_direction=sense_direction,
            all_target_agents=all_target_agents,
        )
        assert len(filtered_agents) == expected_number_of_target_agents

    @parameterized.expand(
        [
            [
                90,  # view angle is 90 so can sense +-45 degrees
                180.0,  # look straight down the negative y-axis (float)
                2,
            ],
            [
                90,  # view angle is 90 so can sense +-45 degrees
                180,  # look straight down the negative y-axis (int)
                2,
            ],
            [
                90,  # view angle is 90 so can sense +-45 degrees
                None,  # no sense direction so default to 90 degrees
                3,
            ],
            [
                90,  # view angle is 90 so can sense +-45 degrees
                (99.0, 100.0),  # coordinate in direction of negative x-axis
                1,
            ],
            [
                90,  # view angle is 90 so can sense +-45 degrees
                (99.0, 100.0),  # coordinate in direction of negative x-axis
                1,
            ],
            [
                90,  # view angle is 90 so can sense +-45 degrees
                "Mock Box Feature",  # direction of box approx 270 degrees
                1,
            ],
        ]
    )
    def test_run_detection(
        self,
        view_angle: Optional[float],
        sense_direction: Optional[Union[float, Coordinate2dOr3d, str]],
        expected_number_of_target_agents: int,
    ):
        """Test sensor run detection method.

        Args:
            view_angle (Optional[float]): View angle
            sense_direction (Optional[Union[float, Coordinate2dOr3d, str]]): sense direction
            expected_number_of_target_agents (int): Expected number of agents to be detected
        """
        fov_sensor = MockFieldOfViewSensor(view_angle=view_angle)

        with catch_warnings(record=True) as w:
            fov_sensor.run_detection(
                sensor_agent=self.sensor_agent_2d, sense_direction=sense_direction
            )
            perceived_agents = fov_sensor.get_results(sensor_agent=self.sensor_agent_2d)

        if sense_direction:
            assert len(w) == 0
        else:
            assert len(w) == 1
            assert "No sense direction provided, assuming positive x direction." in str(
                w[0].message
            )

        assert len(perceived_agents) == expected_number_of_target_agents

    @parameterized.expand(
        [
            [
                90,  # view angle is 90 so can sense +-45 degrees
                45,  # look straight down the line x=y
                3,
            ],
            [
                90,  # view angle is 90 so can sense +-45 degrees
                90,  # look straight down the positive x-axis
                2,
            ],
            [
                90,  # view angle is 90 so can sense +-45 degrees
                None,  # no sense direction so default to 90 degrees
                2,
            ],
            [
                90,  # view angle is 90 so can sense +-45 degrees
                (101.0, 100.0, 0.0),  # coordinate in direction of positive x-axis
                2,
            ],
            [
                90,  # view angle is 90 so can sense +-45 degrees
                "Mock Box Feature",  # direction of box, approx 0 degrees
                2,
            ],
        ]
    )
    def test_run_detection_3d(
        self,
        view_angle: Optional[float],
        sense_direction: Optional[Union[float, Coordinate2dOr3d, str]],
        expected_number_of_target_agents: int,
    ):
        """Test sensor run_detection method.

        Args:
            view_angle (Optional[float]): View angle
            sense_direction (Optional[Union[float, Coordinate2dOr3d, str]]): sense direction
            expected_number_of_target_agents (int): Expected number of agents to be detected
        """
        fov_sensor = MockFieldOfViewSensor(view_angle=view_angle)

        with catch_warnings(record=True) as w:
            fov_sensor.run_detection(
                sensor_agent=self.sensor_agent_3d, sense_direction=sense_direction
            )
            perceived_agents = fov_sensor.get_results(sensor_agent=self.sensor_agent_3d)

        if sense_direction:
            assert len(w) == 0
        else:
            assert len(w) == 1
            assert "No sense direction provided, assuming positive x direction." in str(
                w[0].message
            )

        assert len(perceived_agents) == expected_number_of_target_agents

    def test_unknown_area_sense_direction(self):
        """Test sensor error handling when unknown area is given."""
        fov_sensor = MockFieldOfViewSensor(view_angle=90)

        with self.assertRaises(KeyError) as e:
            fov_sensor.run_detection(
                sensor_agent=self.sensor_agent_2d, sense_direction="Unknown Area"
            )
        assert "'Sense direction area (Unknown Area) not defined.'" == str(e.exception)
