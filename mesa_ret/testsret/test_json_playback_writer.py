"""Test cases for playback writers."""

from __future__ import annotations

import pathlib
from tempfile import TemporaryDirectory

from mesa_ret.agents.affiliation import Affiliation
from mesa_ret.agents.airagent import AirAgent
from mesa_ret.agents.groupagent import GroupAgent
from mesa_ret.agents.infantryagent import InfantryAgent
from mesa_ret.testing.mocks import MockModel2d, MockOrder
from mesa_ret.visualisation.json_writer import JsonWriter


def test_model_start():
    """Tests the JSON writer is initialised correctly."""
    with TemporaryDirectory() as td:
        model = MockModel2d()
        p = pathlib.Path(td, "test_icon.svg")
        icon_path = pathlib.Path(p)
        icon_path.write_text("")

        # Set JSON to output to temporary directory
        writer = JsonWriter(str(td))
        # Set icons to output to temporary directory
        writer.icon_handler.output_folder_name = str(td)

        writer.model_start(model)

        assert len(writer.json_results.step_data) == 0

        map_size = writer.json_results.initial_data.map_size

        assert map_size.x_min == model.space.x_min
        assert map_size.y_min == model.space.y_min
        assert map_size.x_max == model.space.x_max
        assert map_size.y_max == model.space.y_max


def test_model_step_no_agents():
    """Tests the JSON writer appends each model step, with no agents."""
    with TemporaryDirectory() as td:
        model = MockModel2d()
        p = pathlib.Path(td, "test_icon.svg")
        icon_path = pathlib.Path(p)
        icon_path.write_text("")

        # Set JSON to output to temporary directory
        writer = JsonWriter(str(td))
        # Set icons to output to temporary directory
        writer.icon_handler.output_folder_name = str(td)

        writer.model_start(model)
        step_data = writer.json_results.step_data

        assert len(step_data) == 0

        model.step()
        writer.model_step(model)

        assert len(step_data) == 1
        assert step_data[0].step_number == 1
        assert len(step_data[0].agents) == 0

        model.step()
        writer.model_step(model)

        assert len(step_data) == 2
        assert step_data[0].step_number == 1
        assert step_data[1].step_number == 2
        assert len(step_data[0].agents) == 0
        assert len(step_data[1].agents) == 0


def test_model_step_with_agents():
    """Tests the JSON writer appends each model step, with multiple agents."""
    with TemporaryDirectory() as td:
        model = MockModel2d()
        p = pathlib.Path(td, "test_icon.svg")
        icon_path = pathlib.Path(p)
        icon_path.write_text("")

        # Set JSON to output to temporary directory
        writer = JsonWriter(str(td))
        # Set icons to output to temporary directory
        writer.icon_handler.output_folder_name = str(td)

        air_agent = AirAgent(model, (0.0, 0.0), "Test Air Agent", Affiliation.FRIENDLY)
        infantry_agent = InfantryAgent(
            model, (5.0, 5.0), "Test Infantry Agent", Affiliation.NEUTRAL
        )
        writer.model_start(model)
        step_data = writer.json_results.step_data

        assert len(step_data) == 0

        model.step()
        writer.model_step(model)

        assert len(step_data) == 1
        assert step_data[0].step_number == 1
        assert len(step_data[0].agents) == 2

        json_agent_1 = step_data[0].agents[0]

        assert json_agent_1.affiliation == "FRIENDLY"
        assert json_agent_1.agent_type == "AIR"
        assert json_agent_1.name == "Test Air Agent"
        assert json_agent_1.pos == (0.0, 0.0)
        assert json_agent_1.killed == air_agent.killed
        assert json_agent_1.in_group == air_agent.in_group
        assert json_agent_1.mission_messages == air_agent.mission_messages

        json_agent_2 = step_data[0].agents[1]

        assert json_agent_2.affiliation == "NEUTRAL"
        assert json_agent_2.agent_type == "INFANTRY"
        assert json_agent_2.name == "Test Infantry Agent"
        assert json_agent_2.pos == (5.0, 5.0)
        assert json_agent_2.killed == infantry_agent.killed
        assert json_agent_2.in_group == infantry_agent.in_group
        assert json_agent_2.mission_messages == infantry_agent.mission_messages


def test_model_step_with_group_agent():
    """Tests the JSON writer appends each model step, with a group agent."""
    with TemporaryDirectory() as td:
        model = MockModel2d()
        p = pathlib.Path(td, "test_icon.svg")
        icon_path = pathlib.Path(p)
        icon_path.write_text("")

        # Set JSON to output to temporary directory
        writer = JsonWriter(str(td))
        # Set icons to output to temporary directory
        writer.icon_handler.output_folder_name = str(td)

        air_agent = AirAgent(model, (0.0, 0.0), "Test Air Agent", Affiliation.FRIENDLY)
        infantry_agent = InfantryAgent(
            model, (5.0, 5.0), "Test Infantry Agent", Affiliation.NEUTRAL
        )
        group_agent = GroupAgent(
            model, "Test Group Agent", Affiliation.FRIENDLY, agents=[air_agent]
        )
        writer.model_start(model)
        step_data = writer.json_results.step_data

        assert len(step_data) == 0

        model.step()
        writer.model_step(model)

        assert len(step_data) == 1
        assert step_data[0].step_number == 1
        assert len(step_data[0].agents) == 3

        json_agent_1 = step_data[0].agents[0]

        assert json_agent_1.name == "Test Infantry Agent"
        assert json_agent_1.affiliation == "NEUTRAL"
        assert json_agent_1.agent_type == "INFANTRY"
        assert json_agent_1.id == infantry_agent.unique_id
        assert json_agent_1.pos == (5.0, 5.0)
        assert json_agent_1.killed == infantry_agent.killed
        assert json_agent_1.in_group == infantry_agent.in_group
        assert json_agent_1.mission_messages == infantry_agent.mission_messages

        json_agent_2 = step_data[0].agents[1]

        assert json_agent_2.name == "Test Group Agent"
        assert json_agent_2.affiliation == "FRIENDLY"
        assert json_agent_2.agent_type == "GROUP"
        assert json_agent_2.id == group_agent.unique_id
        assert json_agent_2.pos == (0.0, 0.0)
        assert json_agent_2.killed == group_agent.killed
        assert json_agent_2.in_group == group_agent.in_group
        assert json_agent_2.mission_messages == group_agent.mission_messages

        json_agent_3 = step_data[0].agents[2]

        assert json_agent_3.name == "Test Air Agent"
        assert json_agent_3.affiliation == "FRIENDLY"
        assert json_agent_3.agent_type == "AIR"
        assert json_agent_3.id == air_agent.unique_id
        assert json_agent_3.pos == (0.0, 0.0)
        assert json_agent_3.killed == air_agent.killed
        assert json_agent_3.in_group == air_agent.in_group
        assert json_agent_3.mission_messages == air_agent.mission_messages


def test_model_step_with_active_orders():
    """Tests the JSON writer appends each model step, with an agents active order."""
    with TemporaryDirectory() as td:
        model = MockModel2d()
        p = pathlib.Path(td, "test_icon.svg")
        icon_path = pathlib.Path(p)
        icon_path.write_text("")

        # Set JSON to output to temporary directory
        writer = JsonWriter(str(td))
        # Set icons to output to temporary directory
        writer.icon_handler.output_folder_name = str(td)
        agent_order = MockOrder()
        air_agent = AirAgent(model, (0.0, 0.0), "Test Air Agent", Affiliation.FRIENDLY, orders=[])
        air_agent.active_order = agent_order
        writer.model_start(model)
        step_data = writer.json_results.step_data

        assert len(step_data) == 0

        model.step()
        writer.model_step(model)

        assert len(step_data) == 1
        assert step_data[0].step_number == 1
        assert len(step_data[0].agents) == 1

        json_agent_1 = step_data[0].agents[0]

        assert json_agent_1.affiliation == "FRIENDLY"
        assert json_agent_1.agent_type == "AIR"
        assert json_agent_1.name == "Test Air Agent"
        assert json_agent_1.pos == (0.0, 0.0)
        assert json_agent_1.killed == air_agent.killed
        assert json_agent_1.in_group == air_agent.in_group
        assert json_agent_1.active_order == "Mock Task and Mock Trigger"
        assert json_agent_1.mission_messages == air_agent.mission_messages


def test_model_step_with_no_active_orders():
    """Tests the JSON writer appends each model step, handling no active order correctly."""
    with TemporaryDirectory() as td:
        model = MockModel2d()
        p = pathlib.Path(td, "test_icon.svg")
        icon_path = pathlib.Path(p)
        icon_path.write_text("")

        # Set JSON to output to temporary directory
        writer = JsonWriter(str(td))
        # Set icons to output to temporary directory
        writer.icon_handler.output_folder_name = str(td)
        air_agent = AirAgent(model, (0.0, 0.0), "Test Air Agent", Affiliation.FRIENDLY, orders=[])
        writer.model_start(model)
        step_data = writer.json_results.step_data

        assert len(step_data) == 0

        model.step()
        writer.model_step(model)

        assert len(step_data) == 1
        assert step_data[0].step_number == 1
        assert len(step_data[0].agents) == 1

        json_agent_1 = step_data[0].agents[0]

        assert json_agent_1.affiliation == "FRIENDLY"
        assert json_agent_1.agent_type == "AIR"
        assert json_agent_1.name == "Test Air Agent"
        assert json_agent_1.pos == (0.0, 0.0)
        assert json_agent_1.killed == air_agent.killed
        assert json_agent_1.in_group == air_agent.in_group
        assert json_agent_1.active_order == "No active order"
        assert json_agent_1.mission_messages == air_agent.mission_messages


def test_model_step_with_agent_icon():
    """Tests the JSON writer appends each model step, with agent icons."""
    with TemporaryDirectory() as td:
        model = MockModel2d()
        p = pathlib.Path(td, "test_icon.svg")
        icon_path = pathlib.Path(p)
        icon_path.write_text("")

        # Set JSON to output to temporary directory
        writer = JsonWriter(str(td))
        # Set icons to output to temporary directory
        writer.icon_handler.output_folder_name = str(td)

        AirAgent(
            model,
            (0.0, 0.0),
            "Test Air Agent",
            Affiliation.FRIENDLY,
            icon_path=str(icon_path),
        )
        InfantryAgent(
            model,
            (5.0, 5.0),
            "Test Infantry Agent",
            Affiliation.NEUTRAL,
            icon_path=str(icon_path),
        )
        writer.model_start(model)
        step_data = writer.json_results.step_data

        assert len(step_data) == 0

        model.step()
        writer.model_step(model)

        assert pathlib.Path(td, icon_path).is_file()

        json_agent_1 = step_data[0].agents[0]
        assert "test_icon.svg" == json_agent_1.icon
        json_agent_2 = step_data[0].agents[1]
        assert "test_icon.svg" == json_agent_2.icon


def test_model_finish():
    """Tests the JSON writer saves correctly at model finish."""
    with TemporaryDirectory() as td:
        model = MockModel2d()
        p = pathlib.Path(td, "test_icon.svg")
        icon_path = pathlib.Path(p)
        icon_path.write_text("")

        # Set JSON to output to temporary directory
        writer = JsonWriter(str(td))

        writer.model_start(model)
        writer.model_finish()

        with open(pathlib.Path(td, "playback.json"), "r") as json_file:
            json_output_results = json_file.read()

        assert writer.json_results.json() == json_output_results

        writer.model_step(model)
        writer.model_finish()

        with open(pathlib.Path(td, "playback.json"), "r") as json_file:
            json_output_results = json_file.read()

        assert writer.json_results.json() == json_output_results


def test_existing_playback_file():
    """Test that a json file isn't written when one already exists."""
    with TemporaryDirectory() as td:
        model = MockModel2d()
        p = pathlib.Path(td, "playback.json")
        playback_path = pathlib.Path(p)
        playback_path.write_text("test_text")

        # Set JSON to output to temporary directory
        writer = JsonWriter(str(td))

        writer.model_start(model)
        writer.model_finish()

        with open(pathlib.Path(td, "playback.json"), "r") as json_file:
            json_output_results = json_file.read()

        assert json_output_results == "test_text"

        writer.model_step(model)
        writer.model_finish()

        with open(pathlib.Path(td, "playback.json"), "r") as json_file:
            json_output_results = json_file.read()

        assert json_output_results == "test_text"
