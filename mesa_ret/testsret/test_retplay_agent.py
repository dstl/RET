"""Test the RetPlay Agent."""
import pathlib
import warnings
from tempfile import TemporaryDirectory

from mesa_ret.agents.affiliation import Affiliation
from mesa_ret.agents.agent import RetAgent
from mesa_ret.sensing.agentcasualtystate import AgentCasualtyState
from mesa_ret.sensing.perceivedworld import Confidence, PerceivedAgent
from mesa_ret.testing.mocks import MockFriendlyRetPlayAgent, MockModel2d, MockRetPlayMapFromPNG
from mesa_ret.visualisation.json_models import RetPlayAgent, RetPlayPerceivedAgent
from parameterized.parameterized import parameterized


def test_creation_retplay_agent():
    """Test the initialisation of the RetPlayAgent."""
    model = MockModel2d()
    ret_agent = RetAgent(
        model,
        (0, 0),
        "Test Agent",
        Affiliation.NEUTRAL,
        critical_dimension=2.0,
        reflectivity=0.1,
        temperature=20.0,
    )
    retplay_agent = RetPlayAgent.from_model(ret_agent)

    assert retplay_agent.affiliation == str(ret_agent.affiliation.name)
    assert retplay_agent.agent_type == str(ret_agent.agent_type.name)
    assert retplay_agent.id == ret_agent.unique_id
    assert retplay_agent.name == ret_agent.name
    assert retplay_agent.killed == ret_agent.killed
    assert retplay_agent.perceived_agents == []
    assert retplay_agent.pos == ret_agent.pos


def test_adding_perceived_agent():
    """Test RetPlayAgents correctly add RetPlayPerceivedAgents to their perceived world."""
    model = MockModel2d()
    ret_agent = RetAgent(
        model,
        (0, 0),
        "Test Agent",
        Affiliation.NEUTRAL,
        critical_dimension=2.0,
        reflectivity=0.1,
        temperature=20.0,
    )
    ret_agent_to_perceive = RetAgent(
        model,
        (0, 0),
        "Test Perceive Agent",
        Affiliation.HOSTILE,
        critical_dimension=2.0,
        reflectivity=0.1,
        temperature=20.0,
    )
    ret_agent.perceived_world.add_acquisitions(
        PerceivedAgent.to_perceived_agent(ret_agent_to_perceive)
    )

    retplay_agent = RetPlayAgent.from_model(ret_agent)

    assert len(retplay_agent.perceived_agents) == 1
    assert retplay_agent.perceived_agents[0].name is None
    assert retplay_agent.perceived_agents[0].sense_time == "2020-01-01 00:00:00"
    assert retplay_agent.perceived_agents[0].affiliation == str(ret_agent_to_perceive.affiliation)
    assert retplay_agent.perceived_agents[0].agent_type == str(ret_agent_to_perceive.agent_type)
    assert retplay_agent.perceived_agents[0].casualty_state == str(AgentCasualtyState.ALIVE)
    assert retplay_agent.perceived_agents[0].casualty_state_confidence == str(Confidence.RECOGNISE)
    assert retplay_agent.perceived_agents[0].icon is None
    assert retplay_agent.perceived_agents[0].location == ret_agent_to_perceive.pos
    assert retplay_agent.perceived_agents[0].unique_id == 2
    assert retplay_agent.perceived_agents[0].confidence == str(Confidence.KNOWN)


def test_creation_retplay_perceived_agent_from_ret_agent():
    """Test the initialisation of the RetPlayPerceivedAgent from a RetAgent."""
    model = MockModel2d()
    ret_agent = RetAgent(
        model,
        (0, 0),
        "Test Agent",
        Affiliation.NEUTRAL,
        critical_dimension=2.0,
        reflectivity=0.1,
        temperature=20.0,
    )
    retplay_perceived_agent = RetPlayPerceivedAgent.get_retplay_perceived_agent_from_known_agent(
        ret_agent
    )

    assert retplay_perceived_agent.name == "Test Agent"
    assert retplay_perceived_agent.sense_time is None
    assert retplay_perceived_agent.affiliation == str(ret_agent.affiliation)
    assert retplay_perceived_agent.agent_type == str(ret_agent.agent_type)
    assert retplay_perceived_agent.casualty_state == str(AgentCasualtyState.ALIVE)
    assert retplay_perceived_agent.casualty_state_confidence is None
    assert retplay_perceived_agent.icon == "genericagent_neutral.svg"
    assert retplay_perceived_agent.location == ret_agent.pos
    assert retplay_perceived_agent.unique_id == 1
    assert retplay_perceived_agent.confidence is None


def test_creation_retplay_perceived_agent_from_perceived_agent():
    """Test the initialisation of the RetPlayPerceivedAgent from a PerceivedAgent."""
    model = MockModel2d()
    ret_agent = RetAgent(
        model,
        (0, 0),
        "Test Agent",
        Affiliation.NEUTRAL,
        critical_dimension=2.0,
        reflectivity=0.1,
        temperature=20.0,
    )
    ret_agent_to_perceive = RetAgent(
        model,
        (0, 0),
        "Test Perceive Agent",
        Affiliation.HOSTILE,
        critical_dimension=2.0,
        reflectivity=0.1,
        temperature=20.0,
    )
    ret_agent.perceived_world.add_acquisitions(
        PerceivedAgent.to_perceived_agent(ret_agent_to_perceive)
    )
    retplay_perceived_agent = (
        RetPlayPerceivedAgent.get_retplay_perceived_agent_from_perceived_agent(
            ret_agent.perceived_world._perceived_agents[0]
        )
    )

    assert retplay_perceived_agent.name is None
    assert retplay_perceived_agent.sense_time == "2020-01-01 00:00:00"
    assert retplay_perceived_agent.affiliation == str(ret_agent_to_perceive.affiliation)
    assert retplay_perceived_agent.agent_type == str(ret_agent_to_perceive.agent_type)
    assert retplay_perceived_agent.casualty_state == str(AgentCasualtyState.ALIVE)
    assert retplay_perceived_agent.casualty_state_confidence == str(Confidence.RECOGNISE)
    assert retplay_perceived_agent.icon is None
    assert retplay_perceived_agent.location == ret_agent_to_perceive.pos
    assert retplay_perceived_agent.unique_id == 2
    assert retplay_perceived_agent.confidence == str(Confidence.KNOWN)


def test_get_path_to_icon():
    """Test icon paths are correctly obtained."""
    with TemporaryDirectory() as td:
        p = pathlib.Path(td, "assets/icons")
        p.mkdir(parents=True, exist_ok=True)
        icon_path = pathlib.Path(p, "test_icon.svg")
        icon_path.write_text("")

        mock_retplay_map = MockRetPlayMapFromPNG(pathlib.Path("assets"))
        mock_retplay_map.retplay_root_directory = pathlib.Path(td + "/")
        agent = MockFriendlyRetPlayAgent()
        agent.icon = "test_icon.svg"
        assert mock_retplay_map.get_path_to_icon(agent) == str(
            pathlib.Path("assets", "icons", "test_icon.svg")
        )


@parameterized.expand(
    [
        [
            "FRIENDLY",
            str(pathlib.Path("assets", "default_icons", "genericagent_friendly.svg")),
        ],
        [
            "HOSTILE",
            str(pathlib.Path("assets", "default_icons", "genericagent_hostile.svg")),
        ],
        [
            "NEUTRAL",
            str(pathlib.Path("assets", "default_icons", "genericagent_neutral.svg")),
        ],
        [
            "UNKNOWN",
            str(pathlib.Path("assets", "default_icons", "genericagent_unknown.svg")),
        ],
    ]
)
def test_get_path_to_icon_not_found(agent_affiliation: str, icon_path: str):
    """Test the correct affiliation icon is found if the specified icon is missing.

    Args:
        agent_affiliation (str): Agent affiliation
        icon_path (str): Path to icon representing agent
    """
    agent = MockFriendlyRetPlayAgent()
    agent.affiliation = agent_affiliation
    mock_retplay_map = MockRetPlayMapFromPNG(pathlib.Path("assets", "default_icons"))
    assert mock_retplay_map.get_path_to_icon(agent) == str(pathlib.Path(icon_path))


def test_icon_with_unrecognised_affiliation():
    """Tests icons for unrecognised affiliations if the specified icon is missing."""
    with warnings.catch_warnings(record=True) as w:
        agent = MockFriendlyRetPlayAgent()
        agent.affiliation = "not_an_affiliation"
        mock_retplay_map = MockRetPlayMapFromPNG(pathlib.Path("assets"))
        assert mock_retplay_map.get_path_to_icon(agent) == ""
        assert len(w) == 1
        assert "Unrecognised affiliation type" in str(w[0].message)
