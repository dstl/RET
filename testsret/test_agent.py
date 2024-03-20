"""Basic tests for an agent."""

from datetime import timedelta
import unittest
from ret.agents.affiliation import Affiliation
from ret.agents.agent import AgentSensedStatus, RetAgent, WeaponFiredStatus
from ret.agents.agenttype import AgentType
from ret.testing.mocks import MockModel2d, MockWeapon
from ret.weapons.weapon import BasicWeapon


def test_weapon_copying():
    """Test that a weapon provided to two different agents does not have shared memory."""
    weapons = [
        BasicWeapon(
            name="Weapon to Copy",
            radius=1,
            kill_probability_per_round=1,
            time_before_first_shot=timedelta(seconds=1),
            time_between_rounds=timedelta(seconds=1),
        )
    ]

    model = MockModel2d()
    agent1 = RetAgent(
        name="Agent 1",
        model=model,
        pos=(0, 0),
        affiliation=Affiliation.NEUTRAL,
        critical_dimension=1.0,
        reflectivity=1.0,
        temperature=1.0,
        temperature_std_dev=1.0,
        agent_type=AgentType.GENERIC,
        weapons=weapons,
    )

    agent2 = RetAgent(
        name="Agent 2",
        model=model,
        pos=(0, 0),
        affiliation=Affiliation.NEUTRAL,
        critical_dimension=1.0,
        reflectivity=1.0,
        temperature=1.0,
        temperature_std_dev=1.0,
        agent_type=AgentType.GENERIC,
        weapons=weapons,
    )

    agent1.weapons[0]._kill_probability_per_round = 0

    assert agent2.weapons[0]._kill_probability_per_round == 1


def test_agent_sensed_by_statuses():
    """Test agent sensed by status methods."""
    model = MockModel2d()
    agent = RetAgent(
        name="Agent 1",
        model=model,
        pos=(0, 0),
        affiliation=Affiliation.NEUTRAL,
        critical_dimension=1.0,
        reflectivity=1.0,
        temperature=1.0,
        temperature_std_dev=1.0,
        agent_type=AgentType.GENERIC,
    )
    agent2 = RetAgent(
        name="Agent 2",
        model=model,
        pos=(0, 0),
        affiliation=Affiliation.NEUTRAL,
        critical_dimension=1.0,
        reflectivity=1.0,
        temperature=1.0,
        temperature_std_dev=1.0,
        agent_type=AgentType.GENERIC,
    )

    model_time = model.get_time()
    agent.add_sensed_by_status(model_time, agent2)

    assert len(agent._statuses) == 1
    assert len(agent.get_statuses()) == 0

    model.step()

    statuses = agent.get_statuses()
    assert len(statuses) == 1
    assert isinstance(statuses[0], AgentSensedStatus)
    assert statuses[0].time_step == model_time
    assert statuses[0].sensing_agent == agent2

    model.step()

    assert len(agent.get_statuses()) == 0


def test_agent_weapon_fired_status():
    """Test agent weapon fired status methods."""
    model = MockModel2d()
    agent = RetAgent(
        name="Agent 1",
        model=model,
        pos=(0, 0),
        affiliation=Affiliation.NEUTRAL,
        critical_dimension=1.0,
        reflectivity=1.0,
        temperature=1.0,
        temperature_std_dev=1.0,
        agent_type=AgentType.GENERIC,
    )
    weapon = MockWeapon()
    model_time = model.get_time()
    agent.add_fired_status(weapon=weapon, target_location=(0, 1), target_in_range=True)

    assert len(agent._statuses) == 1
    assert len(agent.get_statuses()) == 0

    model.step()

    statuses = agent.get_statuses()
    assert len(statuses) == 1
    assert isinstance(statuses[0], WeaponFiredStatus)
    assert statuses[0].time_step == model_time
    assert statuses[0].weapon_name == weapon.name
    assert statuses[0].target_location == (0, 1)

    model.step()

    assert len(agent.get_statuses()) == 0


def test_agent_multiple_statuses():
    """Test agent status methods where multiple status types are added."""
    model = MockModel2d()
    agent = RetAgent(
        name="Agent 1",
        model=model,
        pos=(0, 0),
        affiliation=Affiliation.NEUTRAL,
        critical_dimension=1.0,
        reflectivity=1.0,
        temperature=1.0,
        temperature_std_dev=1.0,
        agent_type=AgentType.GENERIC,
    )
    agent2 = RetAgent(
        name="Agent 2",
        model=model,
        pos=(0, 0),
        affiliation=Affiliation.NEUTRAL,
        critical_dimension=1.0,
        reflectivity=1.0,
        temperature=1.0,
        temperature_std_dev=1.0,
        agent_type=AgentType.GENERIC,
    )

    weapon = MockWeapon()
    model_time = model.get_time()

    agent.add_fired_status(weapon=weapon, target_location=(0, 1), target_in_range=True)
    agent.add_sensed_by_status(model_time, agent2)

    assert len(agent._statuses) == 2
    assert len(agent.get_statuses()) == 0

    model.step()

    statuses = agent.get_statuses()
    assert len(statuses) == 2

    assert isinstance(statuses[0], WeaponFiredStatus)
    assert statuses[0].time_step == model_time
    assert statuses[0].weapon_name == weapon.name
    assert statuses[0].target_location == (0, 1)

    assert isinstance(statuses[1], AgentSensedStatus)
    assert statuses[1].time_step == model_time
    assert statuses[1].sensing_agent == agent2

    model.step()

    assert len(agent.get_statuses()) == 0


class Test(unittest.TestCase):
    """Test Case for batchrunner."""

    def test_reflectivity_init(self):
        """Tests if reflectivity is outside acceptable range."""
        reflectivity = 5.0
        """Test if warning is raised for large reflectivity."""
        with self.assertRaises(ValueError) as e:
            RetAgent(
                name="Agent 1",
                model=MockModel2d(),
                pos=(0, 0),
                affiliation=Affiliation.NEUTRAL,
                critical_dimension=1.0,
                reflectivity=reflectivity,
                temperature=1.0,
                temperature_std_dev=1.0,
                agent_type=AgentType.GENERIC,
            )
            self.assertEqual(
                f"Agent reflectivity ({reflectivity}) must be greater than 0 and "
                "less than or equal to 1.",
                str(e.exception),
            )
