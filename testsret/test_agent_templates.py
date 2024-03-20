"""Examination of template test cases for basic agent types."""

from unittest.case import TestCase

from ret.agents.agent import RetAgent
from ret.agents.airagent import AirAgent
from ret.agents.airdefenceagent import AirDefenceAgent
from ret.agents.infantryagent import InfantryAgent
from ret.agents.protectedassetagent import ProtectedAssetAgent
from ret.testing.templates.agent import (
    RetAgentNoBehavioursTemplate,
    RetAgentTemplate,
    RetAgentTemplateWithCountermeasures,
)


class TestRetAgent(RetAgentTemplateWithCountermeasures, TestCase):
    """Examination of template test cases for RetAgent."""

    def create_agent(self) -> RetAgent:
        """Return a new RetAgent.

        Returns:
            RetAgent: Agent under test.
        """
        return RetAgent(
            model=self.model,
            pos=self.pos,
            name=self.name,
            affiliation=self.affiliation,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            orders=self.orders,
            behaviours=[
                self.move_behaviour,
                self.wait_behaviour,
                self.hide_behaviour,
                self.sense_behaviour,
                self.communicate_orders_behaviour,
                self.communicate_worldview_behaviour,
                self.fire_behaviour,
                self.disable_communication_behaviour,
                self.deploy_countermeasure_behaviour,
            ],
            sensors=self.sensors,
            communication_receiver=self.communication_receiver,
            refresh_technique=self.refresh_technique,
            countermeasures=self.countermeasures,
        )


class TestInfantryAgent(RetAgentTemplate, TestCase):
    """Test that the RetAgentTemplate tests can be run on an InfantryAgent."""

    def create_agent(self) -> InfantryAgent:
        """Return a new InfantryAgent.

        Returns:
            InfantryAgent: agent under test.
        """
        return InfantryAgent(
            model=self.model,
            pos=self.pos,
            name=self.name,
            affiliation=self.affiliation,
            orders=self.orders,
            behaviours=[
                self.move_behaviour,
                self.wait_behaviour,
                self.hide_behaviour,
                self.sense_behaviour,
                self.communicate_orders_behaviour,
                self.communicate_worldview_behaviour,
                self.fire_behaviour,
                self.disable_communication_behaviour,
            ],
            sensors=self.sensors,
            communication_receiver=self.communication_receiver,
            refresh_technique=self.refresh_technique,
        )


class TestAirDefenceAgent(RetAgentTemplate, TestCase):
    """Test that the RetAgentTemplate tests can be run on an AirDefenceAgent."""

    def create_agent(self) -> AirDefenceAgent:
        """Return a new AirDefenceAgent.

        Returns:
            AirDefenceAgent: agent under test.
        """
        return AirDefenceAgent(
            model=self.model,
            pos=self.pos,
            name=self.name,
            affiliation=self.affiliation,
            orders=self.orders,
            behaviours=[
                self.move_behaviour,
                self.wait_behaviour,
                self.hide_behaviour,
                self.sense_behaviour,
                self.communicate_orders_behaviour,
                self.communicate_worldview_behaviour,
                self.fire_behaviour,
                self.disable_communication_behaviour,
            ],
            sensors=self.sensors,
            communication_receiver=self.communication_receiver,
            refresh_technique=self.refresh_technique,
        )


class TestProtectedAssetAgent(RetAgentNoBehavioursTemplate, TestCase):
    """Test that can be run on a ProtectedAssetAgent."""

    def create_agent(self) -> ProtectedAssetAgent:
        """Return a new ProtectedAssetAgent.

        Returns:
            ProtectedAssetAgent: agent under test.
        """
        return ProtectedAssetAgent(
            model=self.model,
            pos=self.pos,
            name=self.name,
            affiliation=self.affiliation,
        )

    def test_agent_step(self):
        """Test that stepping an agent doesn't move it's position."""
        self.agent.step()
        assert self.agent.pos[0] == 0
        assert self.agent.pos[1] == 0

    def test_model_step(self):
        """Test that stepping the model doesn't move an the agent's position."""
        self.model.step()
        assert self.agent.pos[0] == 0
        assert self.agent.pos[1] == 0

    def test_check_for_new_active_order(self):
        """Test that the protected asset agent can check for new orders."""
        assert self.agent.active_order is None
        assert len(self.agent._orders) == 0
        self.agent.check_for_new_active_order()
        assert self.agent.active_order is None
        assert len(self.agent._orders) == 0


class TestAirAgent(RetAgentTemplateWithCountermeasures, TestCase):
    """Test that the RetAgentTemplate tests can be run on a AirAgent."""

    def create_agent(self) -> AirAgent:
        """Return a new AirAgent.

        Returns:
            AirAgent: agent under test.
        """
        return AirAgent(
            model=self.model,
            pos=self.pos,
            name=self.name,
            affiliation=self.affiliation,
            orders=self.orders,
            behaviours=[
                self.move_behaviour,
                self.wait_behaviour,
                self.hide_behaviour,
                self.sense_behaviour,
                self.communicate_orders_behaviour,
                self.communicate_worldview_behaviour,
                self.fire_behaviour,
                self.disable_communication_behaviour,
                self.deploy_countermeasure_behaviour,
            ],
            sensors=self.sensors,
            communication_receiver=self.communication_receiver,
            refresh_technique=self.refresh_technique,
            countermeasures=self.countermeasures,
        )


class TestRetAgentNoBehaviour(RetAgentNoBehavioursTemplate, TestCase):
    """Test that the RetAgentNoBehavioursTemplate tests can be run on a RetAgent."""

    def create_agent(self):
        """Return a new RetAgent.

        Returns:
            RetAgent: agent under test.
        """
        return RetAgent(
            model=self.model,
            pos=self.pos,
            name=self.name,
            affiliation=self.affiliation,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            orders=self.orders,
        )
