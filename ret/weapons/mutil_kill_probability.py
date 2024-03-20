"""Code constructs for handling Weapon kill probability by type.

Contains class and validator
"""
from ret.agents.agenttype import AgentType
from pydantic import BaseModel


class ProbabilityByType(BaseModel):
    """Class to hold multiple kill probabilities by enemy agent types."""

    kill_probability_by_agent_type: dict[AgentType, float]
    base_kill_probability_per_round: float

    def validate(self):
        """Validate the kill_probability_by_agent_type & base_kill_probability_per_round property.

        This validator checks that kill_probability_by_agent_type is not empty and that all its
        values are floats between 0 and 1.
        Also checks base_kill_probability_per_round is between 0 and 1.

        Raises:
            ValueError: If property is empty or if any value in the dictionary is not
                within the range [0, 1] or if base_kill_probability_per_round is
                also not within range [0, 1] or any types are incorrect
        """
        if not self.kill_probability_by_agent_type:
            raise ValueError("kill_probability_by_agent_type is empty")
        for agent_type, value in self.kill_probability_by_agent_type.items():
            if not 0 <= value <= 1 or not isinstance(value, float):
                raise ValueError(
                    f"Invalid kill probability for {agent_type}: {value}."
                    + "Must be a float between 0 and 1."
                )
            if isinstance(agent_type, AgentType):
                pass
        if not 0 <= self.base_kill_probability_per_round <= 1:
            raise ValueError("Invalid base_kill_probability_per_round. Must between 0 and 1.")

    def get_probability(self, target_type: AgentType) -> float:
        """Retrieves the correct kill probability for a given target type.

        Args:
            target_type (AgentType): The type of agent that is being fired at

        Returns:
            float: kill probability for target type
        """
        if target_type in self.kill_probability_by_agent_type:
            return self.kill_probability_by_agent_type[target_type]
        else:
            return self.base_kill_probability_per_round
