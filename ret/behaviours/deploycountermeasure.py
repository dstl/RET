"""Tests for deployment of a countermeasure."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ret.behaviours.loggablebehaviour import LoggableBehaviour

if TYPE_CHECKING:
    from ret.agents.agent import RetAgent
    from ret.space.clutter.countermeasure import Countermeasure


class DeployCountermeasureBehaviour(LoggableBehaviour):
    """A class representing deploying a countermeasure."""

    def __init__(self, log: bool = True) -> None:
        """Create behaviour.

        Args:
            log (bool): If True logging occurs for this behaviour, if False it does not.
                Defaults to True.
        """
        super().__init__(log)

    def step(self, deployer: RetAgent, countermeasure: Countermeasure) -> None:
        """Do one time steps worth of countermeasure deployment and log.

        Args:
            deployer (RetAgent): The agent doing the deployment
            countermeasure (Countermeasure): The countermeasure being deployed
        """
        self.log(deployer)
        self._step(deployer, countermeasure)

    def _step(self, deployer: RetAgent, countermeasure: Countermeasure) -> None:
        """Do one time steps worth of countermeasure deployment.

        Args:
            deployer (RetAgent): The agent doing the deployment
            countermeasure (Countermeasure): The countermeasure being deployed
        """
        countermeasure.deploy(deployer)
