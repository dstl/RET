"""Sensor fusion receiver."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ret.communication.communicationreceiver import CommunicationReceiver, WorldviewHandler

if TYPE_CHECKING:
    from ret.agents.sensorfusionagent import SensorFusionAgent
    from ret.sensing.perceivedworld import PerceivedAgent


class FusionWorldviewHandler(WorldviewHandler):
    """Communication Handler for Worldview for a Sensor Fusion Agent.

    Behaves identically to the Worldview Handler, except that it updates the Sensor
    Fusion Agent to notify that it has received new information.
    """

    def receive(self, receiver: SensorFusionAgent, payload: list[PerceivedAgent]) -> None:
        """Receive new worldview.

        Args:
            receiver (SensorFusionAgent): The agent receiving the information
            payload (list[PerceivedAgent]): The information being received
        """
        receiver.new_info = True
        super().receive(receiver, payload)
        receiver.model.logger.log_perception_record(receiver)


class SensorFusionReceiver(CommunicationReceiver):
    """Sensor fusion receiver."""

    def __init__(self):
        """Create a new SensorFusionReceiver."""
        handlers = [("worldview", FusionWorldviewHandler())]
        super().__init__(handlers)
