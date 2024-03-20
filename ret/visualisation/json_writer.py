"""Write to JSON."""

from __future__ import annotations

import pathlib
import os
from typing import TYPE_CHECKING

from ret.visualisation.json_icon_handler import IconCopier
from ret.visualisation.json_models import JsonOutObject, RetPlayAgent
from ret.visualisation.playback_writer import PlaybackWriter
from ret.utilities.save_utilities import get_latest_subfolder

if TYPE_CHECKING:
    from ret.agents.agent import RetAgent
    from ret.model import RetModel


class JsonWriter(PlaybackWriter):
    """A class to write the model data to a JSON playback file."""

    def __init__(
        self,
        output_folder_name: str = "./output",
        playback_file_name: str = "playback.json",
    ) -> None:
        """Create a JSON writer object ready to be written to.

        Args:
            output_folder_name (str): Name of output folder that the playback writer writes to.
                Defaults to a relative folder "./output"
            playback_file_name (str): Name of playback file produced, Defaults to "playback.json"
        """
        self.output_folder_name = output_folder_name
        self.playback_file_name = playback_file_name

        self.icon_handler = IconCopier(self.output_folder_name)

        self.write_playback = False
        if not pathlib.Path(self.output_folder_name, self.playback_file_name).exists():
            self.write_playback = True

    def model_start(self, model: RetModel) -> None:
        """Write the initial data to the JSON model.

        Args:
            model (RetModel): The model from which the data will be extracted.
        """
        if self.write_playback:
            self.json_results: JsonOutObject = JsonOutObject.from_model(model.space)

    def model_step(self, model: RetModel) -> None:
        """Write one time's worth of data to the JSON model.

        Args:
            model (RetModel): The model from which the data will be extracted.
        """
        if self.write_playback:
            agents_to_log: list[RetAgent] = []
            for agent in model.get_all_agents():
                agents_to_log.append(agent)  # type: ignore

            self.json_results.add_timestep_data(
                model.schedule.steps,
                model.get_time(),
                [RetPlayAgent.from_model(agent, self.icon_handler) for agent in agents_to_log],
            )

    def _model_finish(self) -> None:
        """Write the accumulated data to a JSON file."""
        if self.write_playback:
            with open(
                pathlib.Path(self.output_folder_name, self.playback_file_name), "w"
            ) as json_file:
                json_file.write(self.json_results.json())

    def model_finish(self) -> None:
        """Prevent errors from delays in datetime folder setup."""
        try:
            self._model_finish()
        except FileNotFoundError:
            subfolder = get_latest_subfolder("output")
            if subfolder is None:
                self.output_folder_name = "output"
            else:
                self.output_folder_name = os.path.join("output", subfolder)
            self._model_finish()
