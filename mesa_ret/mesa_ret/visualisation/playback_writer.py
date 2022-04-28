"""Interface for playback writers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mesa_ret.model import RetModel


class PlaybackWriter(ABC):
    """An abstract class to define the interface for writing RetPlay files."""

    @abstractmethod
    def model_start(self, model: RetModel) -> None:  # pragma: no cover
        """Record model information before the first step.

        Override in subclass.

        Args:
            model (RetModel): The model from which the activity will be recorded.
        """
        pass

    @abstractmethod
    def model_step(self, model: RetModel) -> None:  # pragma: no cover
        """Record one model steps worth of activity.

        Override in subclass.

        Args:
            model (RetModel): The model from which the activity will be recorded.
        """
        pass

    @abstractmethod
    def model_finish(self) -> None:  # pragma: no cover
        """Perform necessary actions for recording completion of the model.

        Override in subclass.
        """
        pass
