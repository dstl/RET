"""Component for model metadata information for rendering in UI."""

from dataclasses import dataclass


@dataclass
class ModelMetadata:
    """Model class metadata."""

    header: str
    subtext: list[str]
