"""Version 1 RET Model Definition File IO Layer."""


from datetime import datetime, timedelta

from pydantic import BaseModel

from . import v2


class SpaceSchema(BaseModel):
    """Model for V1 Space Definition."""

    y_max: int
    x_max: int

    def upgrade(self) -> v2.SpaceSchema:
        """Upgrade V2 space definition.

        Returns:
            v2.SpaceDefinition: V2 Space Definition
        """
        return v2.SpaceSchema(
            x_min=0,
            y_min=0,
            dimensions=2,
            x_max=self.x_max,
            y_max=self.y_max,
            terrain_image_path="",
            height_black=1,
            height_white=0,
            culture_image_path="",
            cultures=[],
            clutter_background_level=0,
            ground_clutter_height=0,
            ground_clutter_value=0,
        )


class RetModelSchema(BaseModel):
    """Model for V1 Model Definition."""

    agents: int
    space: SpaceSchema

    def upgrade(self) -> v2.RetModelSchema:
        """Upgrade v1.RetModelSchema to v2.RetModelSchema.

        Automatic upgrade from V1 to V2 makes the following assumptions:

        - The model has a single parameter, based on the number of agents, which
            can be varied in the range 1 - v1.agents.

        Returns:
            v2.RetModelSchema: [description]
        """
        start_time = datetime(year=2021, month=1, day=1)
        end_time = datetime(year=2021, month=1, day=2)
        time_step = timedelta(minutes=5)

        time = v2.TimeSchema(start_time=start_time, end_time=end_time, time_step=time_step)

        numeric_parameters = {
            "n_agents": v2.NumericParameterSchema(
                name="n_agents", min_val=1, max_val=10, distribution="range"
            )
        }

        v2_schema = v2.RetModelSchema(
            time=time,
            space=self.space.upgrade(),
            model_name="RetModel",
            model_module_name="mesa_ret.model",
            max_steps=1000,
            iterations=10,
            experimental_controls=v2.ExperimentalControlsSchema(
                numeric_parameters=numeric_parameters, categoric_parameters={}
            ),
            scenario_dependent_parameters=v2.ScenarioDependentParametersSchema(
                numeric_parameters={}, categoric_parameters={}
            ),
            playback_writer="None",
        )

        return v2_schema
