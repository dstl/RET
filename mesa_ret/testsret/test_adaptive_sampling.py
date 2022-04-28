"""Tests for adaptive sampling."""
from __future__ import annotations

import os
import random
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
from mesa.time import RandomActivation
from mesa_ret.agents.affiliation import Affiliation
from mesa_ret.agents.agent import RetAgent
from mesa_ret.agents.agenttype import AgentType
from mesa_ret.model import RetModel
from mesa_ret.parameters import SampleableContinuousModelParameter
from mesa_ret.sampling import AdaptiveSamplerInvalidDefaultValueError, GradientAdaptiveSampler
from mesa_ret.sensing.agentcasualtystate import AgentCasualtyState
from mesa_ret.space.space import ContinuousSpaceWithTerrainAndCulture3d
from mesa_ret.testing.mocks import MockParametrisedModel  # noqa: TC002
from mesa_ret.visualisation.json_writer import JsonWriter
from parameterized import parameterized
from pytest import raises

if TYPE_CHECKING:
    from mesa_ret.sampling import Sampler
    from mesa_ret.parameters import SampleableParameter

from mesa_ret.resampling import run_resampling


class MockParametrisedModelWithGradient2d(MockParametrisedModel):
    """Mock parametrised model with 2D result with high gradient.

    See example in
    https://adaptive.readthedocs.io/en/latest/tutorial/tutorial.Learner2D.html

    Area of high gradient is a ring around the origin.
    """

    def __init__(self, radius: float, **kwargs: str):
        """Initialise mock parametrised model.

        Parameters for this model are 'x' and 'y', both are in the range
        [-1,1]. The area of high gradient is a ring around the origin at a given
        distance from the center.

        Args:
            radius (float): The radius of the ring of high gradient.
        """
        super().__init__(**kwargs)

        # overwrite result
        self.result = self._get_result(radius, **kwargs)

    def _get_result(self, radius: float, **kwargs: str) -> float:
        """Calculate the result based on the values of x and y.

        Args:
            radius (float): The radius of the ring of high gradient in
                the 2D result space.

        Returns:
            float: The result.
        """
        x = float(kwargs["x"])
        y = float(kwargs["y"])
        a = 0.2
        return x + float(np.exp(-((x ** 2 + y ** 2 - radius ** 2) ** 2) / a ** 4))


def test_gradient_adaptive_sampler_2d():
    """Test adaptive sampler with two parameters."""
    model_cls: type = MockParametrisedModel
    params: list[SampleableParameter] = [
        SampleableContinuousModelParameter("var_1", min_val=1, max_val=4),
        SampleableContinuousModelParameter("var_2", min_val=10, max_val=40),
    ]
    gradient_adaptive_sampler: Sampler = GradientAdaptiveSampler(random=random.Random(1))
    iterations = 5
    n, experiments, _, _ = run_resampling(
        model_cls=model_cls,
        variable_params=params,
        sampler=gradient_adaptive_sampler,
        response=lambda result: result["result"],
        iterations=iterations,
        model_reporters={"result": lambda m: m.result},
    )
    assert len(experiments) == iterations
    assert n == [1] * iterations


def test_gradient_adaptive_sampler_3d():
    """Test adaptive sampler with three parameters."""
    model_cls: type = MockParametrisedModel
    params: list[SampleableParameter] = [
        SampleableContinuousModelParameter("var_1", min_val=1, max_val=4),
        SampleableContinuousModelParameter("var_2", min_val=10, max_val=40),
        SampleableContinuousModelParameter("var_3", min_val=2, max_val=8),
    ]
    gradient_adaptive_sampler: Sampler = GradientAdaptiveSampler(random=random.Random(1))
    iterations = 200
    n, experiments, _, _ = run_resampling(
        model_cls=model_cls,
        variable_params=params,
        sampler=gradient_adaptive_sampler,
        response=lambda result: result["result"],
        iterations=iterations,
        model_reporters={"result": lambda m: m.result},
    )
    assert len(experiments) == iterations
    assert n == [1] * iterations


def test_gradient_adaptive_sampler_random_1d():
    """Test adaptive sampler returns the same result each time."""
    model_cls: type = MockParametrisedModel
    seed = 1

    # 1D parameter space ensures than samples are chosen randomly as
    # triangulation cannot occur
    params: list[SampleableParameter] = [
        SampleableContinuousModelParameter("var_1", min_val=0, max_val=100),
    ]
    gradient_adaptive_sampler1: Sampler = GradientAdaptiveSampler(random=random.Random(seed))
    n1, experiments1, _, _ = run_resampling(
        model_cls=model_cls,
        variable_params=params,
        sampler=gradient_adaptive_sampler1,
        response=lambda result: result["result"],
        iterations=50,
        model_reporters={"result": lambda m: m.result},
    )

    gradient_adaptive_sampler2: Sampler = GradientAdaptiveSampler(
        random=random.Random(seed), default_result=0.0
    )
    n2, experiments2, _, _ = run_resampling(
        model_cls=model_cls,
        variable_params=params,
        sampler=gradient_adaptive_sampler2,
        response=lambda result: result["result"],
        iterations=50,
        model_reporters={"result": lambda m: m.result},
    )

    assert n1 == n2
    assert experiments1 == experiments2


@parameterized.expand([[0.3], [0.5], [0.75]])
def test_gradient_adaptive_sampler_example_2d(radius: float):
    """Test adaptive sampler with two parameters."""
    model_cls: type = MockParametrisedModelWithGradient2d
    params: list[SampleableParameter] = [
        SampleableContinuousModelParameter("x", min_val=-1, max_val=1),
        SampleableContinuousModelParameter("y", min_val=-1, max_val=1),
    ]
    gradient_adaptive_sampler: Sampler = GradientAdaptiveSampler(random=random.Random(1))
    iterations = 200
    n, experiments, _, _ = run_resampling(
        model_cls=model_cls,
        variable_params=params,
        sampler=gradient_adaptive_sampler,
        response=lambda result: result["result"],
        iterations=iterations,
        model_reporters={"result": lambda m: m.result},
        fixed_parameters={"radius": radius},
    )
    freq, bin_edges = np.histogram(
        [
            np.sqrt(float(experiment[0]["x"]) ** 2 + float(experiment[0]["y"]) ** 2)
            for experiment in experiments
        ],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.42],
    )
    # Find the bin with the largest frequency and check that it contains the radius
    i_max = np.argmax(freq)
    assert bin_edges[i_max] < radius < bin_edges[i_max + 1]

    assert len(experiments) == iterations


@parameterized.expand([[0.3], [0.75]])
def test_gradient_adaptive_sampler_2d_example_with_validity_logic(radius: float):
    """Test adaptive sampler with two parameters and validity logic."""
    model_cls: type = MockParametrisedModelWithGradient2d
    params: list[SampleableParameter] = [
        SampleableContinuousModelParameter("x", min_val=-1, max_val=1),
        SampleableContinuousModelParameter("y", min_val=-1, max_val=1),
    ]
    gradient_adaptive_sampler: Sampler = GradientAdaptiveSampler(
        random=random.Random(1), default_result=0.0
    )
    iterations = 300
    n_valid, experiments, n_invalid, invalid_experiments = run_resampling(
        model_cls=model_cls,
        variable_params=params,
        sampler=gradient_adaptive_sampler,
        response=lambda result: result["result"],
        iterations=iterations,
        validity_check=lambda params: params["x"] > 0,
        fixed_parameters={"radius": radius},
        model_reporters={"result": lambda m: m.result},
    )
    freq, bin_edges = np.histogram(
        [
            np.sqrt(float(experiment[0]["x"]) ** 2 + float(experiment[0]["y"]) ** 2)
            for experiment in experiments
        ],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.42],
    )
    # Find the bin with the largest frequency and check that it contains the radius
    i_max = np.argmax(freq)
    assert bin_edges[i_max] < radius < bin_edges[i_max + 1]

    assert len(n_valid) == iterations
    assert len(n_invalid) == iterations
    assert len(experiments) + len(invalid_experiments) == iterations
    assert sum(n_valid) == iterations - sum(n_invalid)

    for param_values, _ in experiments:
        assert param_values["x"] > 0

    for param_values, _ in invalid_experiments:
        assert param_values["x"] <= 0


def test_gradient_adaptive_sampler_2d_example_with_validity_logic_default_value_error():
    """Test an error is raised when default is None for adaptive sampler with invalid regions."""
    model_cls: type = MockParametrisedModelWithGradient2d
    params: list[SampleableParameter] = [
        SampleableContinuousModelParameter("x", min_val=-1, max_val=1),
        SampleableContinuousModelParameter("y", min_val=-1, max_val=1),
    ]
    gradient_adaptive_sampler: Sampler = GradientAdaptiveSampler(random=random.Random(1))
    iterations = 3
    with raises(AdaptiveSamplerInvalidDefaultValueError) as e:
        run_resampling(
            model_cls=model_cls,
            variable_params=params,
            sampler=gradient_adaptive_sampler,
            response=lambda result: result["result"],
            iterations=iterations,
            validity_check=lambda params: params["x"] > 0,
            fixed_parameters={"radius": 0.5},
            model_reporters={"result": lambda m: m.result},
        )
    assert (
        e.value.args[0]
        == "Adaptive sampler default value cannot be None if there are invalid regions."
    )


def test_gradient_adaptive_sampler_3d_example():
    """Test adaptive sampler with three parameters.

    Adaptive sampler uses triangulation for >= 3 parameters.
    """
    radius = 0.5
    model_cls: type = MockParametrisedModelWithGradient2d
    params: list[SampleableParameter] = [
        SampleableContinuousModelParameter("x", min_val=-1, max_val=1),
        SampleableContinuousModelParameter("y", min_val=-1, max_val=1),
        SampleableContinuousModelParameter("z", min_val=1, max_val=1.1),
    ]
    gradient_adaptive_sampler: Sampler = GradientAdaptiveSampler(random=random.Random(1))
    iterations = 400
    n, experiments, _, _ = run_resampling(
        model_cls=model_cls,
        variable_params=params,
        sampler=gradient_adaptive_sampler,
        response=lambda result: result["result"],
        iterations=iterations,
        model_reporters={"result": lambda m: m.result},
        fixed_parameters={"radius": radius},
    )
    freq, bin_edges = np.histogram(
        [
            np.sqrt(float(experiment[0]["x"]) ** 2 + float(experiment[0]["y"]) ** 2)
            for experiment in experiments
        ],
        bins=[0, 0.35, 0.7, 1.05, 1.42],
    )
    # Find the bin with the largest frequency and check that it contains the radius
    i_max = np.argmax(freq)
    assert bin_edges[i_max] < radius < bin_edges[i_max + 1]

    assert len(experiments) == iterations


def test_gradient_adaptive_sampler_3d_example_with_validity_logic():
    """Test adaptive sampler with three parameters.

    Adaptive sampler uses triangulation for >= 3 parameters.
    """
    radius = 0.5
    model_cls: type = MockParametrisedModelWithGradient2d
    params: list[SampleableParameter] = [
        SampleableContinuousModelParameter("x", min_val=-1, max_val=1),
        SampleableContinuousModelParameter("y", min_val=-1, max_val=1),
        SampleableContinuousModelParameter("z", min_val=1, max_val=1.1),
    ]
    gradient_adaptive_sampler: Sampler = GradientAdaptiveSampler(
        random=random.Random(1), default_result=0.0
    )
    iterations = 300
    n_valid, experiments, n_invalid, invalid_experiments = run_resampling(
        model_cls=model_cls,
        variable_params=params,
        sampler=gradient_adaptive_sampler,
        response=lambda result: result["result"],
        iterations=iterations,
        validity_check=lambda params: params["x"] > 0,
        model_reporters={"result": lambda m: m.result},
        fixed_parameters={"radius": radius},
    )
    freq, bin_edges = np.histogram(
        [
            np.sqrt(float(experiment[0]["x"]) ** 2 + float(experiment[0]["y"]) ** 2)
            for experiment in experiments
        ],
        bins=[0, 0.35, 0.7, 1.05, 1.42],
    )
    # Find the bin with the largest frequency and check that it contains the radius
    i_max = np.argmax(freq)
    assert bin_edges[i_max] < radius < bin_edges[i_max + 1]

    assert len(n_valid) == iterations
    assert len(n_invalid) == iterations
    assert len(experiments) + len(invalid_experiments) == iterations
    assert sum(n_valid) == iterations - sum(n_invalid)

    for param_values, _ in experiments:
        assert param_values["x"] > 0

    for param_values, _ in invalid_experiments:
        assert param_values["x"] <= 0


class TestRetModel(RetModel):
    """Test model to run with n_alive model reporter."""

    def __init__(self, n_friendly_agents: int, n_hostile_agents: int):
        """Initialise model.

        Args:
            n_friendly_agents (int): The number of friendly agents to create.
            n_hostile_agents (int): The number of hostile agents to create.
        """
        super().__init__(
            start_time=datetime(2020, 1, 1, 0, 0),
            time_step=timedelta(hours=1),
            end_time=datetime(2020, 1, 2, 0, 0),
            space=self.setup_space(),
            schedule=RandomActivation(self),
            log_config="all",
            playback_writer=JsonWriter(),
        )

        for i in range(int(n_friendly_agents)):
            RetAgent(
                name=f"Friendly Agent {i}",
                model=self,
                pos=(50.0, 50.0),
                affiliation=Affiliation.FRIENDLY,
                critical_dimension=1.0,
                reflectivity=1.0,
                temperature=1.0,
                temperature_std_dev=1.0,
                agent_type=AgentType.GENERIC,
            )
        for i in range(int(n_hostile_agents)):
            RetAgent(
                name=f"Hostile Agent {i}",
                model=self,
                pos=(25.0, 25.0),
                affiliation=Affiliation.HOSTILE,
                critical_dimension=1.0,
                reflectivity=1.0,
                temperature=1.0,
                temperature_std_dev=1.0,
                agent_type=AgentType.GENERIC,
            )

    def setup_space(self):
        """Setup the space and terrain for the model."""
        return ContinuousSpaceWithTerrainAndCulture3d(
            x_max=100,
            y_max=100,
            terrain_image_path=os.path.abspath(
                os.path.join(os.path.dirname(__file__), "test_space/TestTerrain.png")
            ),
            height_black=0.0,
            height_white=100.0,
        )


def get_number_alive_agents(model: RetModel) -> int:
    """Get number of alive agents from a model."""
    agents = model.get_all_agents()
    return len([agent for agent in agents if agent.casualty_state == AgentCasualtyState.ALIVE])


def test_gradient_adaptive_sampler_2d_alive_agents():
    """Test adaptive sampler with two parameters where the result is the number of alive agents."""
    model_cls: type = TestRetModel
    params: list[SampleableParameter] = [
        SampleableContinuousModelParameter("n_friendly_agents", min_val=1, max_val=5),
        SampleableContinuousModelParameter("n_hostile_agents", min_val=1, max_val=3),
    ]
    gradient_adaptive_sampler: Sampler = GradientAdaptiveSampler(
        random=random.Random(1), default_result=0.0
    )
    iterations = 10
    n, experiments, _, _ = run_resampling(
        model_cls=model_cls,
        variable_params=params,
        sampler=gradient_adaptive_sampler,
        response=lambda result: result["n_alive"],
        iterations=iterations,
        model_reporters={"n_alive": lambda m: get_number_alive_agents(m)},
    )
    assert len(experiments) == iterations
    assert n == [1] * iterations
    for inputs, response in experiments:
        # all agents are alive so response is total number of agents
        assert int(inputs["n_friendly_agents"]) + int(inputs["n_hostile_agents"]) == response
