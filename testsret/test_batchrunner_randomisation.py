"""Tests to ensure batchrunning parameters is random yet repeatable."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from mesa.datacollection import DataCollector
from ret.batchrunner import ParamProductBatchRunner
from ret.testing.mocks import MockAgent, MockModel2d
from parameterized import parameterized
import unittest

if TYPE_CHECKING:
    from pandas import DataFrame


class RandomisationMockAgent(MockAgent):
    """Mock agent that iterates internal value using models random number generator."""

    def __init__(self, unique_id, model, val):
        """Instantiate agent."""
        super().__init__(unique_id=unique_id, pos=(0, 0))
        self.model = model
        self.val = val

    def step(self):
        """Step agent."""
        self.val += self.model.random.random()


class RandomisationMockModel(MockModel2d):
    """Mock model for the purpose of testing batchrunner randomisation."""

    def __init__(self, **kwargs):
        """Instantiate model."""
        self.start_time = datetime(2020, 1, 1, 0, 0)
        self.time_step = timedelta(hours=1)
        self.end_time = datetime(2020, 1, 2, 0, 0)
        self.number_of_agents = 3

        super().__init__(self.start_time, self.time_step, self.end_time, kwargs=kwargs)

        self.datacollector = DataCollector(
            tables={
                "Agent Values": ["Time Step", "Agent Internal Value"],
            },
        )

        for i in range(self.number_of_agents):
            self.schedule.add(RandomisationMockAgent(unique_id=i, model=self, val=1))

    def step(self):
        """Write table values and step model."""
        self.write_table_values()
        super().step()

    def write_table_values(self):
        """Write the internal values of the agents to a table."""
        agent_values = [agent.val for agent in self.schedule.agents]
        row = {"Time Step": self.schedule.time, "Agent Internal Value": agent_values}
        self.datacollector.add_table_row("Agent Values", row)


def test_same_seed_batchrunner_randomisation():
    """Test to check batchrunner and model randomisation is repeatable with same seed.

    Multiple models across multiple batchrunners across are instantiated using the same
    seed and checked to ensure output model results are all identical (except for model
    run number).
    """
    same_batchrunners: list[ParamProductBatchRunner] = []

    for _ in range(5):
        batchrunner = ParamProductBatchRunner(
            model_cls=RandomisationMockModel,
            output_path="./",
            max_steps=10,
            iterations=3,
            collect_datacollector=True,
            model_seed=1,
        )
        batchrunner.run_all()

        same_batchrunners.append(batchrunner)

    list_of_table_dicts: list[dict[int, dict[str, DataFrame]]] = [
        batchrunner.output_datacollector_tables() for batchrunner in same_batchrunners
    ]

    dataframe_list: list[DataFrame] = []

    for table_dict in list_of_table_dicts:
        dict_list: list[dict[str, DataFrame]] = list(table_dict.values())

        for dict in dict_list:
            dataframe_list.extend(list(dict.values()))

    dataframe_dict_list = [
        dataframe.drop(columns="Run_number").to_dict() for dataframe in dataframe_list
    ]

    assert all(item == dataframe_dict_list[0] for item in dataframe_dict_list)


def test_different_seed_batchrunner_randomisation():
    """Test to check batchrunner outputs are random using different seeds.

    Multiple models across multiple batchrunners across are instantiated using different
    seeds between batchrunners. Results are checked to ensure that each batchrunner
    generates unique outputs.
    """
    different_batchrunners: list[ParamProductBatchRunner] = []

    number_of_batchrunners = 5

    for i in range(number_of_batchrunners):
        batchrunner = ParamProductBatchRunner(
            model_cls=RandomisationMockModel,
            max_steps=10,
            iterations=3,
            collect_datacollector=True,
            output_path="./",
            model_seed=i,
        )
        batchrunner.run_all()

        different_batchrunners.append(batchrunner)

    list_of_table_dicts: list[dict[int, dict[str, DataFrame]]] = [
        batchrunner.output_datacollector_tables() for batchrunner in different_batchrunners
    ]

    dataframe_list: list[DataFrame] = []

    for table_dict in list_of_table_dicts:
        dict_list: list[dict[str, DataFrame]] = list(table_dict.values())

        for dict in dict_list:
            dataframe_list.extend(list(dict.values()))

    dataframe_dict_set = []

    for dataframe in dataframe_list:
        dataframe_dict = dataframe.drop(columns="Run_number").to_dict()

        if dataframe_dict not in dataframe_dict_set:
            dataframe_dict_set.append(dataframe_dict)

    assert len(dataframe_dict_set) == number_of_batchrunners


class TestSeeds(unittest.TestCase):
    """Class for testing model_seed inputs."""

    @parameterized.expand([[{}], ["seed"], [-5.0], [7.652], [(7, 8)], [[23.234]]])
    def test_incorrect_batchrunner_seed(self, seed):
        """Test batchrunner catches incorrect seeds."""
        with self.assertRaises(TypeError) as e:
            batchrunner = ParamProductBatchRunner(
                model_cls=RandomisationMockModel,
                max_steps=10,
                iterations=3,
                collect_datacollector=True,
                output_path="./",
                model_seed=seed,
            )
            batchrunner.run_all()

            self.assertEqual("Model seed must be of type int, list[int] or None", str(e.exception))

    @parameterized.expand([[[1, 2]], [[1, 2, 3, 4]]])
    def test_incorrect_batchrunner_length(self, seed):
        """Test batchrunner catches incorrect length seeds."""
        with self.assertRaises(ValueError) as e:
            batchrunner = ParamProductBatchRunner(
                model_cls=RandomisationMockModel,
                max_steps=10,
                iterations=3,
                collect_datacollector=True,
                output_path="./",
                model_seed=seed,
            )
            batchrunner.run_all()

            self.assertEqual(
                "If list, length of model seed must be equal to number of iterations.",
                str(e.exception),
            )

    @parameterized.expand([[None], [[1, 2, 3]]])
    def test_seeds_logged(self, seed):
        """Test batchrunner catches incorrect length seeds."""
        batchrunner = ParamProductBatchRunner(
            model_cls=RandomisationMockModel,
            max_steps=10,
            iterations=3,
            collect_datacollector=True,
            output_path="./",
            model_seed=seed,
        )
        batchrunner.run_all()

        assert batchrunner.model_seed is not None
