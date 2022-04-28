"""Batch runner tests."""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from functools import reduce
from operator import concat
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import pathos.multiprocessing as mp
from mesa.batchrunner import ParameterProduct
from mesa.datacollection import DataCollector
from mesa.time import BaseScheduler
from mesa_ret.batchrunner import (
    FixedReportingBatchRunner,
    LatinHypercubeBatchRunner,
    ParamProductBatchRunner,
)
from mesa_ret.sampling import LatinHypercubeParameterSampler
from mesa_ret.testing.mocks import MockAgent, MockModel2d
from numpy import random
from parameterized import parameterized
from tests.test_batchrunner import TestBatchRunner, TestParameters

if TYPE_CHECKING:
    from typing import Any, Callable, Optional


NUM_AGENTS = 7


class BatchRunnerMockAgent(MockAgent):
    """Minimalistic agent implementation for testing purposes."""

    def __init__(self, unique_id, model, val):
        """Create a mock agent.

        Args:
            unique_id (int): A unique identifier for the agent.
            model (Model): The model within which the agent will exist.
            val (int): A value held by the agent.
        """
        super().__init__(unique_id, (0, 0))
        self.model = model
        self.val = val
        self.local = 0

    def step(self):
        """Do one step of the agent's actions."""
        self.val += 1
        self.local += 0.25


class MockModelWithTables(MockModel2d):
    """Minimalistic model with tables for testing purposes."""

    def __init__(
        self,
        variable_model_param: Optional[dict[str, Any]] = None,
        variable_agent_param: Optional[dict[str, Any]] = None,
        start_time: Optional[datetime] = None,
        time_step: Optional[timedelta] = None,
        end_time: Optional[datetime] = None,
        model_reporters: Optional[dict[str, Callable]] = None,
        agent_reporters: Optional[dict[str, str]] = None,
        fixed_model_param: Optional[list[str]] = None,
        schedule: Optional[BaseScheduler] = None,
        **kwargs,
    ):
        """Create mock model that includes specifically written data tables.

        Args:
            variable_model_param (Optional[dict[str, Any]]): The model parameters which
                are varied.
            variable_agent_param (Optional[dict[str, Any]]): The agent parameters which
                are varied.
            start_time (Optional[datetime]): Date and time for the model to begin. If None or unset,
                defaults to datetime(2020, 1, 1, 0, 0).
            time_step (Optional[timedelta]): The time moved forwards each time step. If None or
                unset, defaults to timedelta(hours=1).
            end_time (Optional[datetime]): Date and time for the model to end. if None or unset,
                defaults to datetime(2020, 1, 2, 0, 0).
            model_reporters (Optional[dict[str, Callable]]): Any model level reporters
                which have been defined. Defaults to None.
            agent_reporters (Optional[dict[str, str]]): Any agent level reporters which
                have been defined. Defaults to None.
            fixed_model_param (Optional[list[str]]): Any model parameters which are held
                constant between iterations. Defaults to None.
            schedule (Optional[BaseScheduler]): The scheduler which organises the order
                in which the agents activate. Defaults to None.
            **kwargs (dict[str, Any]): Keyword arguments -If this contains a "n_agents"
                keyword, this will be used for the number of agents.
        """
        if start_time is None:
            start_time = datetime(year=2020, month=1, day=1)

        if end_time is None:
            end_time = datetime(year=2020, month=1, day=2)

        if time_step is None:
            time_step = timedelta(hours=1)

        super().__init__(start_time, time_step, end_time)
        if model_reporters is None:
            model_reporters = {"reported_model_param": self.get_local_model_param}
        if agent_reporters is None:
            agent_reporters = {"agent_id": "unique_id", "agent_local": "local"}
        self.datacollector = DataCollector(
            model_reporters=model_reporters,
            agent_reporters=agent_reporters,
            tables={
                "Agent_Numbers": ["time_step", "agent_number"],
                "Agent_Numbers2": ["time_step", "agent_number"],
            },
        )
        self.schedule = BaseScheduler(None) if schedule is None else schedule
        self.variable_model_param = variable_model_param
        self.variable_agent_param = variable_agent_param
        self.fixed_model_param = fixed_model_param
        self.n_agents = kwargs.get("n_agents", NUM_AGENTS)
        self.running = True
        self.init_agents()

    def init_agents(self):
        """Initialise the agents by adding them to the scheduler."""
        if self.variable_agent_param is None:
            agent_val = 1
        else:
            agent_val = self.variable_agent_param
        for i in range(self.n_agents):
            self.schedule.add(BatchRunnerMockAgent(i, self, agent_val))

    def get_local_model_param(self) -> int:
        """Return local model parameter.

        Returns:
            int: Local model parameter (always 42)
        """
        return 42

    def step(self):
        """Advance the model by one step and collect data."""
        self.write_table_values()
        super().step()

    def write_table_values(self):
        """Write the numbers of agents to the appropriate table."""
        agent_number = self.n_agents
        row = {"time_step": self.schedule.time, "agent_number": agent_number}
        self.datacollector.add_table_row("Agent_Numbers", row)


class TestFixedReportingBatchRunner(TestBatchRunner):
    """Test Fixed BatchRunner with functionality to expose datacollector tables.

    Test that Fixed Reporting BatchRunner is running batches with parameters
    defined by LatinHypercubeParameterSampler.
    """

    def setUp(self):
        """Set up test case."""
        super().setUp()

        self.mock_model = MockModelWithTables

    @parameterized.expand([[True], [False]])
    def test_batchrunner_instantiation(self, ignore_multiprocessing: bool):
        """Test FixedReportingBatchRunner can be instantiatated."""
        batch = FixedReportingBatchRunner(
            MockModel2d, ignore_multiprocessing=ignore_multiprocessing
        )
        assert isinstance(batch, FixedReportingBatchRunner)

    @parameterized.expand([[True], [False]])
    def test_no_param_model_run_all(self, ignore_multiprocessing: bool):
        """Test the batchrunner will run with no parameters."""
        batch = FixedReportingBatchRunner(
            MockModel2d, ignore_multiprocessing=ignore_multiprocessing
        )

        batch.run_all()
        assert batch.fixed_parameters == {}
        assert batch.parameters_list == []
        assert batch.run_info == [(0,)]
        assert batch.collect_datacollector is False

    @parameterized.expand([[True], [False]])
    def test_no_param_model_single(self, ignore_multiprocessing: bool):
        """Test the batchrunner will run with no parameters."""
        batch = FixedReportingBatchRunner(
            MockModel2d, ignore_multiprocessing=ignore_multiprocessing
        )

        batch.run_iteration({}, None, 0)
        assert batch.fixed_parameters == {}
        assert batch.parameters_list == []
        assert batch.run_info == []
        assert batch.collect_datacollector is False

    def launch_batch_processing_with_datacollector(
        self, ignore_multiprocessing: bool = False
    ) -> ParamProductBatchRunner:
        """Create a new batch runner and execute run_all method.

        Returns:
            ParamProductBatchRunner: Batch runner
        """
        batch = ParamProductBatchRunner(
            self.mock_model,
            variable_parameters=self.variable_params,
            fixed_parameters=self.fixed_params,
            iterations=self.iterations,
            max_steps=self.max_steps,
            model_reporters=self.model_reporters,
            agent_reporters=self.agent_reporters,
            collect_datacollector=True,
            ignore_multiprocessing=ignore_multiprocessing,
        )
        batch.run_all()
        return batch

    @parameterized.expand([[2], [4], [None]])
    def test_valid_multiprocessing_batchrunner_instantiation(self, core_number):
        """Test no errors are raised when a valid multiprocessing batchrunner is instantiated."""
        ParamProductBatchRunner(
            self.mock_model,
            variable_parameters=self.variable_params,
            fixed_parameters=self.fixed_params,
            iterations=self.iterations,
            max_steps=self.max_steps,
            model_reporters=self.model_reporters,
            agent_reporters=self.agent_reporters,
            collect_datacollector=True,
            multiprocessing_cores=core_number,
        )

    @parameterized.expand([[0], [-2]])
    def test_0_and_negative_cores_multiprocessing_batchrunner_instantiation(self, core_number):
        """Test errors are raised when an invalid multiprocessing batchrunner is instantiated."""
        with self.assertRaises(ValueError) as e:
            ParamProductBatchRunner(
                self.mock_model,
                variable_parameters=self.variable_params,
                fixed_parameters=self.fixed_params,
                iterations=self.iterations,
                max_steps=self.max_steps,
                model_reporters=self.model_reporters,
                agent_reporters=self.agent_reporters,
                collect_datacollector=True,
                multiprocessing_cores=core_number,
            )
        self.assertEqual(
            f"Multiprocessing cores must be greater than 0 ({core_number} specified).",
            str(e.exception),
        )

    def test_many_cores_multiprocessing_batchrunner_instantiation(self):
        """Test warning is raised with core count > cpu count."""
        with warnings.catch_warnings(record=True) as w:
            ParamProductBatchRunner(
                self.mock_model,
                variable_parameters=self.variable_params,
                fixed_parameters=self.fixed_params,
                iterations=self.iterations,
                max_steps=self.max_steps,
                model_reporters=self.model_reporters,
                agent_reporters=self.agent_reporters,
                collect_datacollector=True,
                multiprocessing_cores=20,
            )
            assert (
                "Number of multiprocessing cores specified (20) is "
                + f"greater than detected system cpu count{mp.cpu_count()}, this will"
                + "likely result in high memory usage and degraded performance."
                == str(w[0].message)
            )

    @parameterized.expand([[True], [False]])
    def test_output_all_tables(self, ignore_multiprocessing: bool):
        """Test production of all data tables.

        Tests that all lines in a set of produced data are allocated the run number.
        Tests that all tables stored within the data collector are converted to
        dataframes.
        """
        batch = self.launch_batch_processing_with_datacollector(
            ignore_multiprocessing=ignore_multiprocessing
        )
        max_value = len(batch.data_vars) - 1
        random_data_collector = random.randint(0, max_value)

        data_collector = list(batch.data_vars.values())[random_data_collector]["Data Collector"]

        random_run_number = random.randint(0, 1000)

        output_tables = batch.output_all_tables(data_collector, random_run_number)

        for key, data in output_tables.items():
            assert key in data_collector.tables

            if key == "Agent_Numbers2":
                assert not data.any().values.any()
            else:
                run_numbers = data["Run_number"].unique()
                assert len(run_numbers) == 1
                run_number = run_numbers[0]
                assert random_run_number == run_number

    @parameterized.expand([[True], [False]])
    def test_output_one_table(self, ignore_multiprocessing: bool):
        """Test production of single table."""
        batch = self.launch_batch_processing_with_datacollector(
            ignore_multiprocessing=ignore_multiprocessing
        )
        random_data_collector = random.randint(0, len(batch.data_vars) - 1)

        data_collector = list(batch.data_vars.values())[random_data_collector]["Data Collector"]

        run_number = random.randint(1, 1000)
        data = batch.output_one_table(data_collector, run_number, "Agent_Numbers")
        assert data is not None
        run_numbers = data["Run_number"].unique()
        assert len(run_numbers) == 1
        assert run_number == run_numbers[0]

    @parameterized.expand([[True], [False]])
    def test_output_nonexisting_table(self, ignore_multiprocessing: bool):
        """Test handling of request to produce non-existent table."""
        batch = self.launch_batch_processing_with_datacollector(
            ignore_multiprocessing=ignore_multiprocessing
        )
        rdc = random.randint(0, len(batch.data_vars) - 1)

        data_collector = list(batch.data_vars.values())[rdc]["Data Collector"]
        with self.assertWarns(Warning) as w:
            data = batch.output_one_table(data_collector, 1, "Non-existent-table")
            assert data is None

        assert "Specified table 'Non-existent-table' not found in datacollector" in str(
            w.warnings[0].message
        )

    @parameterized.expand([[True], [False]])
    def test_get_aggregated_dataframe(self, ignore_multiprocessing: bool):
        """Test getting aggregated dataframe for table."""
        batch = self.launch_batch_processing_with_datacollector(
            ignore_multiprocessing=ignore_multiprocessing
        )
        df = batch.get_aggregated_dataframe("Agent_Numbers")
        assert df is not None
        assert len(df.index) >= len(batch.model_vars)
        # can't be sure that all runs ran for the full duration (batch.max_steps)
        assert len(df.index) <= len(batch.model_vars) * batch.max_steps

    @parameterized.expand([[True], [False]])
    def test_save_tables_to_csv(self, ignore_multiprocessing: bool):
        """Test saving multiple tables to csv."""
        with TemporaryDirectory() as td:
            batch = self.launch_batch_processing_with_datacollector(
                ignore_multiprocessing=ignore_multiprocessing
            )
            success = batch.save_tables_to_csv(["Agent_Numbers", "Agent_Numbers2"], td)
            assert success
            assert Path(td, "Agent_Numbers.csv").is_file()
            assert Path(td, "Agent_Numbers2.csv").is_file()

    @parameterized.expand([[True], [False]])
    def test_save_table_to_csv(self, ignore_multiprocessing: bool):
        """Test saving single table to CSV files."""
        with TemporaryDirectory() as td:
            batch = self.launch_batch_processing_with_datacollector(
                ignore_multiprocessing=ignore_multiprocessing
            )
            success = batch.save_table_to_csv("Agent_Numbers", td)
            assert success
            assert Path(td, "Agent_Numbers.csv").is_file()

    @parameterized.expand([[True], [False]])
    def test_save_non_existent_table(self, ignore_multiprocessing: bool):
        """Test saving a table that doesn't exist."""
        with TemporaryDirectory() as td:
            batch = self.launch_batch_processing_with_datacollector(
                ignore_multiprocessing=ignore_multiprocessing
            )
            with self.assertWarns(Warning) as ws:
                success = batch.save_table_to_csv("na", td)
                assert not success
                assert not Path(td, "na.csv").is_file()

            assert "No log information for na." in str(ws.warnings[-1])
            for w in ws.warnings[0:-1]:
                self.assertTrue("Specified table na not found in datacollector", str(w))

    @parameterized.expand([[True], [False]])
    def test_save_to_non_existent_location(self, ignore_multiprocessing: bool):
        """Test saving to an invalid location."""
        batch = self.launch_batch_processing_with_datacollector(
            ignore_multiprocessing=ignore_multiprocessing
        )
        with TemporaryDirectory() as td:
            p = Path(td, "not-a-directory")
            with self.assertWarns(Warning) as w:
                batch.save_table_to_csv("Agent_Numbers", p)

            assert "An error occurred writing Agent_Numbers to file" in str(w.warnings[0].message)

    @parameterized.expand([[True], [False]])
    def test_table_not_found(self, ignore_multiprocessing: bool):
        """Test whether choosing a missing table gives a warning."""
        with warnings.catch_warnings(record=True) as w:
            batch = self.launch_batch_processing_with_datacollector(
                ignore_multiprocessing=ignore_multiprocessing
            )
            output = batch.output_datacollector_tables("test")
            assert output[0] is None
            assert len(w) == 2
            assert "not found in datacollector" in str(w[0].message)
            assert "Available tables are" in str(w[-1].message)

    @parameterized.expand([[True], [False]])
    def test_individual_table_returned(self, ignore_multiprocessing: bool):
        """Test whether the a specific table is successfully collected."""
        batch = self.launch_batch_processing_with_datacollector(
            ignore_multiprocessing=ignore_multiprocessing
        )
        variable_params_len = sum(1 for _ in ParameterProduct(self.variable_params)._product)
        assert len(batch.output_datacollector_tables("Agent_Numbers")) == self.model_runs
        assert (
            len(batch.output_datacollector_tables("Agent_Numbers"))
            == self.iterations * variable_params_len
        )
        assert batch.output_datacollector_tables("Agent_Numbers")

    @parameterized.expand([[True], [False]])
    def test_runs_summary_correct_size(self, ignore_multiprocessing: bool):
        """Test whether the run parameter summary is generated correctly."""
        batch = self.launch_batch_processing_with_datacollector(
            ignore_multiprocessing=ignore_multiprocessing
        )
        run_params = batch.get_run_params_dataframe()
        expected_cols = len(self.variable_params) + 1  # extra column with run index
        assert run_params.shape == (self.model_runs, expected_cols)

    @parameterized.expand([[True], [False]])
    def test_all_tables_returned(self, ignore_multiprocessing: bool):
        """Test whether all the datacollector tables are successfully collected."""
        batch = self.launch_batch_processing_with_datacollector(
            ignore_multiprocessing=ignore_multiprocessing
        )
        tables = batch.output_datacollector_tables()
        variable_params_len = sum(1 for _ in ParameterProduct(self.variable_params)._product)
        assert len(tables) == self.model_runs
        assert len(tables) == self.iterations * variable_params_len
        assert len(tables[random.choice(list(tables))]) == 2  # type: ignore

    @parameterized.expand([[True], [False]])
    def test_datacollector_not_collected(self, ignore_multiprocessing: bool):
        """Test for an error if tables are requested without the datacollecter."""
        batch = ParamProductBatchRunner(
            self.mock_model,
            variable_parameters=self.variable_params,
            fixed_parameters=self.fixed_params,
            iterations=self.iterations,
            max_steps=self.max_steps,
            model_reporters=self.model_reporters,
            agent_reporters=self.agent_reporters,
            collect_datacollector=False,
            ignore_multiprocessing=ignore_multiprocessing,
        )
        with warnings.catch_warnings(record=True) as w:
            batch.run_all()
            assert batch.output_datacollector_tables() == {}
            assert len(w) == 1
            assert "Data collector not collected so no tables available" in str(w[0].message)


class TestLatinHypercubeBatchRunner(TestBatchRunner):
    """Test Latin Hypercube BatchRunner.

    Test that Latin Hypercube BatchRunner is running batches with parameters
    defined by LatinHypercubeParameterSampler.
    """

    def setUp(self):
        """Set up test case."""
        super().setUp()

        self.n_experiments = 4
        self.random_state = 1

    def launch_batch_processing(
        self, ignore_multiprocessing: bool = False
    ) -> LatinHypercubeBatchRunner:
        """Create a new batch runner and launch it.

        Returns:
            LatinHypercubeBatchRunner: Batch runner
        """
        batch = LatinHypercubeBatchRunner(
            self.mock_model,
            variable_parameters=self.variable_params,
            fixed_parameters=self.fixed_params,
            iterations=self.iterations,
            max_steps=self.max_steps,
            model_reporters=self.model_reporters,
            agent_reporters=self.agent_reporters,
            n_experiments=self.n_experiments,
            random_state=self.random_state,
            collect_datacollector=False,
            ignore_multiprocessing=ignore_multiprocessing,
        )
        batch.run_all()
        return batch

    @parameterized.expand([[True], [False]])
    def test_agent_level_vars(self, ignore_multiprocessing: bool):
        """Test that agent-level variable collection is of the correct size."""
        batch = self.launch_batch_processing(ignore_multiprocessing=ignore_multiprocessing)
        agent_vars = batch.get_agent_vars_dataframe()
        agent_collector = batch.get_collector_agents()
        # extra columns with run index and agentId
        expected_cols = len(self.variable_params) + len(self.agent_reporters) + 2
        assert agent_vars.shape == (self.model_runs * NUM_AGENTS, expected_cols)
        assert "agent_val" in list(agent_vars.columns)
        assert "val_non_existent" not in list(agent_vars.columns)
        assert len(agent_collector.keys()) == 68
        for key in agent_collector.keys():
            assert "agent_id" in list(agent_collector[key].columns)
            assert "Step" in list(agent_collector[key].index.names)
            assert "nose" not in list(agent_collector[key].columns)
        for var, values in self.variable_params.items():
            assert set(agent_vars[var].unique()) == set(values)  # type: ignore

        temp_key = next(key for key in agent_collector.keys() if key[-1] == 0)
        assert agent_collector[temp_key].shape == (NUM_AGENTS * self.max_steps, 2)

        with self.assertRaises(KeyError):
            agent_collector[(900, "k", 3)]

    @parameterized.expand([[True], [False]])
    def test_model_with_dict_variable_params(self, ignore_multiprocessing: bool):
        """Test model with dictionary of variable parameters."""
        self.variable_params = {
            "variable_model_param": [
                {"mvar1": 1, "mvar2": 1},
                {"mvar1": 2, "mvar2": 2},
            ],
            "variable_agent_param": [1, 8],
        }
        batch = self.launch_batch_processing(ignore_multiprocessing=ignore_multiprocessing)
        model_vars = batch.get_model_vars_dataframe()
        print(model_vars["reported_variable_value"])
        expected_cols = (
            len(self.variable_params) + len(self.model_reporters) + 1
        )  # extra column with run index

        assert model_vars.shape == (self.model_runs, expected_cols)
        possible_reported_variable_values: list[tuple] = [
            (1, 1),
            (1, 2),
            (2, 1),
            (2, 2),
        ]
        first_reported_variable = (
            model_vars["reported_variable_value"][0]["mvar1"],
            model_vars["reported_variable_value"][0]["mvar2"],
        )
        last_reported_variable = (
            model_vars["reported_variable_value"][self.model_runs - 1]["mvar1"],
            model_vars["reported_variable_value"][self.model_runs - 1]["mvar2"],
        )
        assert first_reported_variable in possible_reported_variable_values
        assert last_reported_variable in possible_reported_variable_values

    @property
    def model_runs(self) -> int:
        """Return total number of batch runner's iterations.

        Returns:
            int: Number of iterations
        """
        n_iterations: int = self.n_experiments * self.iterations
        return n_iterations


class TestLatinHypercubeParameterSampler(TestParameters):
    """Test cases for Latin Hypercube Parameter Sampler."""

    def test_latin_hypercube(self):
        """Test Latin Hypercube Parameter Sampler.

        Checks that every parameter value passed into the LatinHypercubeParameterSampler
        appears exactly once in the output.
        """
        params1 = LatinHypercubeParameterSampler(
            {
                "var_alpha": ["a", "b", "c", "d", "e"],
                "var_num": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            },
            n_experiments=10,
            random_state=1,
        )
        params2 = LatinHypercubeParameterSampler(
            {"var_alpha": ["a", "b", "c", "d", "e"], "var_num": range(16)},
            n_experiments=10,
            random_state=1,
        )

        lp = list(params1)
        assert 10 == len(lp)
        assert lp == list(params2)

        params3 = {
            "var_1": ["a", "b", "c"],
            "var_2": ["d", "e", "f"],
            "var_3": ["g", "h", "i"],
        }

        sample = LatinHypercubeParameterSampler(params3, n_experiments=3)

        lhs_sampled_values = []
        for sample_dict in list(sample):
            lhs_sampled_values.append(list(sample_dict.values()))

        lhs_sampled_values_flattened = [item for sublist in lhs_sampled_values for item in sublist]
        params_flattened = reduce(concat, params3.values(), [])

        assert len(lhs_sampled_values_flattened) == len(params_flattened)
