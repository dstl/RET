"""RET Batch Runners."""

from __future__ import annotations

import warnings
from itertools import count
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar, Union, Optional
from shutil import copy
import inspect

import pandas as pd
import pathos.multiprocessing as mp
from tqdm import tqdm
import os

from mesa.batchrunner import FixedBatchRunner, ParameterProduct
from mesa.datacollection import DataCollector
from ret.sampling import LatinHypercubeParameterSampler
from ret.utilities.save_utilities import validate_save_path, add_datetime_stamp, add_iteration_stamp

if TYPE_CHECKING:
    from typing import Any, Callable

    from ret.model import RetModel

T = TypeVar("T")
SingleOrListOptional = Union[Optional[T], list[Optional[T]]]


class FixedReportingBatchRunner(FixedBatchRunner):
    """Batchrunner with methods to expose the datacollector's tables.

    This class is instantiated with a model class, and model parameters
    associated with one or more values. It is also instantiated with model and
    agent-level reporters, dictionaries mapping a variable name to a function
    which collects some data from the model or its agents at the end of the run
    and stores it. It also has a flag to denote whether to collect the
    datacollector.

    Batchrunner by default uses multiprocessing to parallelise batchrunning of
    simulations, which requires some computational overhead for management. Consider
    specifying 1 multiprocessing core if you are instantiating a batchrunner with a
    small number of runs and/or a particularly simple model.

    When running using the multiprocessing on a Windows machine it is necessary to include
    the following line before the batchrunner creation:
    if __name__ == "__main__":
    This is explained in length in the official Python documentation:
    https://docs.python.org/3.9/library/multiprocessing.html#multiprocessing-programming
    and essentially derives from the need to safely import the main module without
    spawning a new process.

    Note that by default, the reporters only collect data at the *end* of the
    run. To get step by step data, simply have a reporter store the model's
    entire DataCollector object.
    """

    def __init__(  # noqa: C901
        self,
        model_cls: type,
        output_path: str,
        parameters_list: Union[Optional[ParameterProduct], list[dict[str, Any]]] = None,
        fixed_parameters: Optional[dict[str, Any]] = None,
        iterations: int = 1,
        max_steps: int = 1000,
        save_every: Optional[int] = None,
        model_reporters: Optional[dict[str, Callable]] = None,
        agent_reporters: Optional[dict[str, str]] = None,
        collect_datacollector: bool = False,
        multiprocessing_cores: Optional[int] = None,
        ignore_multiprocessing: bool = False,
        tables_to_save: Optional[list] = None,
        model_seed: SingleOrListOptional[int] = None,
    ):
        """Create a new BatchRunner for a given model with the given parameters.

        Args:
            model_cls (type): The class of model to batch-run.
            output_path (str): The output location to save data to.
            parameters_list (Union[Optional[ParameterProduct], list[dict[str, Any]]]): A
                ParamProduct or list of dictionaries of parameter sets. The model will be
                run with dictionaries. For example, given parameters_list of
                    [{"homophily": 3, "density": 0.8, "minority_pc": 0.2},
                    {"homophily": 2, "density": 0.9, "minority_pc": 0.1},
                    {"homophily": 4, "density": 0.6, "minority_pc": 0.5}]
                3 models will be run, one for each provided set of parameters.
            fixed_parameters (Optional[dict[str, Any]]): Dictionary of parameters that
                stay same through all batch runs. For example, given fixed_parameters of
                    {"constant_parameter": 3},
                every instantiated model will be passed constant_parameter=3 as a kwarg.
            iterations (int): The total number of times to run the model for each
                combination of parameters. Defaults to 1.
            max_steps (int): Upper limit of steps above which each run will be halted
                if it hasn't halted on its own. Defaults to 1000.
            save_every (Optional[int]): The number of times to run the model before saving
                DataCollector tables. If None, no outputs will be automatically saved.
                Defaults to None.
            model_reporters (Optional[dict[str, Callable]]): The dictionary of variables
                to collect on each run at the end, with variable names mapped to a
                function to collect them. For example:
                    {"agent_count": lambda m: m.schedule.get_agent_count()}
                Defaults to no model reporters.
            agent_reporters (Optional[dict[str, str]]): Like model_reporters, but each
                variable is now collected at the level of each agent present in the
                model at the end of the run. Defaults to no agent reporters.
            collect_datacollector (bool): Collect the entire DataCollector object at
                the end of each run. Defaults to false.
            multiprocessing_cores (Optional[int]): The number of cores to be used for
                multiprocessing batchrunning. Defaults to None, resulting in the number
                of cores to match the system cpu count.
            ignore_multiprocessing (bool): Whether the use the run_all method from base class
                without any multiprocessing. Defaults to False.
            tables_to_save (Optional[list]): List of DataCollector tables to save.
                If None all default tables will be saved. Defaults to None.
            model_seed (SingleOrListOptional[int]): Seed for model. If int, the same model
                seed will be used to set RNG across all batch runs.
                If list[int], model_seed[run_count] will be used to set RNG.
                If none, random_numbers[run_count] will be used to set RNG.
                Defaults to None.
        """
        super().__init__(
            model_cls=model_cls,
            parameters_list=parameters_list,
            fixed_parameters=fixed_parameters,
            iterations=iterations,
            max_steps=max_steps,
            model_reporters=model_reporters,
            agent_reporters=agent_reporters,
        )

        self.tables_to_save = tables_to_save
        self.ignore_multiprocessing = ignore_multiprocessing
        self.collect_datacollector: bool = collect_datacollector

        self.output_path = add_datetime_stamp(output_path)
        if not os.path.exists(self.output_path):
            print(f"{self.output_path} does not exist. Creating directory.")
            os.makedirs(self.output_path, exist_ok=True)

        self.iterations = iterations
        self.run_info: list[Any] = list()
        if collect_datacollector:
            self.data_vars: dict[tuple[Any, ...], Any] = {}

        if tables_to_save is None:
            self.tables_to_save = [
                "agents",
                "behaviour_log",
                "task_log",
                "trigger_log",
                "deaths",
                "shots_fired",
                "observation_record",
                "perception_record",
                "behaviour_selection",
                "position_and_culture",
                "model_seed",
            ]

        if save_every is not None:
            if save_every > self.iterations:
                warnings.warn(
                    "The number of iterations to run before saving is"
                    + " larger than the total number of iterations to run",
                    stacklevel=2,
                )
            if save_every <= 0:
                raise ValueError(
                    "The number of iterations to run before saving must be greater than 0 "
                    + f"({save_every} specified)."
                )
        self.save_every = save_every

        total_iterations, all_kwargs, all_param_values = self._make_model_args()

        if model_seed is None:
            model_seed = [
                int.from_bytes(os.urandom(16), byteorder="big") for x in range(self.iterations)
            ]

        if isinstance(model_seed, list):
            for entry in model_seed:
                if isinstance(entry, int) is False:
                    raise TypeError("Model seed must be of type int, list[int] or None")

            if len(model_seed) != self.iterations:
                raise ValueError(
                    "If list, length of model seed must be equal to number of iterations."
                )
        else:
            if isinstance(model_seed, int) is False:
                raise TypeError("Model seed must be of type int, list[int] or None")
        if len(all_kwargs) > 0:
            self.model_seed = model_seed * len(all_kwargs)
        else:
            self.model_seed = model_seed
        if not self.ignore_multiprocessing:
            if multiprocessing_cores is not None:
                if multiprocessing_cores <= 0:
                    raise ValueError(
                        "Multiprocessing cores must be greater than 0 "
                        + f"({multiprocessing_cores} specified)."
                    )
                if multiprocessing_cores > mp.cpu_count():
                    warnings.warn(
                        f"Number of multiprocessing cores specified ({multiprocessing_cores}) is "
                        + f"greater than detected system cpu count{mp.cpu_count()}, this will"
                        + "likely result in high memory usage and degraded performance.",
                        stacklevel=2,
                    )
                self._multiprocessing_cores = multiprocessing_cores
            else:
                self._multiprocessing_cores = mp.cpu_count()

    @staticmethod
    def model_key(param_values: Optional[list[Any]], run_count: int) -> tuple[Any, ...]:
        """Generates a unique model key for a given iteration.

        Args:
            param_values (Optional[list[Any]]): the list of parameters to be used in the run.
            run_count (int): The run iteration index.

        Returns:
            A Tuple representing the parameter values and run count, used for keying iterations.
        """
        if param_values is None:
            return (run_count,)

        return tuple(param_values) + (run_count,)

    def _run_all_without_multiprocessing(self):
        """Run the model at all parameter combinations and store results."""
        run_count = count()
        total_iterations, all_kwargs, all_param_values = self._make_model_args()

        with tqdm(total=total_iterations, colour="BLUE") as pbar:
            for i, kwargs in enumerate(all_kwargs):
                param_values = all_param_values[i]
                for _ in range(self.iterations):
                    results = self.run_iteration(kwargs, param_values, next(run_count))
                    if self.model_reporters:
                        self.model_vars.update([results[0].items()][0])
                    if self.agent_reporters:
                        self.agent_vars.update([results[1].items()][0])
                    self.datacollector_model_reporters.update([results[2].items()][0])
                    self.datacollector_agent_reporters.update([results[3].items()][0])
                    if self.collect_datacollector:
                        self.data_vars.update([results[4].items()][0])
                    self.run_info.append(results[5])
                    if self.save_every is not None:
                        self.check_and_save_tables(
                            current_iteration=_,
                            save_every=self.save_every,
                            tables_to_save=self.tables_to_save,
                        )
                    pbar.update(1)

    def check_and_save_tables(self, current_iteration: int, save_every: int, tables_to_save: list):
        """Check if DataCollector tables are to be saved this timestep and save them.

        Args:
            current_iteration (int): The current iteration of the run.
            save_every (int): The number of times to run the model before saving
                DataCollector tables.
            tables_to_save (list): List of DataCollector tables to save.
        """
        # Check if current iteration is multiple of how often to save outputs
        if ((current_iteration + 1) % save_every == 0) | (
            (current_iteration + 1) == self.iterations
        ):
            for table in tables_to_save:
                self.save_table_to_csv(str(table))

    def run_all(self):  # noqa: C901
        """Run the model at all parameter combinations and store results."""
        if self.ignore_multiprocessing:
            self._run_all_without_multiprocessing()
            return

        total_iterations, kwargs_list, param_values_list = self._make_model_args()
        # Multiply model arguments by number of iterations necessary for batch process
        run_count_list = [i for i in range(total_iterations)]
        all_kwargs_list = [entry for entry in kwargs_list for _iteration in range(self.iterations)]
        all_param_values_list = [
            entry for entry in param_values_list for _iteration in range(self.iterations)
        ]
        pool = mp.ProcessingPool(self._multiprocessing_cores)
        # Pass run_iteration and arguments to ProcessingPool
        iteration_results = []
        cwd_list = [os.getcwd() for i in range(total_iterations)]
        for results in tqdm(
            pool.imap(
                self.run_iteration,
                all_kwargs_list,
                all_param_values_list,
                run_count_list,
                cwd_list,
            ),
            total=total_iterations,
            colour="BLUE",
        ):
            iteration_results.append(results)

            # If save_every x runs, check if need to update batchrunner internals.
            if self.save_every is None:
                continue
            else:
                if (len(iteration_results)) % self.save_every != 0:
                    continue
                else:
                    for _, iteration_result in enumerate(iteration_results):
                        if self.model_reporters:
                            self.model_vars.update([iteration_result[0].items()][0])
                        if self.agent_reporters:
                            self.agent_vars.update([iteration_result[1].items()][0])
                        self.datacollector_model_reporters.update([iteration_result[2].items()][0])
                        self.datacollector_agent_reporters.update([iteration_result[3].items()][0])
                        if self.collect_datacollector:
                            self.data_vars.update([iteration_result[4].items()][0])
                        self.run_info.append(iteration_result[5])
                    self.check_and_save_tables(
                        current_iteration=len(iteration_results) - 1,
                        save_every=self.save_every,
                        tables_to_save=self.tables_to_save,
                    )
        # Unpack iteration results from ProcessingPool into batchrunner internals
        for _, iteration_result in enumerate(iteration_results):
            if self.model_reporters:
                self.model_vars.update([iteration_result[0].items()][0])
            if self.agent_reporters:
                self.agent_vars.update([iteration_result[1].items()][0])
            self.datacollector_model_reporters.update([iteration_result[2].items()][0])
            self.datacollector_agent_reporters.update([iteration_result[3].items()][0])
            if self.collect_datacollector:
                self.data_vars.update([iteration_result[4].items()][0])
            self.run_info.append(iteration_result[5])
        if self.save_every is not None:
            self.check_and_save_tables(
                current_iteration=len(iteration_results) - 1,
                save_every=self.save_every,
                tables_to_save=self.tables_to_save,
            )

    def run_iteration(  # noqa: C901
        self,
        kwargs: dict[str, list[Any]],
        param_values: Optional[list[Any]],
        run_count: int,
        cwd: Optional[str] = None,
    ) -> tuple[Any, Any, Any, Any, Any, Any]:
        """Run a single iteration of the model.

        Note: this method should not be called directly, the results of a single iteration
        are not stored as part of the batchrunner unless called as a part of "run_all".

        Args:
            kwargs (list[Any]): Any keyword arguments.
            param_values (Optional[list[Any]]): The list of parameters to be used in the run.
            run_count (int): What number run this is from the batchrunner.
            cwd (Optional[str]): The current working directory of the executing terminal for
                a multiproccess batch. Defaults to None.

        Returns:
            tuple[Any, Any, Any, Any, Any]: The model, agent, datacollector model,
                datacollector agent and data variables. The data_vars only contains the
                datacollector.
        """
        if cwd is not None:
            if os.path.exists(cwd):
                os.chdir(cwd)
        kwargs["output_path"] = self.output_path  # type: ignore
        model: RetModel = self.model_cls(**kwargs)
        if isinstance(self.model_seed, int):
            model.reset_randomizer(seed=self.model_seed)
        else:
            seed: list[int] = self.model_seed  # type: ignore
            model.reset_randomizer(seed=seed[run_count])
        results = self.run_model(model)

        # Collect and store results:
        model_key = FixedReportingBatchRunner.model_key(
            param_values=param_values, run_count=run_count
        )

        iteration_model_vars = {}
        iteration_agent_vars = {}
        iteration_datacollector_model_reporters = {}
        iteration_datacollector_agent_reporters = {}
        iteration_data_vars = {}

        if self.collect_datacollector:
            iteration_data_vars[model_key] = self.collect_data_collector(model)
        if self.model_reporters:
            iteration_model_vars[model_key] = self.collect_model_vars(model)
        if self.agent_reporters:
            agent_vars = self.collect_agent_vars(model)
            for agent_id, reports in agent_vars.items():
                agent_key = model_key + (agent_id,)
                iteration_agent_vars[agent_key] = reports
        # Collects data from datacollector object in model
        if results is not None:
            if results.model_reporters is not None:
                iteration_datacollector_model_reporters[
                    model_key
                ] = results.get_model_vars_dataframe()
            if results.agent_reporters is not None:
                iteration_datacollector_agent_reporters[
                    model_key
                ] = results.get_agent_vars_dataframe()
        return (
            iteration_model_vars,
            iteration_agent_vars,
            iteration_datacollector_model_reporters,
            iteration_datacollector_agent_reporters,
            iteration_data_vars,
            model_key,
        )

    def output_datacollector_tables(
        self, table_name: Optional[str] = None
    ) -> dict[int, Optional[Union[pd.DataFrame, dict[str, pd.DataFrame]]]]:
        """Output a dictionary of all datacollector tables.

        The first key is the iteration number. If no argument is given for
        table name, the value is another dictionary containing all the tables
        ('table name: dataframe' as the key:value pair). If a table name is
        specified then the value is the dataframe containing that table.

        Args:
            table_name (Optional[str]): (Optional) Name for desired table. Defaults to
                no table name.

        Returns:
            dict[int, Optional[Union[pd.DataFrame, dict[str, pd.DataFrame]]]]: The
                desired table(s) in a dictionary against iteration number.
        """
        if self.collect_datacollector is False:
            warnings.warn("Data collector not collected so no tables available.", stacklevel=2)
            return {}
        out_data: dict[int, Optional[Union[pd.DataFrame, dict[str, pd.DataFrame]]]] = {}
        test = self.data_vars.items()
        run_with_datacollector = {key[-1]: value for key, value in test}
        for run_number in run_with_datacollector:
            dc = run_with_datacollector[run_number]["Data Collector"]
            if isinstance(dc, DataCollector):
                if table_name is None:
                    out_data[run_number] = self.output_all_tables(dc, run_number)
                else:
                    single_table: Optional[pd.DataFrame] = self.output_one_table(
                        dc, run_number, table_name
                    )
                    out_data[run_number] = single_table
        return out_data

    def output_all_tables(self, dc: DataCollector, run_number: int) -> dict[Any, Any]:
        """Turn all tables in the datacollector into a dictionary.

        Args:
            dc (DataCollector): The datacollector containing the information for
                interrogation is passed in and this is what contains the tables.
            run_number (int): The run number is passed in so that a column for run
                number may be added to each table.

        Returns:
            current_iteration_tables (dict): A dictionary with key-value pairs of the
                table names to tables.
        """
        current_iteration_tables = {}
        for table in list(dc.tables.keys()):
            iteration_run_data = dc.get_table_dataframe(table)
            iteration_run_data["Run_number"] = run_number
            current_iteration_tables[table] = iteration_run_data
        return current_iteration_tables

    def output_one_table(
        self, dc: DataCollector, run_number: int, table_name: str
    ) -> Union[pd.DataFrame, None]:
        """Extract a single named table from the datacollector.

        Args:
            dc (DataCollector): The datacollector containing the table to extracted.
            run_number (int): The run number is passed in so that a column for run
                number may be added to the table.
            table_name (str): The name of the table to be extracted from the
                datacollector.

        Returns:
            (pd.DataFrame|None): A pandas DataFrame containing the table specified in
                the arguments. If this is not found, None is returned.
        """
        if table_name in list(dc.tables.keys()):
            iteration_run_data = dc.get_table_dataframe(table_name)
            iteration_run_data["Run_number"] = run_number
            return iteration_run_data
        else:
            warnings.warn(
                "Specified table '" + table_name + "' not found in datacollector", stacklevel=2
            )
            warnings.warn(
                "Available tables are: " + ", ".join(list(dc.tables.keys())), stacklevel=2
            )
            return None

    def get_aggregated_dataframe(self, table_name: str) -> Optional[pd.DataFrame]:
        """Return aggregated data for table_name in single DataFrame.

        Args:
            table_name (str): The name of the table to save to a dataframe.

        Returns:
            pd.DataFrame: A dataframe containing the aggregated data.
        """
        # Typing is ignored here - We know that the format of table is str, pd.DataFrame
        # as the method is called with an argument
        log: dict[str, pd.DataFrame] = self.output_datacollector_tables(table_name)  # type: ignore
        if all([df is None for df in log.values()]):
            warnings.warn(f"No log information for {table_name}.", stacklevel=2)
            return None

        return pd.concat([d for d in log.values() if d is not None])

    def save_tables_to_csv(self, table_names: list[str]) -> bool:
        """Iterate through the given tables and save each as a CSV.

        Args:
            table_names (list[str]): A list of all the tables you want to save to CSV.
            output_path (str): Folder path to save the CSV file to.

        Returns:
            bool: Success for saving all tables. If one fails, this will return false.
        """
        valid_output_path = validate_save_path(self.output_path)

        if not valid_output_path:
            return False

        success = [self.save_table_to_csv(t) for t in table_names]

        return all(success)

    def save_table_to_csv(self, table_name: str) -> bool:
        """Save a table from the batchrunner to CSV.

        Args:
            table_name (str): The name of the table you wish to save.
            output_path (Union[str, Path]): Folder path to save the CSV file to.

        Returns:
            bool: Success of writing to CSV.
        """
        log_df = self.get_aggregated_dataframe(table_name)

        if log_df is None:
            return False

        data_collector = self.data_vars.keys()
        run_numbers = [key[-1] for key in data_collector]
        valid_output_path = add_iteration_stamp(
            path=self.output_path, run_count=run_numbers[-1] + 1
        )

        if not valid_output_path:
            return False

        output_location = Path.joinpath(Path(valid_output_path), f"{table_name}.csv")

        try:
            with output_location.open(mode="w", newline="") as f:
                log_df.to_csv(f)

            return True
        except Exception as e:
            msg = (
                f"An error occurred writing {table_name} to file "
                + f"[{output_location.absolute()}].\n"
                + f"Caused by {type(e)}"
            )
            warnings.warn(msg, stacklevel=2)
            return False

    def collect_data_collector(self, model: RetModel) -> dict[str, Any]:
        """Collect data collector.

        Args:
            model (RetModel): The model used by the batchrunner containing the
                datacollector.

        Returns:
            data_vars (dict[str, Any]): A dictionary containing the datacollector from
                a given model.
        """
        data_vars = {}
        if self.collect_datacollector:
            data_vars["Data Collector"] = model.datacollector
        return data_vars

    def get_run_params_dataframe(self) -> pd.DataFrame:
        """Generate a pandas DataFrame summarising the run parameters and run numbers.

        Returns:
            df (pd.DataFrame): A dataframe containing a list of all the run parameters
                with the run number.
        """
        index_cols = list()
        for params in self.parameters_list:
            for index in params.keys():
                if index not in index_cols:
                    index_cols.append(index)
        index_cols = list(index_cols) + ["Run"]
        data = self.run_info
        df = pd.DataFrame.from_records(data, columns=index_cols)
        return df

    def find_and_copy_map(self, map_name=None) -> None:
        """Searches for map files and copies them to the output folder.

        This function should be called from a run_batch file or a file in the same folder.
        It either copies a specified map file or defaults to base maps if none is specified.
        """
        image_path = self._determine_image_path()
        map_file = self._find_map_file(map_name, image_path)

        if map_file:
            self._copy_map_file(map_file)
        else:
            warnings.warn(f"No map file found to copy in: {image_path}", stacklevel=2)

    def _determine_image_path(self):
        """Determines the image path based on the caller's file location."""
        caller_path = inspect.stack()[2].filename
        caller_directory = os.path.dirname(os.path.abspath(caller_path))
        folders = [
            entry
            for entry in os.listdir(caller_directory)
            if os.path.isdir(os.path.join(caller_directory, entry)) and entry != "__pycache__"
        ]
        if len(folders) == 1:
            base_path = os.path.join(caller_directory, folders[0])
        elif len(folders) == 0:
            base_path = caller_directory
        else:
            base_path = os.path.join(caller_directory, os.path.basename(caller_directory))

        image_path = os.path.join(base_path, "images")
        return image_path if os.path.exists(image_path) else base_path

    def _find_map_file(self, map_name, image_path):
        """Finds the map file based on the provided map_name or defaults to base maps."""
        if map_name is None:
            for base_map in [
                "Base_map_culture.png",
                "base_map_culture.png",
                "Base_map.png",
                "base_map.png",
            ]:
                path = os.path.join(image_path, base_map)
                if os.path.exists(path):
                    return path
        else:
            path = os.path.join(image_path, map_name)
            if os.path.exists(path):
                return path
            else:
                warnings.warn(f"No map file found by the name: {map_name}", stacklevel=2)
        return None

    def _copy_map_file(self, map_file):
        """Copies the map file to the output folder."""
        output_folder = self._get_output_folder_path()
        if output_folder:
            print(f"Copying {map_file} to {output_folder}")
            copy(map_file, output_folder)

    def _get_output_folder_path(self):
        """Determines the path to the output folder."""
        output_path = validate_save_path(self.output_path)
        return os.path.join(output_path, "assets", "Base_map.png")


class LatinHypercubeBatchRunner(FixedReportingBatchRunner):
    """Batchrunner using the Latin Hypercube sampling method.

    This class is instantiated with a model class, and model parameters
    associated with one or more values. It is also instantiated with model and
    agent-level reporters, dictionaries mapping a variable name to a function
    which collects some data from the model or its agents at the end of the run
    and stores it.

    Note that by default, the reporters only collect data at the *end* of the
    run. To get step by step data, simply have a reporter store the model's
    entire DataCollector object.
    """

    def __init__(
        self,
        model_cls: type,
        output_path: str,
        variable_parameters: Optional[dict[str, Any]] = None,
        fixed_parameters: Optional[dict[str, Any]] = None,
        iterations: int = 1,
        max_steps: int = 1000,
        save_every: Optional[int] = None,
        model_reporters: Optional[dict[str, Callable]] = None,
        agent_reporters: Optional[dict[str, str]] = None,
        n_experiments: int = 1,
        random_state: Optional[int] = None,
        collect_datacollector: bool = False,
        multiprocessing_cores: Optional[int] = None,
        ignore_multiprocessing: bool = False,
        tables_to_save: Optional[list] = None,
        model_seed: SingleOrListOptional[int] = None,
    ):
        """Create a new BatchRunner for a given model with the given parameters.

        Args:
            model_cls (type): The class of model to batch-run.
            output_path (str): The output location to save data to.
            variable_parameters (Optional[dict[str, Any]]): Dictionary of parameters to
                lists of values. The model will be run with every combination of these
                parameters. For example, given variable_parameters of
                    {"param_1": range(5),
                     "param_2": [1, 5, 10]}
                models will be run with {param_1=1, param_2=1},
                    {param_1=2, param_2=1}, ..., {param_1=4, param_2=10}.
                Defaults to None.
            fixed_parameters (Optional[dict[str, Any]]): Dictionary of parameters that
                stay same through all batch runs. For example, given fixed_parameters of
                    {"constant_parameter": 3},
                every instantiated model will be passed constant_parameter=3 as a kwarg.
                Defaults to None
            iterations (int): The total number of times to run the model for each
                combination of parameters. Defaults to 1.
            max_steps (int): Upper limit of steps above which each run will be halted if
                it hasn't halted on its own. Defaults to 1000.
            save_every (Optional[int]): The number of times to run the model before saving
                DataCollector tables. If None, no outputs will be automatically saved.
                Defaults to None.
            model_reporters (Optional[dict[str, Callable]]): The dictionary of variables
                to collect on each run at the end, with variable names mapped to a
                function to collect them. For example:
                    {"agent_count": lambda m: m.schedule.get_agent_count()}
                Defaults to None.
            agent_reporters (Optional[dict[str, str]]): Like model_reporters, but each
                variable is now collected at the level of each agent present in the
                model at the end of the run. Defaults to None
            n_experiments (int): The number of experiments to run. Defaults to 1.
            random_state (Optional[int]): A seed to ensure the LHS is reproducible.
                Where None, the sampling cannot be replicated. Defaults to None.
            collect_datacollector (bool): Whether or not to collect output data.
                Defaults to False.
            multiprocessing_cores (Optional[int]): The number of cores to be used for
                multiprocessing batchrunning. Defaults to None, resulting in the number
                of cores to match the system cpu count.
            ignore_multiprocessing (bool): Whether the use the run_all method from base class
                without any multiprocessing. Defaults to False.
            tables_to_save (Optional[list]): List of DataCollector tables to save.
                If None all default tables will be saved. Defaults to None.
            model_seed (SingleOrListOptional[int]): Seed for model. If int, the same model
                seed will be used to set RNG across all batch runs.
                If list[int], model_seed[run_count] will be used to set RNG.
                If none, random_numbers[run_count] will be used to set RNG.
                Defaults to None.
        """
        super().__init__(
            model_cls,
            output_path,
            LatinHypercubeParameterSampler(variable_parameters, n_experiments, random_state),
            fixed_parameters,
            iterations,
            max_steps,
            save_every,
            model_reporters,
            agent_reporters,
            collect_datacollector,
            multiprocessing_cores,
            ignore_multiprocessing,
            tables_to_save,
            model_seed,
        )


class ParamProductBatchRunner(FixedReportingBatchRunner):
    """Batchrunner using the standard parameter product.

    This class is instantiated with a model class, and model parameters
    associated with one or more values. It is also instantiated with model and
    agent-level reporters, dictionaries mapping a variable name to a function
    which collects some data from the model or its agents at the end of the run
    and stores it.

    Note that by default, the reporters only collect data at the *end* of the
    run. To get step by step data, simply choose to collect the DataCollector.

    """

    def __init__(
        self,
        model_cls: type,
        output_path: str,
        variable_parameters: Optional[dict[str, Any]] = None,
        fixed_parameters: Optional[dict[str, Any]] = None,
        iterations: int = 1,
        max_steps: int = 1000,
        save_every: Optional[int] = None,
        model_reporters: Optional[dict[str, Callable]] = None,
        agent_reporters: Optional[dict[str, str]] = None,
        collect_datacollector: bool = False,
        multiprocessing_cores: Optional[int] = None,
        ignore_multiprocessing: bool = False,
        tables_to_save: Optional[list] = None,
        model_seed: SingleOrListOptional[int] = None,
    ):
        """Create a new BatchRunner for a given model with the given parameters.

        Args:
            model_cls (type): The class of model to batch-run.
            output_path (str): The output location to save data to.
            variable_parameters (Optional[dict[str, Any]): Dictionary of parameters to
                lists of values. The model will be run with every combination of these
                parameters. For example, given variable_parameters of
                    {"param_1": range(5),
                     "param_2": [1, 5, 10]}
                models will be run with {param_1=1, param_2=1},
                    {param_1=2, param_2=1}, ..., {param_1=4, param_2=10}.
                Defaults to None.
            fixed_parameters (Optional[dict[str, Any]]): Dictionary of parameters that
                stay same through all batch runs. For example, given fixed_parameters of
                    {"constant_parameter": 3},
                every instantiated model will be passed constant_parameter=3 as a kwarg.
                Defaults to None.
            iterations (int): The total number of times to run the model for each
                combination of parameters. Defaults to 1.
            max_steps (int): Upper limit of steps above which each run will be halted
                if it hasn't halted on its own. Defaults to 1000.
            save_every (Optional[int]): The number of times to run the model before saving
                DataCollector tables. If None, no outputs will be automatically saved.
                Defaults to None.
            model_reporters (Optional[dict[str, Callable]]): The dictionary of variables
                to collect on each run at the end, with variable names mapped to a
                function to collect them. For example:
                    {"agent_count": lambda m: m.schedule.get_agent_count()}
                Defaults to None.
            agent_reporters (Optional[dict[str, str]]): Like model_reporters, but each
                variable is now collected at the level of each agent present in the
                model at the end of the run. Defaults to None.
            collect_datacollector (bool): Whether or not to collect output data.
                Defaults to False.
            multiprocessing_cores (Optional[int]): The number of cores to be used for
                multiprocessing batchrunning. Defaults to None, resulting in the number
                of cores to match the system cpu count.
            ignore_multiprocessing (bool): Whether the use the run_all method from base class
                without any multiprocessing. Defaults to False.
            tables_to_save (Optional[list]): List of DataCollector tables to save.
                If None all default tables will be saved. Defaults to None.
            model_seed (SingleOrListOptional[int]): Seed for model. If int, the same model
                seed will be used to set RNG across all batch runs.
                If list[int], model_seed[run_count] will be used to set RNG.
                If none, random_numbers[run_count] will be used to set RNG.
                Defaults to None.
        """
        super().__init__(
            model_cls,
            output_path,
            ParameterProduct(variable_parameters),
            fixed_parameters,
            iterations,
            max_steps,
            save_every,
            model_reporters,
            agent_reporters,
            collect_datacollector,
            multiprocessing_cores,
            ignore_multiprocessing,
            tables_to_save,
            model_seed,
        )
