"""Run Example model for playback output."""
from datetime import datetime, timedelta

from mesa_ret.batchrunner import FixedReportingBatchRunner

from ret_example.model import ExampleModel

fixed_params = {
    "start_time": datetime(year=2022, month=1, day=19, hour=14),
    "time_step": timedelta(minutes=1),
    "end_time": datetime(year=2022, month=1, day=19, hour=16),
}
if __name__ == "__main__":
    br = FixedReportingBatchRunner(
        model_cls=ExampleModel,
        parameters_list=None,
        fixed_parameters=fixed_params,
        iterations=1,
        max_steps=121,
        collect_datacollector=True,
        ignore_multiprocessing=True,
    )

    br.run_all()
    br.save_tables_to_csv(
        [
            "agents",
            "behaviour_log",
            "task_log",
            "trigger_log",
            "deaths",
            "shots_fired",
            "observation_record",
            "perception_record",
            "behaviour_selection",
        ]
    )
