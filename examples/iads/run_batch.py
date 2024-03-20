"""Demonstrator for the IADS scenario using batch-runner.

Iterates over one iteration.
"""

from ret.batchrunner import FixedReportingBatchRunner
from iads.model import IADS


params = None

outputs = "./output/IADS"

br = FixedReportingBatchRunner(
    model_cls=IADS,
    output_path=outputs,
    parameters_list=params,
    iterations=5,
    max_steps=400,
    collect_datacollector=True,
    ignore_multiprocessing=True,
)

if __name__ == "__main__":
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
            "position_and_culture",
        ],
    )
    br.find_and_copy_map()
