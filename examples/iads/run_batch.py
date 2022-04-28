"""Demonstrator for the IADS scenario using batch-runner.

Iterates over one iteration.
"""

from mesa_ret.batchrunner import FixedReportingBatchRunner

from iads.model import IADS

params = None

br = FixedReportingBatchRunner(
    model_cls=IADS,
    parameters_list=params,
    iterations=1,
    max_steps=400,
    collect_datacollector=True,
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
        ]
    )
