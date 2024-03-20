"""Demonstrator for the IADSIntegration scenario using batch-runner.

Currently set up to:
Iterates over 6 iteration at  maximum of 500 steps.
create output folder in current directory
saving agents and position_and_culture tables every 2 iterations
"""

from ret.batchrunner import FixedReportingBatchRunner

from iads_v2.model import IADSIntegration


br = FixedReportingBatchRunner(
    model_cls=IADSIntegration,
    output_path="./output/IADS_v2",
    iterations=2,
    max_steps=500,
    collect_datacollector=True,
    save_every=2,
    tables_to_save=["agents", "position_and_culture"],
    model_seed=[2, 4],
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
            "model_seed",
        ],
    )
    br.find_and_copy_map(map_name="Base_map.png")
