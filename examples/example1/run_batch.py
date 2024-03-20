"""Batch run Example model."""

from datetime import datetime, timedelta

from ret.batchrunner import FixedReportingBatchRunner

from ret_example.model import ExampleModel

fixed_params = {
    "start_time": datetime(year=2022, month=1, day=19, hour=14),
    "time_step": timedelta(minutes=1),
    "end_time": datetime(year=2022, month=1, day=19, hour=16),
}

br = FixedReportingBatchRunner(
    model_cls=ExampleModel,
    parameters_list=None,
    fixed_parameters=fixed_params,
    iterations=5,
    max_steps=121,
    collect_datacollector=True,
    output_path="./output/Example_1",
    ignore_multiprocessing=True,
)

br.run_all()
deaths = br.get_aggregated_dataframe("deaths")
print("A sample of the deaths table...")
try:
    print(deaths.sample(5))
except ValueError:
    print("There were no deaths. Congratulations on solving war.")
