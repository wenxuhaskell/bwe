import d3rlpy
import os
from typing import Any

class BweAdapterFactory(d3rlpy.logging.FileAdapterFactory):
    def __init__(self, root_dir: str = "bwelogs", output_model_name: str = "model"):
        super().__init__(root_dir)
        self._output_model_name = output_model_name

    def create(self, experiment_name: str) -> d3rlpy.logging.FileAdapter:
        logdir = self._root_dir
#        logdir = os.path.join(self._root_dir, experiment_name)
        return BweAdapter(experiment_name, self._root_dir, self._output_model_name)

class BweAdapter(d3rlpy.logging.FileAdapter):
    def __init__(self, experiment_name, root_dir: str = "d3rllogs", output_model_name: str = "model"):
        super().__init__(root_dir)
        self._experiment_name = experiment_name
        self._output_model_name = output_model_name

    def write_metric(
            self, epoch: int, step: int, name: str, value: float
    ) -> None:
        path = os.path.join(self._logdir, f"{name}.csv")
        with open(path, "a") as f:
            print(f"{self._experiment_name},{epoch},{step},{value}", file=f)

#    def after_write_metric(self, epoch: int, step: int) -> None:
#        path = os.path.join(self._logdir, "experiments.csv")
#        with open(path, "a") as f:
#            print(f"{self._experiment_name}", file=f)


    def save_model(self, epoch: int, algo: Any) -> None:
        # save entire model
#        model_path = os.path.join(self._logdir, f"model_{epoch}.d3")
        model_path = os.path.join(self._logdir, f"{self._output_model_name}.d3")
        algo.save(model_path)
        print(f"Model parameters are saved to {model_path}")