import numpy as np
import os
import pickle as pkl
import time
from typing import Callable, Optional, Sequence, Union

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler


class BweTuner:
    def __init__(
        self,
        direction: str = "maximize",
        aggregation_type: str = "average",
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
        study_name: str = "",
        save_path: str = None,
    ):
        self.direction = direction
        self.aggregation_type = aggregation_type
        if self.aggregation_type == "average":
            self.aggregation_fn = np.average
        elif self.aggregation_type == "median":
            self.aggregation_fn = np.median
        elif self.aggregation_type == "max":
            self.aggregation_fn = np.max
        elif self.aggregation_type == "min":
            self.aggregation_fn = np.min
        else:
            raise ValueError(f"Unknown aggregation type {self.aggregation_type}")
        self.pruner = pruner if pruner is not None else MedianPruner()
        self.sampler = sampler if sampler is not None else TPESampler()
        self.study_name = study_name if study_name.strip() else f"study_{int(time.time())}"
        self.save_path = save_path
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def tune(
        self,
        objective: Callable[[optuna.Trial], Union[float, Sequence[float]]],
        n_trials: int | None = None,
        n_jobs: int = -1,
    ) -> optuna.trial.FrozenTrial:
        # create study
        study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            load_if_exists=True,
            pruner=self.pruner,
            sampler=self.sampler,
        )

        try:
            study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
        except KeyboardInterrupt:
            pass

        if self.save_path:
            with open(f"{self.save_path}/{self.study_name}.pkl", "wb+") as f:
                pkl.dump(study, f)

        print(
            f"Number of finished trials: {len(study.trials)}\n"
            "Best trial:\n"
            f"  Value: {study.best_trial.value}\n"
            "  Params:\n"
            f"    {chr(10).join(f'{key}: {value}' for key, value in study.best_trial.params.items())}\n"
            "  User attrs:\n"
            f"    {chr(10).join(f'{key}: {value}' for key, value in study.best_trial.user_attrs.items())}"
        )

        return study.best_trial
