import queue
from typing import Optional, Sequence
import d3rlpy
import optuna
from optuna import TrialPruned, Trial

from d3rlpy.metrics import TDErrorEvaluator, ContinuousActionDiffEvaluator, AverageValueEstimationEvaluator
from d3rlpy.dataset import EpisodeBase, ReplayBuffer
from d3rlpy.interface import QLearningAlgoProtocol

class BweTDErrorEvaluator(TDErrorEvaluator):

    def __init__(self, episodes: Optional[Sequence[EpisodeBase]], trial: Trial = None):
        super().__init__(episodes)
        self._trial = trial
        self._epoch_idx = 0
        self._last_td_error = 0.0

    def get_last_value(self) -> float:
        return self._last_td_error

    def __call__(
        self,
        algo: QLearningAlgoProtocol,
        dataset: ReplayBuffer,
    ) -> float:
        self._last_td_error = super().__call__(algo, dataset)
        self._epoch_idx += 1
        # check if finetuning is needed.
        if self._trial:
            self._trial.report(value=self._last_td_error, step=self._epoch_idx)
            print(f"report to trial: {self._last_td_error}")
            if self._trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return self._last_td_error


class BweContinuousActionDiffEvaluator(ContinuousActionDiffEvaluator):

    def __init__(self, episodes: Optional[Sequence[EpisodeBase]], trial: Trial = None):
        super().__init__(episodes)
        self._trial = trial
        self._epoch_idx = 0
        self._last_ad_error = 0.0

    def get_last_value(self) -> float:
        return self._last_ad_error

    def __call__(
        self,
        algo: QLearningAlgoProtocol,
        dataset: ReplayBuffer,
    ) -> float:
        self._last_ad_error = super().__call__(algo, dataset)
        self._epoch_idx += 1
        # check if finetuning is needed.
        if self._trial:
            self._trial.report(value=self._last_ad_error, step=self._epoch_idx)
            print(f"report to trial: {self._last_ad_error}")
            if self._trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return self._last_ad_error


class BweAverageValueEstimationEvaluator(AverageValueEstimationEvaluator):

    def __init__(self, episodes: Optional[Sequence[EpisodeBase]], trial: Trial = None):
        super().__init__(episodes)
        self._trial = trial
        self._epoch_idx = 0
        self._last_av_error = 0.0

    def get_last_value(self) -> float:
        return self._last_av_error

    def __call__(
        self,
        algo: QLearningAlgoProtocol,
        dataset: ReplayBuffer,
    ) -> float:
        self._last_av_error = super().__call__(algo, dataset)
        self._epoch_idx += 1
        # check if finetuning is needed.
        if self._trial:
            self._trial.report(value=self._last_av_error, step=self._epoch_idx)
            print(f"report to trial: {self._last_av_error}")
            if self._trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return self._last_av_error