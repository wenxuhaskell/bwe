import d3rlpy
import numpy as np
import optuna
from optuna.pruners import MedianPruner, PatientPruner
from sklearn.preprocessing import MinMaxScaler
import time

from BweReward import RewardFunction
from BweUtils import get_device, load_multiple_files, load_train_data_from_file
from hparam import BweHyperParamSampler, BweTuner


def objective(trial: optuna.Trial) -> float:
    # 1 np = 100 json files = 100 episodes
    datafiles = load_multiple_files("/home/code/bandwidth_challenge/data/testbed_np_comp100_clean", 1)
    file = datafiles[0]
    log_dir = "./logs/test_optuna_log"

    rf = RewardFunction("QOE_V1")

    device, rank, world_size = get_device(False)

    print(f"Worker {rank} starts creating MDP dataset...")
    t1 = time.process_time()
    observations, actions, rewards, terminals = load_train_data_from_file(file)
    obsScaler = MinMaxScaler()
    observations = obsScaler.fit_transform(observations)
    if not rewards:
        rewards = np.array([rf(o) for o in observations])
        r_last = rewards[-1]
        rewards = np.append(rewards[1:], r_last)

    start = 0
    end = len(actions)
    # divide dataset
    if world_size > 1:
        num_transitions = end
        num_transitions_per_worker = num_transitions // world_size
        start = rank * num_transitions_per_worker
        end = (rank + 1) * num_transitions_per_worker

    terminals[end - 1] = 1

    dataset = d3rlpy.dataset.MDPDataset(
        observations=observations[start:end],
        actions=actions[start:end],
        rewards=rewards[start:end],
        terminals=terminals[start:end],
        action_space=d3rlpy.ActionSpace.CONTINUOUS,
    )

    t2 = time.process_time()
    print(f"Worker {rank} finishes with creating MDP dataset - {t2-t1} s.")

    n_steps = dataset.transition_count
    n_steps_per_epoch = min(n_steps, 10000)  # ~ 36 big epochs, evaluate after each epoch
    n_epochs = n_steps // n_steps_per_epoch
    print(f"Worker {rank} train {n_steps} steps, {n_steps_per_epoch} steps per epoch for {n_epochs} epochs")

    # model parameters
    hparams = BweHyperParamSampler.create_sampler("CQL").sample_hyperparams(trial)
    cql = d3rlpy.algos.CQLConfig(**hparams).create(device=device)

    try:
        for epoch, metrics in cql.fitter(
            dataset=dataset,
            n_steps=n_steps,
            n_steps_per_epoch=n_steps_per_epoch,
            with_timestamp=False,
            logger_adapter=d3rlpy.logging.TensorboardAdapterFactory(log_dir),
            evaluators={
                'td_error': d3rlpy.metrics.TDErrorEvaluator(dataset.episodes[:1]),
                'discounted_advantage': d3rlpy.metrics.evaluators.DiscountedSumOfAdvantageEvaluator(
                    dataset.episodes[:1]
                ),
                'average_value': d3rlpy.metrics.evaluators.AverageValueEstimationEvaluator(dataset.episodes[:1]),
                'action_diff': d3rlpy.metrics.evaluators.ContinuousActionDiffEvaluator(dataset.episodes[:1]),
            },
        ):
            print(f"epoch: {epoch}, metrics: {metrics}")
            # choose reporting metric here, e.g., action_diff
            trial.report(metrics["action_diff"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    except AssertionError as e:
        # sometimes random hyperparams can generate NaNs
        print(e)
        return float("nan")

    return metrics["action_diff"]


if __name__ == "__main__":
    tuner = BweTuner(
        pruner=PatientPruner(wrapped_pruner=MedianPruner(), patience=3),
        study_name="test",
        save_path="./logs/test_optuna_log",
    )

    tuner.tune(objective, n_jobs=1)  # later switch to n_jobs=-1 to use all cores
