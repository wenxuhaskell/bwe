import d3rlpy
import numpy as np
import optuna
from optuna.pruners import MedianPruner, PatientPruner
from sklearn.preprocessing import MinMaxScaler
import time
from datetime import datetime
import json

from BweReward import RewardFunction
from BweUtils import get_device, load_multiple_files, load_train_data_from_file
from hparam import BweHyperParamSampler, BweTuner


def objective(trial: optuna.Trial) -> float:
    # 1 np = 100 json files = 100 episodes
    datafiles = load_multiple_files("/home/code/bandwidth_challenge/data/testbed_dataset", 1)
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


    # model parameters
    hparams = BweHyperParamSampler.create_sampler("CQL").sample_hyperparams(trial)
    cql = d3rlpy.algos.CQLConfig(**hparams).create(device=device)

    n_steps = dataset.transition_count // hparams['batch_size']
    n_steps_per_epoch = min(n_steps, 10000)  # ~ 36 big epochs, evaluate after each epoch
    print(n_steps)
    print(n_steps_per_epoch)
    n_epochs = n_steps // n_steps_per_epoch
    print(f"Worker {rank} train {n_steps} steps, {n_steps_per_epoch} steps per epoch for {n_epochs} epochs")

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

    study = tuner.tune(objective, n_jobs=1, n_trials=10)  # later switch to n_jobs=-1 to use all cores
    
    print("Number of finished trials: ", len(study.trials))
    completed_trials = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
    print("Number of pruned trials: ", len(study.trials)-len(completed_trials))

    best_trial = study.best_trial
    print("  value:", best_trial.value)

    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in best_trial.user_attrs.items():
        print("    {}: {}".format(key, value))


    num_completed = len(completed_trials)
    num_pruned = len(study.trials) - len(completed_trials)
    ts_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    best_trial.params['completed'] = num_completed
    best_trial.params['pruned'] = num_pruned
    best_trial.params['timestamp'] = ts_str

    best_trial.user_attrs['completed'] = num_completed
    best_trial.user_attrs['pruned'] = num_pruned
    best_trial.user_attrs['timestamp'] = ts_str


    params_filename = f"./trials/to_CQL_params.json"
    attrs_filename = f"./trials/to_CQL_attrs.json"
    with open(params_filename, "a") as outfile:
        outfile.write('\n')
        json.dump(best_trial.params, outfile)
    with open(attrs_filename, "a") as outfile:
        outfile.write('\n')
        json.dump(best_trial.user_attrs, outfile)
