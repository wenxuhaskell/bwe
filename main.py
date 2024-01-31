import argparse
import json
import os
from datetime import datetime
import random
from typing import Dict, Any
import d3rlpy
import optuna.exceptions
import optuna.trial
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

import torch
import torch.distributed as dist
from functools import partial
import BweModels
from BweEvaluators import BweTDErrorEvaluator
from BweUtils import get_device

N_TRIALS = 100
N_STARTUP_TRIALS = 5


def sample_sac_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for SAC hyperparameters."""
    tau = trial.suggest_float("tau", low=1e-3, high=1e-2, step=1e-3)
    gamma = 1.0 - trial.suggest_float("gamma", low=1e-2, high=0.1, step=1e-2)
    actor_learning_rate = 1.0 - trial.suggest_float("actor_learning_rate", low=1e-4, high=1e-2, step=1e-4)
    critic_learning_rate = 1.0 - trial.suggest_float("critic_learning_rate", low=1e-4, high=1e-2, step=1e-4)
    temp_learning_rate = 1.0 - trial.suggest_float("temp_learning_rate", low=1e-4, high=1e-2, step=1e-4)

    # Display true values.
    trial.set_user_attr("tau_", tau)
    trial.set_user_attr("gamma_", gamma)
    trial.set_user_attr("actor_learning_rate_", actor_learning_rate)
    trial.set_user_attr("critic_learning_rate_", critic_learning_rate)
    trial.set_user_attr("temp_learning_rate_", temp_learning_rate)

    return {
        "tau": tau,
        "gamma": gamma,
        "actor_learning_rate": actor_learning_rate,
        "critic_learning_rate": critic_learning_rate,
        "temp_learning_rate": temp_learning_rate
    }


def sample_cql_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for CQL hyperparameters."""
    tau = trial.suggest_float("tau", low=1e-3, high=1e-2, step=1e-3)
    gamma = 1.0 - trial.suggest_float("gamma", low=1e-2, high=0.1, step=1e-2)
    conservative_weight = trial.suggest_float("conservative_weight", low=1.0, high=10, step=1)
    actor_learning_rate = 1.0 - trial.suggest_float("actor_learning_rate", low=1e-4, high=1e-2, step=1e-4)
    critic_learning_rate = 1.0 - trial.suggest_float("critic_learning_rate", low=1e-4, high=1e-2, step=1e-4)
    temp_learning_rate = 1.0 - trial.suggest_float("temp_learning_rate", low=1e-2, high=0.1, step=1e-2)
    alpha_learning_rate = trial.suggest_float("alpha_learning_rate", low=1e-4, high=0.1, step=1e-4)

    # Display true values.
    trial.set_user_attr("tau_", tau)
    trial.set_user_attr("gamma_", gamma)
    trial.set_user_attr("conservative_weight_", conservative_weight)
    trial.set_user_attr("actor_learning_rate_", actor_learning_rate)
    trial.set_user_attr("critic_learning_rate_", critic_learning_rate)
    trial.set_user_attr("temp_learning_rate_", temp_learning_rate)
    trial.set_user_attr("alpha_learning_rate_", alpha_learning_rate)

    return {
        "tau": tau,
        "gamma": gamma,
        "conservative_weight": conservative_weight,
        "actor_learning_rate": actor_learning_rate,
        "critic_learning_rate": critic_learning_rate,
        "temp_learning_rate": temp_learning_rate,
        "alpha_learning_rate": alpha_learning_rate
    }


def sample_bcq_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for BCQ hyperparameters."""
    tau = trial.suggest_float("tau", low=1e-3, high=1e-2, step=1e-3)
    beta = trial.suggest_float("beta", low=0.1, high=0.9, step=0.1)
    gamma = 1.0 - trial.suggest_float("gamma", low=1e-2, high=0.1, step=1e-2)
    lam = 1.0 - trial.suggest_float("lam", 0.1, 0.9, step=0.1)
    actor_learning_rate = 1.0 - trial.suggest_float("actor_learning_rate", low=1e-4, high=1e-2, step=1e-4)
    critic_learning_rate = 1.0 - trial.suggest_float("critic_learning_rate", low=1e-4, high=1e-2, step=1e-4)
    imitator_learning_rate = 1.0 - trial.suggest_float("imitator_learning_rate", low=1e-4, high=1e-2, step=1e-4)
    action_flexibility = 1.0 - trial.suggest_float("action_flexibility", low=1e-2, high=0.1, step=1e-2)
    learning_rate = trial.suggest_float("learning_rate", low=1e-4, high=0.1, step=1e-4)

    # Display true values.
    trial.set_user_attr("tau_", tau)
    trial.set_user_attr("beta_", beta)
    trial.set_user_attr("gamma_", gamma)
    trial.set_user_attr("lam_", lam)
    trial.set_user_attr("actor_learning_rate_", actor_learning_rate)
    trial.set_user_attr("critic_learning_rate_", critic_learning_rate)
    trial.set_user_attr("imitator_learning_rate_", imitator_learning_rate)
    trial.set_user_attr("action_flexibility_", action_flexibility)
    trial.set_user_attr("learning_rate_", learning_rate)

    return {
        "tau": tau,
        "beta": beta,
        "gamma": gamma,
        "lam": lam,
        "actor_learning_rate": actor_learning_rate,
        "critic_learning_rate": critic_learning_rate,
        "imitator_learning_rate": imitator_learning_rate,
        "action_flexibility": action_flexibility,
        "learning_rate": learning_rate
    }


def objective(drl: BweModels.BweDrl,
              dataset: d3rlpy.dataset.MDPDataset,
              params: dict,
              trial: optuna.Trial) -> float:
    # Sample hyperparameters.
    algo_name = drl.get_algo_name().upper()
    if algo_name == 'CQL':
        params.update(sample_cql_params(trial))
    elif algo_name == 'BCQ':
        params.update(sample_bcq_params(trial))
    elif algo_name == 'SAC':
        params.update(sample_sac_params(trial))
    else:
        print("Algorithm is not supported")
        raise Exception()

    # Create the RL model.
    drl.create_model(params)

    # create evaluators
    test_episodes = dataset.episodes[:1]
    td_eva = BweTDErrorEvaluator(test_episodes, trial)
    da_eva = d3rlpy.metrics.evaluators.DiscountedSumOfAdvantageEvaluator(test_episodes)
    av_eva = d3rlpy.metrics.evaluators.AverageValueEstimationEvaluator(test_episodes)
    ad_eva = d3rlpy.metrics.evaluators.ContinuousActionDiffEvaluator(test_episodes)
    evaluators = {
        'td_error': td_eva,
        'discounted_advantage': da_eva,
        'average_value': av_eva,
        'action_diff': ad_eva
    }

    nan_encountered = False
    try:
        drl.train(dataset, evaluators)
    except optuna.exceptions.TrialPruned as e:
        print('Pruned')
        raise optuna.exceptions.TrialPruned()
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN.
        print(e)
        nan_encountered = True

    # Tell the optimizer that the trial failed.
    if nan_encountered:
        return float("nan")

    value = td_eva.get_last_td_error()
    # report the value (contained in evaluator) back to trail?
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--conf", type=str, default="cqlconf.json")
    parser.add_argument("-d", "--ddp", default=False, action="store_true")
    args = parser.parse_args()

    # load the configuration parameters
    f = open(args.conf, "r")
    params = json.load(f)
    f.close()

    # add device
    params['ddp'] = args.ddp
    # get devices for training (overwrite the "device" parameter in json file)
    if 'device' not in params:
        params['device'], params['rank'], params['world_size'] = get_device(args.ddp)
    else:
        params['rank'] = 0
        params['world_size'] = 1

    # create the DRL object
    bwe = BweModels.BweDrl(params)

    if params['finetune']:
        sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
        # Do not prune before 1/3 of the max budget is used.
        pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=2)
        # list of training data files
        filenames = bwe.get_list_data_files()
        # randomly choose one file
        filename = random.choice(filenames)
        # load MDP dataset
        dataset = bwe.load_MDP_dataset(filename)
        # create a partial function for optuna
        objective_cb = partial(objective, bwe, dataset, params.copy())
        study = optuna.create_study(sampler=sampler, pruner=pruner, direction="minimize")
        try:
            study.optimize(objective_cb, n_trials=N_TRIALS, timeout=600)
        except KeyboardInterrupt:
            pass

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

        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        with open(f"./trials/best_trial_params_{timestamp}.json", "w") as outfile:
            best_trial.params['completed'] = len(completed_trials)
            best_trial.params['pruned'] = len(study.trials) - len(completed_trials)
            json.dump(best_trial.params, outfile)
        with open(f"./trials/best_trial_attrs_{timestamp}.json", "w") as outfile:
            best_trial.user_attrs['complete'] = len(completed_trials)
            best_trial.user_attrs['pruned'] = len(study.trials) - len(completed_trials)
            json.dump(best_trial.user_attrs, outfile)

    else:
        bwe.create_model(params)
        bwe.train_model_gradually()


    if params['ddp'] == True and torch.cuda.is_available():
        print("DDP finishes.")
        d3rlpy.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
