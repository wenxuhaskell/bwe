import argparse
import json
import os
from datetime import datetime
import random
from typing import Dict, Any
import d3rlpy
import optuna.exceptions
import optuna.trial
from optuna.pruners import MedianPruner, PatientPruner
from optuna.samplers import TPESampler

import torch
import torch.distributed as dist
from functools import partial
import BweModels
from BweEvaluators import BweTDErrorEvaluator, BweContinuousActionDiffEvaluator, BweAverageValueEstimationEvaluator
from BweUtils import get_device

N_TRIALS = 100
N_STARTUP_TRIALS = 5

def sample_td3plusbc_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for TD3+BC hyperparameters."""
    tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.05, 0.1])
    gamma = trial.suggest_categorical("gamma", [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    alpha = trial.suggest_float("alpha", low=2, high=3, step=0.1)
    actor_learning_rate = 1.0 - trial.suggest_float("actor_learning_rate", low=1e-4, high=1e-2, step=1e-4)
    critic_learning_rate = 1.0 - trial.suggest_float("critic_learning_rate", low=1e-4, high=1e-2, step=1e-4)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])

    # Display true values.
    trial.set_user_attr("tau_", tau)
    trial.set_user_attr("gamma_", gamma)
    trial.set_user_attr("alpha_", alpha)
    trial.set_user_attr("actor_learning_rate_", actor_learning_rate)
    trial.set_user_attr("critic_learning_rate_", critic_learning_rate)
    trial.set_user_attr("batch_size_", batch_size)

    return {
        "tau": tau,
        "gamma": gamma,
        "alpha": alpha,
        "actor_learning_rate": actor_learning_rate,
        "critic_learning_rate": critic_learning_rate,
        "batch_size": batch_size
    }


def sample_crr_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for CRR hyperparameters."""
    tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.05, 0.1])
    gamma = trial.suggest_categorical("gamma", [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
#    beta = trial.suggest_float("beta", low=0.7, high=1.0, step=0.1)
    actor_learning_rate = 1.0 - trial.suggest_float("actor_learning_rate", low=1e-4, high=1e-2, step=1e-4)
    critic_learning_rate = 1.0 - trial.suggest_float("critic_learning_rate", low=1e-4, high=1e-2, step=1e-4)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])

    # Display true values.
    trial.set_user_attr("tau_", tau)
    trial.set_user_attr("gamma_", gamma)
#    trial.set_user_attr("beta_", beta)
    trial.set_user_attr("actor_learning_rate_", actor_learning_rate)
    trial.set_user_attr("critic_learning_rate_", critic_learning_rate)
    trial.set_user_attr("batch_size_", batch_size)

    return {
        "tau": tau,
        "gamma": gamma,
#        "beta": beta,
        "actor_learning_rate": actor_learning_rate,
        "critic_learning_rate": critic_learning_rate,
        "batch_size": batch_size
    }


def sample_ddpg_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for DDPG hyperparameters."""
    tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.05, 0.1])
    gamma = trial.suggest_categorical("gamma", [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    actor_learning_rate = 1.0 - trial.suggest_float("actor_learning_rate", low=1e-4, high=1e-2, step=1e-4)
    critic_learning_rate = 1.0 - trial.suggest_float("critic_learning_rate", low=1e-4, high=1e-2, step=1e-4)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])

    # Display true values.
    trial.set_user_attr("tau_", tau)
    trial.set_user_attr("gamma_", gamma)
    trial.set_user_attr("actor_learning_rate_", actor_learning_rate)
    trial.set_user_attr("critic_learning_rate_", critic_learning_rate)
    trial.set_user_attr("batch_size_", batch_size)

    return {
        "tau": tau,
        "gamma": gamma,
        "actor_learning_rate": actor_learning_rate,
        "critic_learning_rate": critic_learning_rate,
        "batch_size": batch_size
    }


def sample_sac_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for SAC hyperparameters."""
    tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.05, 0.1])
    gamma = trial.suggest_categorical("gamma", [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    actor_learning_rate = 1.0 - trial.suggest_float("actor_learning_rate", low=1e-4, high=1e-2, step=1e-4)
    critic_learning_rate = 1.0 - trial.suggest_float("critic_learning_rate", low=1e-4, high=1e-2, step=1e-4)
    temp_learning_rate = 1.0 - trial.suggest_float("temp_learning_rate", low=1e-4, high=1e-2, step=1e-4)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    
    # Display true values.
    trial.set_user_attr("tau_", tau)
    trial.set_user_attr("gamma_", gamma)
    trial.set_user_attr("actor_learning_rate_", actor_learning_rate)
    trial.set_user_attr("critic_learning_rate_", critic_learning_rate)
    trial.set_user_attr("temp_learning_rate_", temp_learning_rate)
    trial.set_user_attr("batch_size_", batch_size)
    
    return {
        "tau": tau,
        "gamma": gamma,
        "actor_learning_rate": actor_learning_rate,
        "critic_learning_rate": critic_learning_rate,
        "temp_learning_rate": temp_learning_rate,
        "batch_size": batch_size
    }


def sample_cql_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for CQL hyperparameters."""
    tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.05, 0.1])
    gamma = trial.suggest_categorical("gamma", [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    conservative_weight = trial.suggest_categorical("conservative_weight", [1.0, 5.0, 10.0])
    actor_learning_rate = 1.0 - trial.suggest_float("actor_learning_rate", low=1e-4, high=1e-2, step=1e-4)
    critic_learning_rate = 1.0 - trial.suggest_float("critic_learning_rate", low=1e-4, high=1e-2, step=1e-4)
    temp_learning_rate = 1.0 - trial.suggest_float("temp_learning_rate", low=1e-2, high=0.1, step=1e-2)
    alpha_learning_rate = trial.suggest_float("alpha_learning_rate", low=1e-4, high=0.1, step=1e-4)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    
    # Display true values.
    trial.set_user_attr("tau_", tau)
    trial.set_user_attr("gamma_", gamma)
    trial.set_user_attr("conservative_weight_", conservative_weight)
    trial.set_user_attr("actor_learning_rate_", actor_learning_rate)
    trial.set_user_attr("critic_learning_rate_", critic_learning_rate)
    trial.set_user_attr("temp_learning_rate_", temp_learning_rate)
    trial.set_user_attr("alpha_learning_rate_", alpha_learning_rate)
    trial.set_user_attr("batch_size_", batch_size)
    
    return {
        "tau": tau,
        "gamma": gamma,
        "conservative_weight": conservative_weight,
        "actor_learning_rate": actor_learning_rate,
        "critic_learning_rate": critic_learning_rate,
        "temp_learning_rate": temp_learning_rate,
        "alpha_learning_rate": alpha_learning_rate,
        "batch_size": batch_size
    }


def sample_bcq_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for BCQ hyperparameters."""
    tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.05, 0.1])
    gamma = trial.suggest_categorical("gamma", [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    beta = trial.suggest_float("beta", low=0.1, high=0.9, step=0.1)
    lam = 1.0 - trial.suggest_float("lam", 0.1, 0.9, step=0.1)
    actor_learning_rate = 1.0 - trial.suggest_float("actor_learning_rate", low=1e-4, high=1e-2, step=1e-4)
    critic_learning_rate = 1.0 - trial.suggest_float("critic_learning_rate", low=1e-4, high=1e-2, step=1e-4)
    imitator_learning_rate = 1.0 - trial.suggest_float("imitator_learning_rate", low=1e-4, high=1e-2, step=1e-4)
    action_flexibility = 1.0 - trial.suggest_float("action_flexibility", low=1e-2, high=0.1, step=1e-2)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    
    # Display true values.
    trial.set_user_attr("tau_", tau)
    trial.set_user_attr("beta_", beta)
    trial.set_user_attr("gamma_", gamma)
    trial.set_user_attr("lam_", lam)
    trial.set_user_attr("actor_learning_rate_", actor_learning_rate)
    trial.set_user_attr("critic_learning_rate_", critic_learning_rate)
    trial.set_user_attr("imitator_learning_rate_", imitator_learning_rate)
    trial.set_user_attr("action_flexibility_", action_flexibility)
    trial.set_user_attr("batch_size_", batch_size)

    return {
        "tau": tau,
        "beta": beta,
        "gamma": gamma,
        "lam": lam,
        "actor_learning_rate": actor_learning_rate,
        "critic_learning_rate": critic_learning_rate,
        "imitator_learning_rate": imitator_learning_rate,
        "action_flexibility": action_flexibility,
        "batch_size": batch_size
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
    elif algo_name == 'DDPG':
        params.update(sample_ddpg_params(trial))
    elif algo_name == 'CRR':
        params.update(sample_crr_params(trial))
    elif algo_name == 'TD3PLUSBC':
        params.update(sample_td3plusbc_params(trial))
    else:
        print("Algorithm is not supported")
        raise Exception()

    # Create the RL model.
    drl.create_model(params)

    # create evaluators
    test_episodes = dataset.episodes[:1]
    tune_eva = None
    match(params['tune_evaluator'].upper()):
        case "ACTION_DIFF":
            tune_eva = BweContinuousActionDiffEvaluator(test_episodes, trial)
        case "TD_ERROR":
            tune_eva = BweTDErrorEvaluator(test_episodes, trial)
#        case "AVE_VALUE_EST":
#            tune_eva = BweAverageValueEstimationEvaluator(test_episodes, trial)
        case _:
            raise ValueError(f"Unsupported evaluator for finetuning - {params['tune_evaluator']}!")

    evaluators = {
        params['tune_evaluator']: tune_eva
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

    value = tune_eva.get_last_value()
    # report the value (contained in evaluator) back to trail?
    return value


def save_best_trail(study, params):
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
    best_trial.params['tune_evaluator'] = params['tune_evaluator']
    best_trial.params['reward_func'] = params['reward_func_name']

    best_trial.user_attrs['completed'] = num_completed
    best_trial.user_attrs['pruned'] = num_pruned
    best_trial.user_attrs['timestamp'] = ts_str

    params_filename = f"./trials/{params['algorithm_name']}_params.json"
    attrs_filename = f"./trials/{params['algorithm_name']}_attrs.json"
    with open(params_filename, "a") as outfile:
        outfile.write('\n')
        json.dump(best_trial.params, outfile)
    with open(attrs_filename, "a") as outfile:
        outfile.write('\n')
        json.dump(best_trial.user_attrs, outfile)
    

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

    if 'finetune' in params and params['finetune']:
        sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
        # Do not prune before 1/3 of the max budget is used.
        #pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=2)
        pruner = PatientPruner(wrapped_pruner=MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=2), patience=3)
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
            study.optimize(objective_cb, n_trials=N_TRIALS)
        except KeyboardInterrupt:
            pass

        save_best_trail(study, params)

    else:
        bwe.create_model(params)
        bwe.train_model_gradually()


    if params['ddp'] == True and torch.cuda.is_available():
        print("DDP finishes.")
        d3rlpy.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
