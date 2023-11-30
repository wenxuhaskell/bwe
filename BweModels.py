from datetime import datetime
import d3rlpy
import numpy as np

import os
import BweReward
import BweEnv
from BweUtils import load_data
from BweLogger import BweAdapter, BweAdapterFactory
import onnxruntime as ort

class BweDrl:
    def __init__(self, params, algo : d3rlpy.base.LearnableBase):
        self._log_dir = params['logFolderName']
        self._train_data_dir = params['trainDataFolder']
        self._test_data_dir = params['testDataFolder']
        self._device = params['device']
        self._output_model_name = params['outputModelName']
        self._algo = algo
        rf_name = params['rewardFuncName']
        self._reward_func = getattr(BweReward, rf_name)

    def train_model (self):
        # load the list of log files under the given directory
        # iterate over files in
        # that directory
        files = sorted(os.listdir(self._train_data_dir))
        train_data_files = []
        for name in files:
            f = os.path.join(self._train_data_dir, name)
            # checking if it is a file
            if os.path.isfile(f):
                train_data_files.append(f)
        # Initially there is no pre-trained model
        start_date = datetime.now().strftime("%Y%m%d%H%M%S")
        self._log_dir = self._log_dir + "_" + start_date
#        datafiles = train_data_files[0:10]
        datafiles = train_data_files
        for filename in datafiles:
            # load the log file and prepare the dataset
            observations, actions = load_data(filename)
            # terminal flags
            terminals = np.random.randint(2, size=len(actions))
            # calculate reward
            rewards = np.zeros(len(observations))
            for i, o in enumerate(list(observations)):
                rewards[i] = self._reward_func(o)

            # create the offline learning dataset
            dataset = d3rlpy.dataset.MDPDataset(
                observations=observations,
                actions=actions,
                rewards=rewards,
                terminals=terminals,
            )

            # remove path and extension
            exp_filename = filename.split('/')[1].split('.')[0]
            n_steps = len(observations)
            n_steps_per_epoch = n_steps
            # offline training
            self._algo.fit(
                dataset,
                n_steps,
                n_steps_per_epoch,
                experiment_name=exp_filename,
                with_timestamp=False,
                logger_adapter=BweAdapterFactory(root_dir=self._log_dir, output_model_name=self._output_model_name),
                #        evaluators={
                #            'environment': d3rlpy.metrics.EnvironmentEvaluator(bwe_env),
                #        },
            )

        #
        print("Training of " + str(len(datafiles)) + " episodes!\n")
        print("The latest trained model is placed under the log folder" + self._log_dir)

    def evaluate_model_offline (self):

        # setup algorithm manually
        # if there is already a pre-trained model, load it
        if os.path.exists(os.path.join(self._log_dir, f"{self._output_model_name}.d3")):
            output_model_full_name = self._log_dir + '/' + self._output_model_name + '.d3'
            algo = d3rlpy.load_learnable(output_model_full_name, device=self._device)
            print("Load the pre-trained model from the file" + output_model_full_name + " for evaluation with emulated data!")
        else:
            print("There is no pre-trained model for evaluation!\n")
            return

        # to be used for online learning (or evaluation)
        bwe_env = BweEnv(observations, actions)
        _ = bwe_env.reset()

        # predict
        observation, _ = bwe_env.reset()
        while True:
            # Add batch dimension
            # observation = observation.reshape((1, len(observation)))
            # tt = np.expand_dims(observation, axis=0)
            tt = observation.reshape(1, len(observation))
            action = algo.predict(tt)[0]
            observation, reward, done, truncated, _ = bwe_env.step(action)
            if done:
                break

        # export as ONNX
        cql_new.save_policy("policy.onnx")

        # load ONNX policy via onnxruntime
        ort_session = ort.InferenceSession('policy.onnx', providers=["CPUExecutionProvider"])

        # observation
        observation, _ = bwe_env.reset()
        observation = observation.reshape((1, len(observation))).astype(np.float32)
        # returns greedy action
        action = ort_session.run(None, {'input_0': observation})
        print(action)
        assert action[0].shape == (1, 1)


def createCQL(params):
    # parameters for algorithm
    _device = params['device']

    _batch_size = params['batch_size']
    _initial_alpha = params["initial_alpha"]
    _gamma = params["gamma"]
    _alpha_threshold = params["alpha_threshold"]
    _conservative_weight = params["conservative_weight"]
    _initial_temperature = params["initial_temperature"]
    _n_action_samples = params["n_action_samples"]
    _n_critics = params["n_critics"]
    _tau = params["tau"]
    _actor_learning_rate = params["actor_learning_rate"]
    _critic_learning_rate = params["critic_learning_rate"]
    _temp_learning_rate = params["temp_learning_rate"]
    _alpha_learning_rate = params["alpha_learning_rate"]

    cql = d3rlpy.algos.CQLConfig(batch_size=_batch_size,
                                 gamma=_gamma,
                                 actor_learning_rate=_actor_learning_rate,
                                 critic_learning_rate=_critic_learning_rate,
                                 temp_learning_rate=_temp_learning_rate,
                                 alpha_learning_rate=_alpha_learning_rate,
                                 tau=_tau,
                                 n_critics=_n_critics,
                                 initial_alpha=_initial_alpha,
                                 initial_temperature=_initial_temperature,
                                 alpha_threshold=_alpha_threshold,
                                 conservative_weight=_conservative_weight,
                                 n_action_samples=_n_action_samples).create(device=_device)

    return cql

def createSAC(params):
    # parameters for algorithm
    _device = params['device']
    _batch_size = params['batch_size']
    _gamma = params["gamma"]
    _initial_temperature = params["initial_temperature"]
    _n_critics = params["n_critics"]
    _tau = params["tau"]
    _actor_learning_rate = params["actor_learning_rate"]
    _critic_learning_rate = params["critic_learning_rate"]
    _temp_learning_rate = params["temp_learning_rate"]

    sac = d3rlpy.algos.SACConfig(batch_size=_batch_size,
                                 gamma=_gamma,
                                 actor_learning_rate=_actor_learning_rate,
                                 critic_learning_rate=_critic_learning_rate,
                                 temp_learning_rate=_temp_learning_rate,
                                 tau=_tau,
                                 n_critics=_n_critics,
                                 initial_temperature=_initial_temperature
                                 ).create(device=_device)

    return sac

def createBCQ(params):
    # parameters for algorithm
    _batch_size = params['batch_size']
    _update_actor_interval = params["update_actor_interval"]
    _n_action_samples = params["n_action_samples"]
    _n_critics = params["n_critics"]
    _tau = params["tau"]
    _gamma = params["gamma"]
    _lam = params["lam"]
    _beta = params["beta"]
    _actor_learning_rate = params["actor_learning_rate"]
    _critic_learning_rate = params["critic_learning_rate"]
    _imitator_learning_rate = params["imitator_learning_rate"]
    _action_flexibility = params["action_flexibility"]
    _rl_start_step = params["rl_start_step"]
    _device = params['device']

    bcq = d3rlpy.algos.BCQConfig(batch_size=_batch_size,
                                 gamma=_gamma,
                                 actor_learning_rate=_actor_learning_rate,
                                 critic_learning_rate=_critic_learning_rate,
                                 imitator_learning_rate=_imitator_learning_rate,
                                 tau=_tau,
                                 n_critics=_n_critics,
                                 update_actor_interval=_update_actor_interval,
                                 lam=_lam,
                                 n_action_samples=_n_action_samples,
                                 action_flexibility=_action_flexibility,
                                 beta=_beta,
                                 rl_start_step=_rl_start_step).create(device=_device)

    return bcq

def createDT(params):
    # parameters for algorithm
    _batch_size = params["batch_size"]
    _context_size = params["context_size"]
    _gamma = params["gamma"]
    _max_timestep = params["max_timestep"]
    _num_heads = params["num_heads"]
    _num_layers = params["num_layers"]
    _attn_dropout = params["attn_dropout"]
    _resid_dropout = params["resid_dropout"]
    _embed_dropout = params["embed_dropout"]
    _activation_type = params["activation_type"]
    _warmup_steps = params["warmup_steps"]
    _clip_grad_norm = params["clip_grad_norm"]
    _learning_rate = params['learning_rate']
    _device = params['device']

    dt = d3rlpy.algos.DecisionTransformerConfig(batch_size=_batch_size,
                                                 gamma=_gamma,
                                                 context_size=_context_size,
                                                 max_timestep=_max_timestep,
                                                 learning_rate=_learning_rate,
                                                 num_heads=_num_heads,
                                                 num_layers=_num_layers,
                                                 attn_dropout=_attn_dropout,
                                                 resid_dropout=_resid_dropout,
                                                 embed_dropout=_embed_dropout,
                                                 activation_type=_activation_type,
                                                 warmup_steps=_warmup_steps,
                                                 clip_grad_norm=_clip_grad_norm).create(device=_device)

    return dt