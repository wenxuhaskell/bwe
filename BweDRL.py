from datetime import datetime
import d3rlpy
import numpy as np

import os
import BweReward
from BweUtils import load_data
from BweLogger import BweAdapter, BweAdapterFactory
import onnxruntime as ort

class BweCQL:
    def __init__(self, params):
        self._output_model_name = params['outputModelName']
        self._log_dir_pre = params['logFolderName']
        self._train_data_dir = params['trainDataFolder']
        self._test_data_dir = params['testDataFolder']
        self._device = params['device']
        rf_name = params['rewardFuncName']
        self._reward_func = getattr(BweReward, rf_name)

        self._batch_size = params['batch_size']
        self._initial_alpha = params["initial_alpha"]
        self._gamma = params["gamma"]
        self._alpha_threshold = params["alpha_threshold"]
        self._conservative_weight = params["conservative_weight"]
        self._initial_temperature = params["initial_temperature"]
        self._n_action_samples = params["n_action_samples"]
        self._n_critics = params["n_critics"]
        self._tau = params["tau"]
        self._actor_learning_rate = params["actor_learning_rate"]
        self._critic_learning_rate = params["critic_learning_rate"]
        self._temp_learning_rate = params["temp_learning_rate"]
        self._alpha_learning_rate = params["alpha_learning_rate"]
        # for storing output models and metrics
        self._log_dir = self._log_dir_pre


    # arguments:
    #  log_filename - fullname including path of training data file
    # return:
    #  success: True/False
    def train_episode (self, log_filename = ''):
        # flag for successful/unsuccessful traning of one episode
        success = False

        # if the log_filename is not given
        if(log_filename == ''):
            print("train_episode() - Please provide a valid log filename! \n")
            return ('', success)

        # load the log file and prepare the dataset
        observations, actions = load_data(log_filename)
        # terminal flags
        terminals = np.random.randint(2, size=len(actions))
        #terminals[-1] = 1
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

        # if there is already a pre-trained model, load it
        if os.path.exists(os.path.join(self._log_dir, f"{self._output_model_name}.d3")):
            output_model_full_name = self._log_dir + '/' + self._output_model_name + '.d3'
            cql = d3rlpy.load_learnable(output_model_full_name, device=self._device)
        else:
            cql = d3rlpy.algos.CQLConfig(batch_size=self._batch_size,
                                         gamma=self._gamma,
                                         actor_learning_rate=self._actor_learning_rate,
                                         critic_learning_rate=self._critic_learning_rate,
                                         temp_learning_rate=self._temp_learning_rate,
                                         alpha_learning_rate=self._actor_learning_rate,
                                         tau=self._tau,
                                         n_critics=self._n_critics,
                                         initial_temperature=self._initial_temperature,
                                         initial_alpha=self._initial_alpha,
                                         alpha_threshold=self._alpha_threshold,
                                         conservative_weight=self._conservative_weight,
                                         n_action_samples=self._n_action_samples).create(device=self._device)

        # remove path and extension
        log_filename = log_filename.split('/')[1].split('.')[0]
        n_steps = len(observations)
        n_steps_per_epoch = n_steps
        # offline training
        cql.fit(
            dataset,
            n_steps,
            n_steps_per_epoch,
            experiment_name=log_filename,
            with_timestamp=False,
            logger_adapter=BweAdapterFactory(root_dir=self._log_dir, output_model_name=self._output_model_name),
        )
        # export onnx model
#        policy_file = output_model_full_name.split('.'),[1] + '.onnx'
#        cql.save_policy(policy_file)
        success = True
        return success

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
        self._log_dir = self._log_dir_pre + "_" + start_date
        # keep for debugging purpose
#        datafiles = train_data_files[0:10]
        datafiles = train_data_files
        for file in datafiles:
            success = self.train_episode(file)
            if success == False:
                break
        #
        print("Training of " + str(len(datafiles)) + " episodes!\n")
        print("The latest trained model is placed under the log folder")
        return success

class BweSAC:
    def __init__(self, params):
        self._output_model_name = params['outputModelName']
        self._log_dir_pre = params['logFolderName']
        self._train_data_dir = params['trainDataFolder']
        self._test_data_dir = params['testDataFolder']
        self._device = params['device']
        rf_name = params['rewardFuncName']
        self._reward_func = getattr(BweReward, rf_name)

        self._batch_size = params['batch_size']
        self._gamma = params["gamma"]
        self._n_critics = params["n_critics"]
        self._initial_temperature = params["initial_temperature"]
        self._tau = params["tau"]
        self._actor_learning_rate = params["actor_learning_rate"]
        self._critic_learning_rate = params["critic_learning_rate"]
        self._temp_learning_rate = params["temp_learning_rate"]
        # for storing output models and metrics
        self._log_dir = self._log_dir_pre

    # arguments:
    #  log_filename - fullname including path of training data file
    # return:
    #  success: True/False
    def train_episode (self, log_filename = ''):
        # flag for successful/unsuccessful traning of one episode
        success = False

        # if the log_filename is not given
        if(log_filename == ''):
            print("train_episode() - Please provide a valid log filename! \n")
            return ('', success)

        # load the log file and prepare the dataset
        observations, actions = load_data(log_filename)
        # terminal flags
        terminals = np.random.randint(2, size=len(actions))
        #terminals[-1] = 1
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

        # if there is already a pre-trained model, load it
        if os.path.exists(os.path.join(self._log_dir, f"{self._output_model_name}.d3")):
            output_model_full_name = self._log_dir + '/' + self._output_model_name + '.d3'
            sac = d3rlpy.load_learnable(output_model_full_name, device=self._device)
        else:
            sac = d3rlpy.algos.SACConfig(batch_size=self._batch_size,
                                         gamma=self._gamma,
                                         actor_learning_rate=self._actor_learning_rate,
                                         critic_learning_rate=self._critic_learning_rate,
                                         temp_learning_rate=self._temp_learning_rate,
                                         tau=self._tau,
                                         n_critics=self._n_critics,
                                         initial_temperature=self._initial_temperature).create(device=self._device)

        # remove path and extension
        log_filename = log_filename.split('/')[1].split('.')[0]
        n_steps = len(observations)
        n_steps_per_epoch = n_steps
        # offline training
        sac.fit(
            dataset,
            n_steps,
            n_steps_per_epoch,
            experiment_name=log_filename,
            with_timestamp=False,
            logger_adapter=BweAdapterFactory(root_dir=self._log_dir, output_model_name=self._output_model_name),
        )
        # export onnx model
#        policy_file = output_model_full_name.split('.'),[1] + '.onnx'
#        sac.save_policy(policy_file)
        success = True
        return success

class BweBCQ:
    def __init__(self, params, ):
        self._output_model_name = params['outputModelName']
        self._log_dir_pre = params['logFolderName']
        self._train_data_dir = params['trainDataFolder']
        self._test_data_dir = params['testDataFolder']
        self._device = params['device']
        rf_name = params['rewardFuncName']
        self._reward_func = getattr(BweReward, rf_name)

        self._batch_size = params['batch_size']
        self._update_actor_interval = params["update_actor_interval"]
        self._n_action_samples = params["n_action_samples"]
        self._n_critics = params["n_critics"]
        self._tau = params["tau"]
        self._gamma = params["gamma"]
        self._lam = params["lam"]
        self._beta = params["beta"]
        self._actor_learning_rate = params["actor_learning_rate"]
        self._critic_learning_rate = params["critic_learning_rate"]
        self._imitator_learning_rate = params["imitator_learning_rate"]
        self._action_flexibility = params["action_flexibility"]
        self._rl_start_step = params["rl_start_step"]
        # for storing output models and metrics
        self._log_dir = self._log_dir_pre

    # arguments:
    #  log_filename - fullname including path of training data file
    # return:
    #  success: True/False
    def train_episode (self, log_filename = ''):
        # flag for successful/unsuccessful traning of one episode
        success = False

        # if the log_filename is not given
        if(log_filename == ''):
            print("train_episode() - Please provide a valid log filename! \n")
            return ('', success)

        # load the log file and prepare the dataset
        observations, actions = load_data(log_filename)
        # terminal flags
        terminals = np.random.randint(2, size=len(actions))
        #terminals[-1] = 1
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

        # if there is already a pre-trained model, load it
        if os.path.exists(os.path.join(self._log_dir, f"{self._output_model_name}.d3")):
            output_model_full_name = self._log_dir + '/' + self._output_model_name + '.d3'
            bcq = d3rlpy.load_learnable(output_model_full_name, device=self._device)
        else:
            bcq = d3rlpy.algos.BCQConfig(batch_size=self._batch_size,
                                         gamma=self._gamma,
                                         actor_learning_rate=self._actor_learning_rate,
                                         critic_learning_rate=self._critic_learning_rate,
                                         imitator_learning_rate=self._imitator_learning_rate,
                                         tau=self._tau,
                                         n_critics=self._n_critics,
                                         update_actor_interval=self._update_actor_interval,
                                         lam=self._lam,
                                         n_action_samples=self._n_action_samples,
                                         action_flexibility=self._action_flexibility,
                                         beta=self._beta,
                                         rl_start_step=self._rl_start_step).create(device=self._device)

        # remove path and extension
        log_filename = log_filename.split('/')[1].split('.')[0]
        n_steps = len(observations)
        n_steps_per_epoch = n_steps
        # offline training
        bcq.fit(
            dataset,
            n_steps,
            n_steps_per_epoch,
            experiment_name=log_filename,
            with_timestamp=False,
            logger_adapter=BweAdapterFactory(root_dir=self._log_dir, output_model_name=self._output_model_name),
        )
        # export onnx model
#        policy_file = output_model_full_name.split('.'),[1] + '.onnx'
#        bcq.save_policy(policy_file)
        success = True
        return success

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
        self._log_dir = self._log_dir_pre + "_" + start_date
        datafiles = train_data_files[0:10]
        for file in datafiles:
            success = self.train_episode(file)
            if success == False:
                break
        #
        print("Training of " + str(len(datafiles)) + " episodes!\n")
        print("The latest trained model is placed under the log folder")
        return success

class BweDT:
    def __init__(self, params):
        self._output_model_name = params['outputModelName']
        self._log_dir_pre = params['logFolderName']
        self._train_data_dir = params['trainDataFolder']
        self._test_data_dir = params['testDataFolder']
        self._device = params['device']
        rf_name = params['rewardFuncName']
        self._reward_func = getattr(BweReward, rf_name)

        self._batch_size = params["batch_size"]
        self._context_size = params["context_size"]
        self._gamma = params["gamma"]
        self._max_timestep = params["max_timestep"]
        self._num_heads = params["num_heads"]
        self._num_layers = params["num_layers"]
        self._attn_dropout = params["attn_dropout"]
        self._resid_dropout = params["resid_dropout"]
        self._embed_dropout = params["embed_dropout"]
        self._activation_type = params["activation_type"]
        self._warmup_steps = params["warmup_steps"]
        self._clip_grad_norm = params["clip_grad_norm"]
        # for storing output models and metrics
        self._log_dir = self._log_dir_pre

    # arguments:
    #  log_filename - fullname including path of training data file
    # return:
    #  success: True/False
    def train_episode (self, log_filename = ''):
        # flag for successful/unsuccessful traning of one episode
        success = False

        # if the log_filename is not given
        if(log_filename == ''):
            print("train_episode() - Please provide a valid log filename! \n")
            return ('', success)

        # load the log file and prepare the dataset
        observations, actions = load_data(log_filename)
        print(observations.shape)
        print(actions.shape)
        # terminal flags
        terminals = np.random.randint(2, size=len(actions))
#        terminals[-1] = 1
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

        # if there is already a pre-trained model, load it
        if os.path.exists(os.path.join(self._log_dir, f"{self._output_model_name}.d3")):
            output_model_full_name = self._log_dir + '/' + self._output_model_name + '.d3'
            dt = d3rlpy.load_learnable(output_model_full_name, device=self._device)
        else:
            dt = d3rlpy.algos.DecisionTransformerConfig().create(device=self._device)

        # remove path and extension
        log_filename = log_filename.split('/')[1].split('.')[0]
        n_steps = len(observations)
        n_steps_per_epoch = n_steps
        # offline training
        dt.fit(
            dataset,
            n_steps,
            n_steps_per_epoch,
            experiment_name=log_filename,
            with_timestamp=False,
            logger_adapter=BweAdapterFactory(root_dir=self._log_dir, output_model_name=self._output_model_name),
        )

        # export as ONNX
#        policy_file = output_model_full_name.split('.'),[1] + '.onnx'
#        dt.save_policy(policy_file)
        success = True
        return success

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
        self._log_dir = self._log_dir_pre + "_" + start_date
        datafiles = train_data_files[0:10]
        for file in datafiles:
            success = self.train_episode(file)
            if success == False:
                break
        #
        print("Training of " + str(len(datafiles)) + " episodes!\n")
        print("The latest trained model is placed under the log folder")
        return success
