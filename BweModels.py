import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import d3rlpy
import numpy as np
import onnxruntime as ort
import os
import math
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from d3rlpy.models.encoders import register_encoder_factory
from BweReward import RewardFunction
from BweUtils import load_train_data, load_multiple_files, load_train_data_from_file
from BweLogger import BweAdapterFactory
from BweEncoder import LSTMEncoderFactory, ACEncoderFactory


class BweDrl:
    def __init__(self, params, algo: d3rlpy.algos.qlearning.QLearningAlgoBase):
        self._params = params
        self._log_dir = params['log_folder_name']
        self._train_data_dir = params['train_data_folder']
        self._test_data_dir = params['test_data_folder']
        self._train_on_max_files = params['train_on_max_files']
        self._output_model_name = params['output_model_name']
        self._batch_size = params['batch_size']
        self._n_steps_per_epoch = params['n_steps_per_epoch']
        self._dataset_coverage = params['dataset_coverage']
        self._algo = algo
        self._algo_name = params['algorithm_name']
        self._reward_func = RewardFunction(params['reward_func_name'])
        self._device = params['device']
        self._ddp = params['ddp']
        self._rank = params['rank']
        self._world_size = params['world_size']
        # register your own encoder factory
        register_encoder_factory(ACEncoderFactory)
        register_encoder_factory(LSTMEncoderFactory)


    def get_algo_name(self) -> str:
        return self._algo_name


    def train_model_gradually(self, evaluator: bool):

        datafiles = load_multiple_files(self._train_data_dir, self._train_on_max_files)

        if self._rank == 0:
            # name of log folder
            start_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            self._log_dir = self._log_dir + "_" + start_date
            print(f"Worker {self._rank} logging folder {self._log_dir} will be created.")

        for filename in datafiles:
            print(f"Worker {self._rank} starts creating MDP dataset...")
            t1 = time.process_time()
            observations, actions, rewards, terminals = load_train_data_from_file(filename)
            obsScaler = MinMaxScaler()
            observations = obsScaler.fit_transform(observations)
            # calculate rewards if needed
            if not rewards:
                rewards = np.array([self._reward_func(o) for o in observations])
                rewards = np.append(rewards[1:], 0)

            start = 0
            end = len(actions)
            # divide dataset
            if self._world_size > 1:
                num_transitions = end
                num_transitions_per_worker = num_transitions // self._world_size
                start = self._rank * num_transitions_per_worker
                end = (self._rank + 1) * num_transitions_per_worker

            terminals[end - 1] = 1
            dataset = d3rlpy.dataset.MDPDataset(
                observations=observations[start:end],
                actions=actions[start:end],
                rewards=rewards[start:end],
                terminals=terminals[start:end],
                action_space=d3rlpy.ActionSpace.CONTINUOUS
            )

            t2 = time.process_time()
            print(f"Worker {self._rank} finishes with creating MDP dataset - {t2-t1} s.")

            n_steps = math.floor(dataset.transition_count*self._dataset_coverage // self._batch_size)
            n_steps = min(n_steps, 10000)
            print(f"Worker {self._rank} train {n_steps} steps, {self._n_steps_per_epoch} steps per epoch for {dataset.transition_count} records")

            test_episodes = dataset.episodes[:1]
            if self._rank == 0:
                # offline training
                if evaluator:
                    self._algo.fit(
                        dataset,
                        n_steps=n_steps,
                        n_steps_per_epoch=self._n_steps_per_epoch,
#                        experiment_name=f"experiment_{start_date}",
                        with_timestamp=False,
                        logger_adapter=BweAdapterFactory(root_dir=self._log_dir, output_model_name=self._output_model_name),
                        evaluators={
                            'td_error': d3rlpy.metrics.TDErrorEvaluator(test_episodes),
                            'discounted_advantage': d3rlpy.metrics.evaluators.DiscountedSumOfAdvantageEvaluator(test_episodes),
                            'average_value': d3rlpy.metrics.evaluators.AverageValueEstimationEvaluator(test_episodes),
                            'action_diff': d3rlpy.metrics.evaluators.ContinuousActionDiffEvaluator(test_episodes),
                        },
                        save_interval=10,
                        enable_ddp=self._ddp,
                    )
                else:
                    self._algo.fit(
                        dataset,
                        n_steps=n_steps,
                        n_steps_per_epoch=self._n_steps_per_epoch,
#                        experiment_name=f"experiment_{start_date}",
                        with_timestamp=False,
                        logger_adapter=BweAdapterFactory(root_dir=self._log_dir,
                                                         output_model_name=self._output_model_name),
                        save_interval=10,
                        enable_ddp=self._ddp,
                    )

            else:
                # offline training
                self._algo.fit(
                    dataset,
                    n_steps=n_steps,
                    n_steps_per_epoch=self._n_steps_per_epoch,
                    with_timestamp=False,
                    logger_adapter=d3rlpy.logging.NoopAdapterFactory(),
                    evaluators={},
                    enable_ddp=self._ddp,
                )

            t3 = time.process_time()
            print(f'Worker {self._rank} training time: {t3-t2} s')

#            print(f"Worker {self._rank} saves the trained model. Train data file {filename}")
#            policy_file_name = self._log_dir + '/' + self._output_model_name + '.onnx'
#            self._algo.save_policy(policy_file_name)

    def train_model(self):
        maybe_dataset_path = f"datasets/dataset_{self._train_on_max_files}.h5"
        if os.path.exists(maybe_dataset_path):
            print(f"Loading the MDP dataset: {maybe_dataset_path}")
            dataset = self.load_mdp_dataset(maybe_dataset_path)
        else:
            print("Creating the MDP dataset")
            dataset = self.create_mdp_dataset(save_name=f"dataset_{self._train_on_max_files}.h5")
        n_steps = dataset.transition_count
        n_steps_per_epoch = min(n_steps, 1000)
        n_episodes = dataset.size()
        print(f"Training on {n_steps} steps, {n_steps_per_epoch} steps per epoch for {n_episodes} episodes")

        start_date = datetime.now().strftime("%Y%m%d%H%M%S")
        self._log_dir = self._log_dir + "_" + start_date
        print("The latest trained model is placed under the log folder " + self._log_dir)

        test_episodes = dataset[:3]
        # offline training
        self._algo.fit(
            dataset,
            n_steps=n_steps,
            n_steps_per_epoch=n_steps_per_epoch,
            experiment_name=f"experiment_{start_date}",
            with_timestamp=False,
            logger_adapter=BweAdapterFactory(root_dir=self._log_dir, output_model_name=self._output_model_name),
            evaluators={
                'td_error': d3rlpy.metrics.TDErrorEvaluator(test_episodes),
                'discounted_advantage': d3rlpy.metrics.evaluators.DiscountedSumOfAdvantageEvaluator(test_episodes),
                'average_value': d3rlpy.metrics.evaluators.AverageValueEstimationEvaluator(test_episodes),
                'soft_opc': d3rlpy.metrics.evaluators.SoftOPCEvaluator(2),
                'action_diff': d3rlpy.metrics.evaluators.ContinuousActionDiffEvaluator(test_episodes),
            },
            save_interval=10,
            enable_ddp=self._ddp,
        )

#        policy_file_name = self._log_dir + '/' + self._output_model_name + '.onnx'
#        self._algo.save_policy(policy_file_name)

    def create_mdp_dataset(self, save_name: str | None = None) -> d3rlpy.dataset.MDPDataset:
        files = sorted(os.listdir(self._train_data_dir))
        train_data_files = []
        for name in files:
            f = os.path.join(self._train_data_dir, name)
            if os.path.isfile(f):
                train_data_files.append(f)
                if self._train_on_max_files > 0 and len(train_data_files) == self._train_on_max_files:
                    break
        print(f"Files to load: {len(train_data_files)}")

        # fill the MDP dataset
        observations = []
        actions = []
        rewards = []
        terminals = []
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self._process_file, filename) for filename in train_data_files]
            for future in tqdm(as_completed(futures), desc="Loading MDP", unit="file"):
                result = future.result()
                observations_file, actions_file, rewards_file, terminals_file = result
                observations.append(observations_file)
                actions.append(actions_file)
                rewards.append(rewards_file)
                terminals.append(terminals_file)
        observations = np.concatenate(observations)
        actions = np.concatenate(actions)
        rewards = np.concatenate(rewards)
        terminals = np.concatenate(terminals)

        # create the offline learning dataset
        dataset = d3rlpy.dataset.MDPDataset(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            action_space=d3rlpy.ActionSpace.CONTINUOUS,
        )
        print("MDP dataset is created")

        if save_name is not None:
            os.makedirs("datasets", exist_ok=True)
            dataset_path = os.path.join("datasets", f"{save_name}")
            with open(dataset_path, "w+b") as f:
                print(f"Saving the MDP dataset to {dataset_path}")
                dataset.dump(f)

        return dataset

    def load_mdp_dataset(self, filepath: str) -> d3rlpy.dataset.ReplayBuffer:
        with open(filepath, "r+b") as f:
            return d3rlpy.dataset.ReplayBuffer(
                d3rlpy.dataset.InfiniteBuffer(),
                episodes=d3rlpy.dataset.io.load(d3rlpy.dataset.components.Episode, f),
                action_space=d3rlpy.ActionSpace.CONTINUOUS,
            )

    def _process_file(self, filename: str) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        observations_file, actions_file, _, _ = load_train_data(filename)
        assert len(observations_file) > 0, f"File {filename} is empty"
        rewards_file = np.array([self._reward_func(o) for o in observations_file])
        terminals_file = np.zeros(len(observations_file))
        terminals_file[-1] = 1.0
        return observations_file, actions_file, rewards_file, terminals_file

    def export_policy(self):
        if os.path.exists(os.path.join(self._log_dir, f"{self._output_model_name}.d3")):
            output_model_full_name = self._log_dir + '/' + self._output_model_name + '.onnx'
            self._algo.save_policy(output_model_full_name)
            print("The model is exported as " + output_model_full_name)

    def evaluate_model_offline(self):
        # setup algorithm manually
        # if there is already a pre-trained model, load it
        if os.path.exists(os.path.join(self._log_dir, f"{self._output_model_name}.d3")):
            output_model_full_name = self._log_dir + '/' + self._output_model_name + '.d3'
            algo = d3rlpy.load_learnable(output_model_full_name, device=self._device)
            print(
                "Load the pre-trained model from the file"
                + output_model_full_name
                + " for evaluation with emulated data!"
            )
        else:
            print("There is no pre-trained model for evaluation!\n")
            return

        # to be used for online learning (or evaluation)
        files = sorted(os.listdir(self._test_data_dir))
        test_data_files = []
        for name in files:
            f = os.path.join(self._train_data_dir, name)
            # checking if it is a file
            if os.path.isfile(f):
                test_data_files.append(f)

        #        test_data_files = test_data_files[0:10]
        # predict
        for file in test_data_files:
            observations, actions, _, _ = load_train_data(file)
            predictions_diff = []
            for observation, action in zip(observations, actions):
                # Add batch dimension
                # observation = observation.reshape((1, len(observation)))
                # tt = np.expand_dims(observation, axis=0)
                observation = observation.reshape(1, len(observation))
                prediction = algo.predict(observation)[0]
                value = prediction - action
                filename = file.split('/')[1].split('.')[0]
                predictions_diff.append(f"{filename},{value[0]}\n")

            # logging the difference of predictions to logfile preddiff.csv
            path = os.path.join(self._log_dir, f"preddiff.csv")
            with open(path, "a") as f:
                f.writelines(predictions_diff)
                f.close()

    # INCOMPLETE
    def evaluatePolicy(self, policyFileName):
        # load ONNX policy via onnxruntime
        ort_session = ort.InferenceSession(policyFileName, providers=["CPUExecutionProvider"])

        # to obtain observations from the dataset
        observation = []
        # add batch dimension
        observation = observation.reshape((1, len(observation))).astype(np.float32)
        # returns greedy action
        action = ort_session.run(None, {'input_0': observation})
        print(action)


def createCQL(params):
    # parameters for algorithm
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

    ac_encoder_factory = d3rlpy.models.encoders.VectorEncoderFactory(hidden_units=[256,256,64,32])
    cql = d3rlpy.algos.CQLConfig(
        batch_size=_batch_size,
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
        n_action_samples=_n_action_samples,
        actor_encoder_factory=ac_encoder_factory,
        critic_encoder_factory=ac_encoder_factory,
#        observation_scaler=d3rlpy.preprocessing.StandardObservationScaler(),
        action_scaler=d3rlpy.preprocessing.MinMaxActionScaler(),
        reward_scaler=d3rlpy.preprocessing.MinMaxRewardScaler(),
        q_func_factory=d3rlpy.models.q_functions.QRQFunctionFactory(n_quantiles=32),
    ).create(device=params['device'])

    return cql


def createSAC(params):
    # parameters for algorithm
    _batch_size = params['batch_size']
    _gamma = params["gamma"]
    _initial_temperature = params["initial_temperature"]
    _n_critics = params["n_critics"]
    _tau = params["tau"]
    _actor_learning_rate = params["actor_learning_rate"]
    _critic_learning_rate = params["critic_learning_rate"]
    _temp_learning_rate = params["temp_learning_rate"]

    lstm_encoder_factory = LSTMEncoderFactory(1)

    sac = d3rlpy.algos.SACConfig(
        batch_size=_batch_size,
        gamma=_gamma,
        actor_learning_rate=_actor_learning_rate,
        critic_learning_rate=_critic_learning_rate,
        temp_learning_rate=_temp_learning_rate,
        tau=_tau,
        n_critics=_n_critics,
        initial_temperature=_initial_temperature,
        actor_encoder_factory=lstm_encoder_factory,
        critic_encoder_factory=lstm_encoder_factory,
#        observation_scaler=d3rlpy.preprocessing.StandardObservationScaler(),
        action_scaler=d3rlpy.preprocessing.MinMaxActionScaler(),
        reward_scaler=d3rlpy.preprocessing.MinMaxRewardScaler()
    ).create(device=params['device'])

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

    ac_encoder_factory = d3rlpy.models.encoders.VectorEncoderFactory(hidden_units=[256,256,64,32])
    
    bcq = d3rlpy.algos.BCQConfig(
        batch_size=_batch_size,
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
        rl_start_step=_rl_start_step,
        actor_encoder_factory=ac_encoder_factory,
        critic_encoder_factory=ac_encoder_factory,
#        observation_scaler=d3rlpy.preprocessing.StandardObservationScaler(),
        action_scaler=d3rlpy.preprocessing.MinMaxActionScaler(),
        reward_scaler=d3rlpy.preprocessing.MinMaxRewardScaler()
    ).create(device=params['device'])

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

    dt = d3rlpy.algos.DecisionTransformerConfig(
        batch_size=_batch_size,
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
        clip_grad_norm=_clip_grad_norm,
#        observation_scaler=d3rlpy.preprocessing.StandardObservationScaler(),
        action_scaler=d3rlpy.preprocessing.MinMaxActionScaler(),
        reward_scaler=d3rlpy.preprocessing.MinMaxRewardScaler()
    ).create(device=params['device'])

    return dt
