from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import d3rlpy
from d3rlpy.models.encoders import register_encoder_factory
import numpy as np
import onnxruntime as ort
import os
from tqdm import tqdm

from BweReward import RewardFunction
from BweUtils import load_train_data
from BweLogger import BweAdapterFactory
from BweEncoder import LSTMEncoderFactory


class BweDrl:
    def __init__(self, params, algo: d3rlpy.base.LearnableBase):
        self._log_dir = params['logFolderName']
        self._train_data_dir = params['trainDataFolder']
        self._test_data_dir = params['testDataFolder']
        self._train_on_max_files = params['trainOnMaxFiles']
        self._output_model_name = params['outputModelName']
        self._algo = algo
        self._reward_func = RewardFunction(params['rewardFuncName'])
        self._device = params['device']
        self._ddp = params['ddp']

        # register your own encoder factory
        register_encoder_factory(LSTMEncoderFactory)

    def train_model_gradually(self):
        # load the list of log files under the given directory
        # iterate over files in that directory
        files = sorted(os.listdir(self._train_data_dir))
        train_data_files = []
        for name in files:
            f = os.path.join(self._train_data_dir, name)
            # checking if it is a file
            if os.path.isfile(f):
                train_data_files.append(f)
                if self._train_on_max_files > 0 and len(train_data_files) == self._train_on_max_files:
                    break
        print(f"Files to load: {len(train_data_files)}")

        # create log folder
        start_date = datetime.now().strftime("%Y%m%d%H%M%S")
        self._log_dir = self._log_dir + "_" + start_date
        print(f"Logging folder {self._log_dir} is created.")

        for filename in train_data_files:
            # load the file (.npz), fill the MDP dataset
            print(f"Load file {filename}.")
            loaded = np.load(filename, 'rb')
            observations = np.array(loaded['obs'])
            actions = np.array(loaded['acts'])
            terminals = np.array(loaded['terms'])
            rewards = np.array([self._reward_func(o) for o in observations])

            # create the offline learning dataset
            dataset = d3rlpy.dataset.MDPDataset(
                observations=observations,
                actions=actions,
                rewards=rewards,
                terminals=terminals,
                action_space=d3rlpy.ActionSpace.CONTINUOUS,
            )
            print("MDP dataset is created")

            n_steps = len(observations)
            # FIXME: tune it? 10000 is the default value for all Q-learning algorithms but maybe it is too big?
            n_steps_per_epoch = min(n_steps, 10000)
            print(f"Training on {n_steps} steps, {n_steps_per_epoch} steps per epoch for {dataset.size()} episodes")

            # offline training
            self._algo.fit(
                dataset,
                n_steps=n_steps,
                n_steps_per_epoch=n_steps_per_epoch,
                experiment_name=f"experiment_{start_date}",
                with_timestamp=False,
                logger_adapter=BweAdapterFactory(root_dir=self._log_dir, output_model_name=self._output_model_name),
                enable_ddp=self._ddp,
            )

            print(f"Saving the trained model.")
            policy_file_name = self._log_dir + '/' + self._output_model_name + '.onnx'
            self._algo.save_policy(policy_file_name)
            model_file_name = self._log_dir + '/' + self._output_model_name + '.d3'
            self._algo.save_model(model_file_name)


        print("The latest trained model is placed under the log folder " + self._log_dir)
        print("Training completed.\n")

    def train_model(self):
        # load the list of log files under the given directory
        # iterate over files in that directory
        files = sorted(os.listdir(self._train_data_dir))
        train_data_files = []
        for name in files:
            f = os.path.join(self._train_data_dir, name)
            # checking if it is a file
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
        timeouts = []
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
            #            timeouts=timeouts,
            action_space=d3rlpy.ActionSpace.CONTINUOUS,
        )
        print("MDP dataset is created")

        n_steps = len(observations)
        # FIXME: tune it? 10000 is the default value for all Q-learning algorithms but maybe it is too big?
        n_steps_per_epoch = min(n_steps, 10000)
        print(f"Training on {n_steps} steps, {n_steps_per_epoch} steps per epoch for {dataset.size()} episodes")

        start_date = datetime.now().strftime("%Y%m%d%H%M%S")
        self._log_dir = self._log_dir + "_" + start_date
        print("The latest trained model is placed under the log folder " + self._log_dir)

        # offline training
        self._algo.fit(
            dataset,
            n_steps=n_steps,
            n_steps_per_epoch=n_steps_per_epoch,
            experiment_name=f"experiment_{start_date}",
            with_timestamp=False,
            logger_adapter=BweAdapterFactory(root_dir=self._log_dir, output_model_name=self._output_model_name),
            enable_ddp=self._ddp,
        )

        policy_file_name = self._log_dir + '/' + self._output_model_name + '.onnx'
        self._algo.save_policy(policy_file_name)

    # FIXME: if I understood correctly, you can actually train once on all the files, it will internally sseparated into the episodes,
    # if one gives the proper terminals/timeouts, please validate it (Nikita)
    def _process_file(self, filename: str) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        # load the log file and prepare the dataset
        observations_file, actions_file, _, _ = load_train_data(filename)
        assert len(observations_file) > 0, f"File {filename} is empty"
        # calculate reward
        rewards_file = np.array([self._reward_func(o) for o in observations_file])
        # terminals are not used so they should be non 1.0
        #        terminals_file = np.zeros(len(observations_file))
        terminals_file = np.random.randint(2, size=len(observations_file))
        # timeout at the end of the file
        # timeouts_file = np.zeros(len(observations_file))
        # timeouts_file[-1] = 1.0
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
    ).create(device=params['device'])

    return dt
