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
from BweReward import RewardFunction, Feature, MI, MIType, get_feature_for_mi

from BweUtils import load_train_data, load_multiple_files, load_train_data_from_file
from BweLogger import BweAdapterFactory
from BweEncoder import LSTMEncoderFactory, ACEncoderFactory

OBSERVATION_MIN = [    0.0,             0.0,             0.0,             0.0,
    0.0,             0.0,             0.0,             0.0,
    0.0,             0.0,             0.0,             0.0,
    0.0,             0.0,             0.0,             0.0,
    0.0,             0.0,             0.0,             0.0,
    0.0,             0.0,             0.0,             0.0,
    0.0,             0.0,             0.0,             0.0,
    0.0,             0.0,             0.0,             0.0,
    0.0,             0.0,             0.0,             0.0,
    0.0,             0.0,             0.0,             0.0,
    -2187.5,        -2186.0,         -2188.0,         -2188.0,
    -2188.0,         -2179.538461540, -2179.538461540, -2179.818181820,
    -2179.5,        -2179.538461540, -1992.0,         -1992.0,
    -1992.0,         -1992.0,         -1992.0,         -1992.0,
    -1992.0,         -1992.0,         -1992.0,         -1992.0,
    0.0,             0.0,             0.0,             0.0,
    0.0,             0.0,             0.0,             0.0,
    0.0,             0.0,             0.0,             0.0,
    0.0,             0.0,             0.0,             0.0,
    0.0,             0.0,             0.0,             0.0,
    0.0,             0.0,             0.0,             0.0,
    0.0,             0.0,             0.0,             0.0,
    0.0,             0.0,             0.0,             0.0,
    0.0,             0.0,             0.0,             0.0,
    0.0,             0.0,             0.0,             0.0,
    0.0,             0.0,             0.0,             0.0,
    0.0,             0.0,             0.0,             0.0,
    0.0,             0.0,             0.0,             0.0,
    0.0,             0.0,             0.0,             0.0,
    0.0,             0.0,             0.0,             0.0,
    0.0,             0.0,             0.0,             0.0,
    0.0,             0.0,             0.0,             0.0,
    0.0,             0.0,             0.0,             0.0,
    0.0,             0.0,             0.0,             0.0,
    0.0,             0.0,             0.0,             0.0,
    0.0,             0.0,             0.0,             0.0,
    0.0,             0.0,             0.0,             0.0,
    0.0,             0.0        ]
OBSERVATION_MAX = [2.29611867e+08, 2.29611867e+08, 2.29611867e+08, 2.29611867e+08,
     2.25766133e+08, 3.11890800e+07, 3.11692800e+07, 3.11692800e+07,
     3.11750267e+07, 3.12027333e+07, 1.74800000e+03, 1.74800000e+03,
     1.74800000e+03, 1.74800000e+03, 1.71900000e+03, 2.86700000e+03,
     2.86600000e+03, 2.86600000e+03, 2.86600000e+03, 2.86800000e+03,
     1.72208900e+06, 1.72208900e+06, 1.72208900e+06, 1.72208900e+06,
     1.69324600e+06, 2.33918100e+06, 2.33769600e+06, 2.33769600e+06,
     2.33812700e+06, 2.34020500e+06, 8.02900000e+03, 8.07168421e+03,
     8.07168421e+03, 7.81300000e+03, 7.67600000e+03, 8.02900000e+03,
     8.07168421e+03, 7.69900000e+03, 7.67600000e+03, 7.67821739e+03,
     8.02700000e+03, 8.06968421e+03, 8.06968421e+03, 7.78600000e+03,
     7.66500000e+03, 8.02700000e+03, 8.06968421e+03, 7.66500000e+03,
     7.66500000e+03, 7.66721739e+03, 2.00000000e+02, 2.00000000e+02,
     2.00000000e+02, 2.00000000e+02, 2.00000000e+02, 2.00000000e+02,
     2.00000000e+02, 2.00000000e+02, 2.00000000e+02, 2.00000000e+02,
     4.33642857e+02, 3.84840000e+02, 4.16592593e+02, 2.33040816e+02,
     4.48111111e+02, 5.52259681e+02, 6.04000000e+02, 8.30888889e+02,
     6.04000000e+02, 5.82626424e+02, 4.70979730e+03, 4.08230645e+03,
     4.70979730e+03, 4.70979730e+03, 4.33043019e+03, 6.00428238e+03,
     5.85539306e+03, 5.99811719e+03, 6.00122078e+03, 6.00122078e+03,
     7.72100000e+03, 7.72100000e+03, 7.72100000e+03, 7.72100000e+03,
     7.30600000e+03, 7.72100000e+03, 4.66200000e+03, 7.72100000e+03,
     7.72100000e+03, 7.72100000e+03, 5.64940320e+03, 6.11731487e+03,
     5.64940320e+03, 3.52382534e+03, 3.94350000e+03, 5.64940320e+03,
     5.64940320e+03, 6.66000312e+03, 6.66000312e+03, 3.83600000e+03,
     9.99444753e-01, 9.98940678e-01, 9.98676198e-01, 9.97765363e-01,
     9.99449642e-01, 9.99444753e-01, 9.98676198e-01, 9.99205298e-01,
     9.99205298e-01, 9.98684211e-01, 1.88600000e+03, 1.88600000e+03,
     1.88600000e+03, 1.88600000e+03, 1.88600000e+03, 1.88600000e+03,
     1.88600000e+03, 1.88600000e+03, 1.88600000e+03, 1.88600000e+03,
     1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
     1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
     1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
     1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
     1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
     1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
     1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
     1.00000000e+00, 1.00000000e+00]

OBSERVATION_MEAN = [1.93663500e+06, 1.77384042e+06,  1.78181603e+06,  1.74979287e+06,
    1.72435037e+06,  1.76951604e+06,  1.73935227e+06,  1.73589912e+06,
    1.72929810e+06,  1.72387474e+06,  1.61544534e+01,  1.47095189e+01,
    1.47754257e+01,  1.45324404e+01,  1.43607080e+01,  1.47059433e+02,
    1.44549616e+02,  1.44244884e+02,  1.43707974e+02,  1.43270752e+02,
    1.45247625e+04,  1.33038031e+04,  1.33636202e+04,  1.31234465e+04,
    1.29326278e+04,  1.32713703e+05,  1.30451420e+05,  1.30192434e+05,
    1.29697357e+05,  1.29290606e+05,  2.04223799e+01,  1.76300161e+01,
    1.84191965e+01,  1.78941613e+01,  1.81959718e+01,  2.14826214e+01,
    2.07620545e+01,  2.08947474e+01,  2.07176323e+01,  2.06223327e+01,
    -1.92081006e+01, -1.99729157e+01, -1.97849512e+01, -1.99318159e+01,
    -1.98816837e+01, -1.81478591e+01, -1.87686302e+01, -1.85616910e+01,
    -1.86214218e+01, -1.86028651e+01,  1.60369519e+02,  1.60303553e+02,
    1.60242532e+02,  1.60187716e+02,  1.60133362e+02,  1.60369519e+02,
    1.59775608e+02,  1.59233998e+02,  1.58701258e+02,  1.58174837e+02,
    1.02760114e+00,  1.00521993e+00,  1.00741903e+00,  1.00472545e+00,
    1.00590493e+00,  1.07405129e+00,  1.06662523e+00,  1.06350105e+00,
    1.05988562e+00,  1.05613214e+00,  6.34008768e+00,  5.30865053e+00,
    5.48488267e+00,  5.32288418e+00,  5.41449406e+00,  1.25301092e+01,
    1.19660472e+01,  1.19457845e+01,  1.18883736e+01,  1.18035440e+01,
    9.84065108e+00,  9.06462410e+00,  9.02335626e+00,  9.20722435e+00,
    9.10437985e+00,  8.70757505e+00,  8.61945855e+00,  8.63192269e+00,
    8.57910299e+00,  8.56538572e+00,  6.85616612e+00,  6.00467481e+00,
    6.18285085e+00,  6.02991850e+00,  6.16190274e+00,  8.53323748e+00,
    8.30901053e+00,  8.35234686e+00,  8.28807403e+00,  8.27705709e+00,
    8.67008032e-03,  7.56531210e-03,  7.67195618e-03,  7.63833693e-03,
    7.61576744e-03,  1.24066191e-02,  1.16975500e-02,  1.20921852e-02,
    1.18567405e-02,  1.21014736e-02,  9.22884499e-02,  8.01101027e-02,
    8.05330491e-02,  7.95441613e-02,  7.88345817e-02,  2.94028752e-01,
    2.74226241e-01,  2.82895939e-01,  2.75885749e-01,  2.83594372e-01,
    6.86789290e-01,  6.73967402e-01,  6.76804701e-01,  6.72995679e-01,
    6.68933251e-01,  7.08784204e-01,  7.05396258e-01,  7.03355478e-01,
    7.00646610e-01,  6.98403179e-01,  3.10089820e-01,  3.06521445e-01,
    3.04754509e-01,  3.07098354e-01,  3.11525934e-01,  2.87361389e-01,
    2.86836832e-01,  2.85849389e-01,  2.85247112e-01,  2.84306336e-01,
    3.11798045e-03,  2.83029366e-03,  2.80833850e-03,  2.83100888e-03,
    2.83922270e-03,  3.85091983e-03,  3.83702111e-03,  3.83882257e-03,
    3.83242903e-03,  3.80277944e-03 ]

OBSERVATION_STD = [ 1.46588941e+06, 1.35396780e+06, 1.36271677e+06, 1.35079649e+06,
    1.34028090e+06, 1.00956779e+06, 1.00169491e+06, 1.00680559e+06,
    1.00973528e+06, 1.01270744e+06, 1.04547330e+01, 9.59928206e+00,
    9.65098537e+00, 9.56300738e+00, 9.49768058e+00, 7.24484542e+01,
    7.21116201e+01, 7.25820201e+01, 7.29186173e+01, 7.32280977e+01,
    1.09941706e+04, 1.01547585e+04, 1.02203757e+04, 1.01309737e+04,
    1.00521067e+04, 7.57175840e+04, 7.51271179e+04, 7.55104195e+04,
    7.57301457e+04, 7.59530582e+04, 7.82575160e+01, 6.14665483e+01,
    6.23948153e+01, 6.15084154e+01, 6.21141153e+01, 8.11269903e+01,
    7.00810514e+01, 7.44516413e+01, 7.09572642e+01, 7.07288226e+01,
    8.91738606e+01, 7.41274444e+01, 7.47427291e+01, 7.41388138e+01,
    7.46566234e+01, 9.15601281e+01, 8.18743065e+01, 8.54048069e+01,
    8.22671143e+01, 8.19158124e+01, 6.26291495e+01, 6.26925612e+01,
    6.27555500e+01, 6.28128795e+01, 6.28702507e+01, 6.26291495e+01,
    6.32503689e+01, 6.38247351e+01, 6.43993928e+01, 6.49557211e+01,
    2.81957857e-01, 2.39369862e-01, 2.50549727e-01, 2.27919139e-01,
    2.55014288e-01, 7.29618478e-01, 6.98852353e-01, 7.05272289e-01,
    6.98834832e-01, 6.91473274e-01, 1.50253971e+01, 9.65744435e+00,
    9.85108575e+00, 9.36942235e+00, 9.35587219e+00, 3.47879642e+01,
    2.64465764e+01, 2.70419853e+01, 2.67716184e+01, 2.61368097e+01,
    1.77620336e+01, 1.43389516e+01, 1.39113219e+01, 1.46146362e+01,
    1.40062774e+01, 1.49854046e+01, 1.32333677e+01, 1.41129752e+01,
    1.35412775e+01, 1.36986533e+01, 8.61131852e+00, 5.49953909e+00,
    5.56964351e+00, 5.39956239e+00, 5.54583411e+00, 1.13871125e+01,
    8.73570797e+00, 9.80175187e+00, 9.11537323e+00, 8.88401182e+00,
    6.21616870e-02, 5.88193642e-02, 5.92090075e-02, 5.92268596e-02,
    5.91562715e-02, 5.64244438e-02, 5.49164749e-02, 5.60001514e-02,
    5.55122214e-02, 5.59186441e-02, 1.42131108e+00, 1.28729170e+00,
    1.30655077e+00, 1.27927366e+00, 1.29794311e+00, 2.62500676e+00,
    2.56219221e+00, 2.57543585e+00, 2.55304079e+00, 2.55515906e+00,
    2.88065833e-01, 3.08445793e-01, 3.03559454e-01, 3.06735707e-01,
    3.05860025e-01, 2.55874175e-01, 2.59746379e-01, 2.62365488e-01,
    2.65487511e-01, 2.68065214e-01, 2.84847474e-01, 2.95694714e-01,
    2.91137813e-01, 2.93711906e-01, 2.93304053e-01, 2.50782002e-01,
    2.51515555e-01, 2.51734091e-01, 2.52313232e-01, 2.52483261e-01,
    3.94003935e-02, 3.90592867e-02, 3.87900218e-02, 3.90514751e-02,
    3.90804436e-02, 2.14171127e-02, 2.16734777e-02, 2.18111491e-02,
    2.17950330e-02, 2.17156514e-02 ]

ACTION_MAX = 61662000.0
ACTION_MIN = 0.0
REWARD_MAX = 5.0
REWARD_MIN = 0.0

class BweDrl:
    def __init__(self, params):
        self._algo = None
        self._params = params
        self._log_dir = params['log_folder_name']
        self._train_data_dir = params['train_data_folder']
        self._test_data_dir = params['test_data_folder']
        self._train_on_max_files = params['train_on_max_files']
        self._output_model_name = params['output_model_name']
        self._batch_size = params['batch_size']
        self._n_steps_per_epoch = params['n_steps_per_epoch']
        self._dataset_coverage = params['dataset_coverage']
        self._algo_name = params['algorithm_name']
        if self._algo_name.upper() != 'DT':
            self._evaluator = params['evaluator']
            self._finetune = params['finetune']
        else:
            self._evaluator = None
            self._finetune = None
        self._reward_func = RewardFunction(params['reward_func_name'])
        self._device = params['device']
        self._ddp = params['ddp']
        self._rank = params['rank']
        self._world_size = params['world_size']
        # register your own encoder factory
        register_encoder_factory(ACEncoderFactory)
        register_encoder_factory(LSTMEncoderFactory)

    # create the model
    def create_model(self, params=None):
        if not params:
            self._params = params

        if self._params['algorithm_name'] == 'CQL':
            self._algo = createCQL(self._params)
        elif self._params['algorithm_name'] == "SAC":
            self._algo = createSAC(self._params)
        elif self._params['algorithm_name'] == "BCQ":
            self._algo = createBCQ(self._params)
        elif self._params['algorithm_name'] == "DDPG":
            self._algo = createDDPG(self._params)
        elif self._params['algorithm_name'] == "DT":
            self._algo = createDT(self._params)
            # Decision Transformer does not support evaluator
            print("Decision transformer does not support evaluator")
            self._evaluator = False
        else:
            print("Please provide a configuration file with a valid algorithm name!\n")
            return


    # retrieve the name of algorithms
    def get_algo_name(self) -> str:
        return self._algo_name


    # retieve the list of data files
    def get_list_data_files(self) -> list:
        return load_multiple_files(self._train_data_dir, self._train_on_max_files)


    # create MDP dataset for current worker
    def load_MDP_dataset(self, filename) -> d3rlpy.dataset.MDPDataset:
        observations, actions, rewards, terminals = load_train_data_from_file(filename)
        # calculate rewards if needed
        if not rewards:
            rewards = np.array([self._reward_func(o) for o in observations])
            r_last = rewards[-1]
            rewards = np.append(rewards[1:], r_last)

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

        return dataset

    # training using a given MDP dataset
    def train(self, dataset: d3rlpy.dataset.MDPDataset, evaluators):
        n_steps = math.floor(dataset.transition_count * self._dataset_coverage // self._batch_size)
        n_steps = min(n_steps, 10000)
        print(
            f"Worker {self._rank} train {n_steps} steps, {self._n_steps_per_epoch} steps per epoch for {dataset.transition_count} records")

        t2 = time.process_time()
        test_episodes = dataset.episodes[:1]
        if self._rank == 0:
            # offline training for work of rank = 0, logging is enabled
            # offline training with evaluators
            save_interval = min(n_steps // self._n_steps_per_epoch, 10)
            if self._evaluator:
                self._algo.fit(
                    dataset,
                    n_steps=n_steps,
                    n_steps_per_epoch=self._n_steps_per_epoch,
                    with_timestamp=False,
                    logger_adapter=BweAdapterFactory(root_dir=self._log_dir,
                                                    output_model_name=self._output_model_name),
                    evaluators=evaluators,
                    save_interval=save_interval,
                    enable_ddp=self._ddp,
                )
            else:
                # offline training without evaluators
                self._algo.fit(
                    dataset,
                    n_steps=n_steps,
                    n_steps_per_epoch=self._n_steps_per_epoch,
                    with_timestamp=False,
                    logger_adapter=BweAdapterFactory(root_dir=self._log_dir,
                                                    output_model_name=self._output_model_name),
                    save_interval=save_interval,
                    enable_ddp=self._ddp,
                )
        else:
            # offline training for other workers rank=1,2,... No logging
            self._algo.fit(
                dataset,
                n_steps=n_steps,
                n_steps_per_epoch=self._n_steps_per_epoch,
                with_timestamp=False,
                logger_adapter=d3rlpy.logging.NoopAdapterFactory(),
                enable_ddp=self._ddp,
            )

        t3 = time.process_time()
        print(f'Worker {self._rank} training time: {t3 - t2} s')


    # train the model using all data files under given folder
    def train_model_gradually(self):
        # load the list of files under given folder
        datafiles = self.get_list_data_files()
        if self._rank == 0:
            # name of log folder
            start_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            self._log_dir = self._log_dir + "_" + start_date
            print(f"Worker {self._rank} logging folder {self._log_dir} will be created.")

        for file in datafiles:
            dataset = self.load_MDP_dataset(file)
            test_episodes = dataset.episodes[:1]
            evaluators = {
                'td_error' : d3rlpy.metrics.evaluators.TDErrorEvaluator(test_episodes),
                'discounted_advantage' : d3rlpy.metrics.evaluators.DiscountedSumOfAdvantageEvaluator(test_episodes),
                'average_value' : d3rlpy.metrics.evaluators.AverageValueEstimationEvaluator(test_episodes),
                'action_diff' : d3rlpy.metrics.evaluators.ContinuousActionDiffEvaluator(test_episodes),
            }

            self.train(dataset, evaluators)

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

    ac_encoder_factory = d3rlpy.models.encoders.VectorEncoderFactory(hidden_units=[256,256,256])
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
        observation_scaler=d3rlpy.preprocessing.StandardObservationScaler(),
        action_scaler=d3rlpy.preprocessing.MinMaxActionScaler(),
        reward_scaler=d3rlpy.preprocessing.MinMaxRewardScaler(minimum=REWARD_MIN, maximum=REWARD_MAX),
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

#    ac_encoder_factory = LSTMEncoderFactory(1)
    ac_encoder_factory = d3rlpy.models.encoders.VectorEncoderFactory(hidden_units=[256,256,256])
    sac = d3rlpy.algos.SACConfig(
        batch_size=_batch_size,
        gamma=_gamma,
        actor_learning_rate=_actor_learning_rate,
        critic_learning_rate=_critic_learning_rate,
        temp_learning_rate=_temp_learning_rate,
        tau=_tau,
        n_critics=_n_critics,
        initial_temperature=_initial_temperature,
        actor_encoder_factory=ac_encoder_factory,
        critic_encoder_factory=ac_encoder_factory,
        observation_scaler=d3rlpy.preprocessing.StandardObservationScaler(),
        action_scaler=d3rlpy.preprocessing.MinMaxActionScaler(),
        reward_scaler=d3rlpy.preprocessing.MinMaxRewardScaler(minimum=REWARD_MIN, maximum=REWARD_MAX)
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

    ac_encoder_factory = d3rlpy.models.encoders.VectorEncoderFactory(hidden_units=[256,256,256])
    
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
        observation_scaler=d3rlpy.preprocessing.StandardObservationScaler(),
        action_scaler=d3rlpy.preprocessing.MinMaxActionScaler(),
        reward_scaler=d3rlpy.preprocessing.MinMaxRewardScaler(minimum=REWARD_MIN, maximum=REWARD_MAX)
    ).create(device=params['device'])

    return bcq


def createDDPG(params):
    # parameters for algorithm
    _batch_size = params['batch_size']
    _gamma = params["gamma"]
    _n_critics = params["n_critics"]
    _tau = params["tau"]
    _actor_learning_rate = params["actor_learning_rate"]
    _critic_learning_rate = params["critic_learning_rate"]

#    ac_encoder_factory = LSTMEncoderFactory(1)
    ac_encoder_factory = d3rlpy.models.encoders.VectorEncoderFactory(hidden_units=[256,256,256])
    ddpg = d3rlpy.algos.DDPGConfig(
        batch_size=_batch_size,
        gamma=_gamma,
        actor_learning_rate=_actor_learning_rate,
        critic_learning_rate=_critic_learning_rate,
        tau=_tau,
        n_critics=_n_critics,
        actor_encoder_factory=ac_encoder_factory,
        critic_encoder_factory=ac_encoder_factory,
        observation_scaler=d3rlpy.preprocessing.StandardObservationScaler(),
        action_scaler=d3rlpy.preprocessing.MinMaxActionScaler(),
        reward_scaler=d3rlpy.preprocessing.MinMaxRewardScaler(minimum=REWARD_MIN, maximum=REWARD_MAX)
    ).create(device=params['device'])

    return ddpg


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

    ac_encoder_factory = d3rlpy.models.encoders.VectorEncoderFactory(hidden_units=[256,256,256])

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
        observation_scaler=d3rlpy.preprocessing.StandardObservationScaler(),
        action_scaler=d3rlpy.preprocessing.MinMaxActionScaler(),
        reward_scaler=d3rlpy.preprocessing.MinMaxRewardScaler(minimum=REWARD_MIN, maximum=REWARD_MAX)
    ).create(device=params['device'])

    return dt
