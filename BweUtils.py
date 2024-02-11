import os

from typing import Dict, Optional, Tuple
import numpy as np
import pathlib
import json
import random
import d3rlpy
import torch
import torch.distributed as dist


def get_device(is_ddp: bool = False) -> str:
    is_cuda = torch.cuda.is_available()
    world_size = 1
    if is_cuda:
        rank = d3rlpy.distributed.init_process_group("nccl") if is_ddp else 0
        device = f"cuda:{rank}"
        world_size = dist.get_world_size() if is_ddp else 1
    else:
        rank = 0
        device = "cpu:0"

    print(f"Training on {device} with rank {rank} and world_size {world_size}")
    return device, rank, world_size


def load_train_data(
    datafile: os.PathLike | str,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    data = _load_data(datafile)
    if data is None:
        return None

    bandwidth_predictions = np.array(data['bandwidth_predictions']).astype(np.float32)
    observations = np.array(data['observations'])
    video_quality = np.array(data['video_quality'])
    audio_quality = np.array(data['audio_quality'])
    return observations, bandwidth_predictions, video_quality, audio_quality


def load_test_data(
    datafile: os.PathLike | str,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    data = _load_data(datafile)
    if data is None:
        return None

    bandwidth_predictions = np.array(data['bandwidth_predictions'])
    # to make it as type of float32
    bandwidth_predictions = bandwidth_predictions.astype(np.float32)

    observations = np.array(data['observations'])
    video_quality = np.array(data['video_quality'])
    audio_quality = np.array(data['audio_quality'])
    capacity_data = np.array(data['true_capacity'])
    lossrate_data = np.array(data['true_loss_rate'])
    return observations, bandwidth_predictions, video_quality, audio_quality, capacity_data, lossrate_data


def _load_data(datafile: os.PathLike | str) -> Optional[Dict]:
    try:
        with open(datafile, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"File not found: {datafile}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in file {datafile}: {e}")
        return None
    return data


def get_decay_weights(num_weights: int, start_weight: float = 0.4, ratio: float = 0.5) -> np.ndarray:
    weights = start_weight * np.power(ratio, np.arange(num_weights))
    weights /= np.sum(weights)
    return weights


def load_multiple_files(train_data_dir: str, train_on_max_files: int, random_choice: bool = False):
    files = sorted(os.listdir(train_data_dir))
    if random_choice:
        random.shuffle(files)
    train_data_files = []
    for name in files:
        f = os.path.join(train_data_dir, name)
        # checking if it is a file
        if os.path.isfile(f) and len(train_data_files) < train_on_max_files:
            train_data_files.append(f)

    print(f"Files to load: {len(train_data_files)}")
    return train_data_files


def create_mdp_dataset_from_files(train_data_files, rw_func) -> d3rlpy.dataset.MDPDataset:
    observations = []
    actions = []
    rewards = []
    terminals = []

    for filename in train_data_files:
        observations_file = []
        actions_file = []
        rewards_file = []
        terminals_file = []
        print(f"Load file {filename}...")
        ext = pathlib.Path(filename).suffix
        if ext.upper() == '.NPZ':
            loaded = np.load(filename, 'rb')
            observations_file = np.array(loaded['obs'])
            actions_file = np.array(loaded['acts'])
            terminals_file = np.array(loaded['terms'])
            if 'rws' in loaded:
                rewards_file = np.array(loaded['rws'])
            else:
                rewards_file = np.array([rw_func(o) for o in observations_file])
        elif ext.upper() == '.JSON':
            observations_file, actions_file, _, _ = load_train_data(filename)
            rewards_file = np.array([rw_func(o) for o in observations_file])
            terminals_file = np.zeros(len(observations_file))
            terminals_file[-1] = 1

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

    return dataset


def create_mdp_dataset_from_file(train_data_file, rw_func) -> d3rlpy.dataset.MDPDataset:
    observations_file = []
    actions_file = []
    rewards_file = []
    terminals_file = []
    print(f"Load file {train_data_file}...")
    ext = pathlib.Path(train_data_file).suffix
    if ext.upper() == '.NPZ':
        loaded = np.load(train_data_file, 'rb')
        observations_file = np.array(loaded['obs'])
        actions_file = np.array(loaded['acts'])
        terminals_file = np.array(loaded['terms'])
        if 'rws' in loaded:
            rewards_file = np.array(loaded['rws'])
        else:
            rewards_file = np.array([rw_func(o) for o in observations_file])
    elif ext.upper() == '.JSON':
        observations_file, actions_file, _, _ = load_train_data(train_data_file)
        rewards_file = np.array([rw_func(o) for o in observations_file])
        terminals_file = np.zeros(len(observations_file))
        terminals_file[-1] = 1

    # create the offline learning dataset
    dataset = d3rlpy.dataset.MDPDataset(
        observations=observations_file,
        actions=actions_file,
        rewards=rewards_file,
        terminals=terminals_file,
        action_space=d3rlpy.ActionSpace.CONTINUOUS,
    )

    return dataset


def create_gym_dataset_from_file(train_data_file, rw_func):

    observations_file = []
    actions_file = []
    rewards_file = []
    terminals_file = []
    print(f"Load file {train_data_file}...")
    ext = pathlib.Path(train_data_file).suffix
    if ext.upper() == '.NPZ':
        loaded = np.load(train_data_file, 'rb')
        observations_file = np.array(loaded['obs'])
        actions_file = np.array(loaded['acts'])
        terminals_file = np.array(loaded['terms'])
        if 'rws' in loaded:
            rewards_file = np.array(loaded['rws'])
    elif ext.upper() == '.JSON':
        observations_file, actions_file, _, _ = load_train_data(train_data_file)
        terminals_file = np.zeros(len(observations_file))
        terminals_file[-1] = 1

    if not rewards_file:
        rewards_file = np.array([rw_func(o) for o in observations_file])
    # remove the first reward since it corresponds to the observation before the first observation.
    rewards_file = np.append(rewards_file[1:], rewards_file[-1])

    # prepare next_observations
    next_observations_file = np.copy(observations_file[1:])
    next_observations_file = np.append(next_observations_file, np.array(observations_file[-1]).reshape(1,-1), axis=0)
    dataset = {'observations': observations_file,
               'next_observations': next_observations_file,
               'actions': actions_file,
               'rewards': rewards_file,
               'terminals': terminals_file}


    return dataset


def load_train_data_from_file(train_data_file):
    observations_file = []
    actions_file = []
    video_file = []
    audio_file = []
    rewards_file = []
    terminals_file = []
    print(f"Load file {train_data_file}...")
    ext = pathlib.Path(train_data_file).suffix
    if ext.upper() == '.NPZ':
        loaded = np.load(train_data_file, 'rb')
        
        observations_file = np.array(loaded['obs'])
        
        actions_file = np.array(loaded['acts'])
        
        if 'vds' in loaded:
            video_file = np.array(loaded['vds'])
        
        if 'ads' in loaded:
            audio_file = np.array(loaded['ads'])
        
        terminals_file = np.array(loaded['terms'])
        
        if 'rws' in loaded:
            rewards_file = np.array(loaded['rws'])
    elif ext.upper() == '.JSON':
        observations_file, actions_file, video_file, audio_file = load_train_data(train_data_file)
        terminals_file = np.zeros(len(observations_file))
        terminals_file[-1] = 1

    terminals_file = np.random.randint(2, size=len(actions_file))

    return observations_file, actions_file, rewards_file, terminals_file, video_file, audio_file
