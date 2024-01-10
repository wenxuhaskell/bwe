import os
import sys

from typing import Dict, Optional, Tuple
import numpy as np
import pathlib
import random
import json

import d3rlpy


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


def load_multiple_files(train_data_dir: str,train_on_max_files: int):
    files = sorted(os.listdir(train_data_dir))
    train_data_files = []
    for name in files:
        f = os.path.join(train_data_dir, name)
        # checking if it is a file
        if os.path.isfile(f):
            train_data_files.append(f)

    # randomly select the specified amount of log files.
    if 0 < train_on_max_files < len(train_data_files):
        train_data_files = random.sample(train_data_files, train_on_max_files)

    print(f"Files to load: {len(train_data_files)}")
    return train_data_files

def create_mdp_dataset_from_files(train_data_files, rw_func)\
        -> d3rlpy.dataset.MDPDataset:

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
