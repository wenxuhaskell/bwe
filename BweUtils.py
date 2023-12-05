import json
from typing import Dict, Optional, Tuple
import numpy as np
import os


def load_train_data(
    datafile: os.PathLike | str,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    data = _load_data(datafile)
    if data is None:
        return None

    bandwidth_predictions = np.array(data['bandwidth_predictions'])
    # to make it as type of float32
    # this is required for creating continuous actions space

    # TODO: why +0.01? You can cast the whole np array to float. I don't understand this line (Nikita)
    bandwidth_predictions = bandwidth_predictions + 0.01

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
    # this is required for creating continuous actions space
    bandwidth_predictions = bandwidth_predictions + 0.01

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
