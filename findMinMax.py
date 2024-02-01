import argparse
import json
from typing import Dict, Optional, Tuple
import os
import time
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def load_data(datafile: os.PathLike | str) -> Optional[Dict]:
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

def load_train_data(
    datafile: os.PathLike | str,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    data = load_data(datafile)
    if data is None:
        return None

    bandwidth_predictions = np.array(data['bandwidth_predictions'])
    observations = np.array(data['observations'])
    video_quality = np.array(data['video_quality'])
    audio_quality = np.array(data['audio_quality'])
    return observations, bandwidth_predictions, video_quality, audio_quality


def process_file(filename: str) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    # load the log file and prepare the dataset
    observations_file, actions_file, _, _ = load_train_data(filename)
    assert len(observations_file) > 0, f"File {filename} is empty"
    obs_min = np.min(observations_file, axis=0)
    obs_max = np.max(observations_file, axis=0)
    act_min = np.min(actions_file)
    act_max = np.max(actions_file)

    return obs_min, obs_max, act_min, act_max


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--idir", type=str, default="data")
    parser.add_argument("-o", "--odir", type=str, default="../data_np")
    args = parser.parse_args()

    # load the list of log files under the given directory
    # iterate over files in that directory
    files = sorted(os.listdir(args.idir))
    train_data_files = []
    for name in files:
        f = os.path.join(args.idir, name)
        # checking if it is a file
        if os.path.isfile(f):
            train_data_files.append(f)

    print(f"Files to load: {len(train_data_files)}")

    t_start = time.process_time()

    observation_min = np.zeros(shape=(150,))
    observation_max = np.zeros(shape=(150,))
    action_min = 0.0
    action_max = 0.0
    # load data log files
    counter = 0
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file, filename) for filename in train_data_files]
        for future in tqdm(as_completed(futures), desc=f'Batch {counter+1} - Loading MDP', unit="file"):
            obs_min, obs_max, act_min, act_max = future.result()
            observation_min = np.minimum(observation_min, obs_min)
            observation_max = np.maximum(observation_max, obs_max)
            action_min = min(action_min, act_min)
            action_max = max(action_max, act_max)
            counter += 1

    # create the file
    f_name = f"{args.odir}/minmaxvalues.npz"
    with open(f_name, 'wb') as f_o:
        np.savez_compressed(f_o,
                            obs_min=observation_min,
                            obs_max=observation_max,
                            act_min=action_min,
                            act_max=action_max)

    # elapsed time
    t_end = time.process_time()
    print(f"Time (s) for converting data log file {f_name}: {t_end - t_start}")

    with np.load(f_name, 'rb') as loaded:
        print("obs min: ")
        print(loaded['obs_min'])
        print("obs max: ")
        print(loaded['obs_max'])
        print("act min: ")
        print(loaded['act_min'])
        print("act max: ")
        print(loaded['act_max'])

if __name__ == "__main__":
    main()