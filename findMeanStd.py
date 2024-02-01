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

    observations = np.array(data['observations'])

    return observations


def process_file(filename: str) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    # load the log file and prepare the dataset
    observations_file = load_train_data(filename)

    assert len(observations_file) > 0, f"File {filename} is empty"

    return observations_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--idir", type=str, default="data")
    parser.add_argument("-e", "--edir", type=str, default="emudata")
    parser.add_argument("-o", "--odir", type=str, default="../data_np")
    args = parser.parse_args()

    train_data_files = []
    # load the list of log files under the given directory
    # iterate over files in that directory
    files = sorted(os.listdir(args.idir))
    for name in files:
        f = os.path.join(args.idir, name)
        # checking if it is a file
        if os.path.isfile(f):
            train_data_files.append(f)

    # load the list of log files under the given directory
    # iterate over files in that directory
    files = sorted(os.listdir(args.edir))
    for name in files:
        f = os.path.join(args.edir, name)
        # checking if it is a file
        if os.path.isfile(f):
            train_data_files.append(f)


    print(f"Files to load: {len(train_data_files)}")

    t_start = time.process_time()

    observations = []
    
    # load data log files
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file, filename) for filename in train_data_files]
        for future in tqdm(as_completed(futures), desc=f'Loading', unit="file"):
            obs_file = future.result()
            observations.append(obs_file)

    observations = np.concatenate(observations)
    obs_mean = observations.mean(axis=0)
    obs_std = observations.std(axis=0)
    # create the file
    f_name = f"{args.odir}/meanstdvalues.npz"
    with open(f_name, 'wb') as f_o:
        np.savez_compressed(f_o,
                            obs_mean=obs_mean,
                            obs_std=obs_std)

    # elapsed time
    t_end = time.process_time()
    print(f"Saving mean and std in {f_name}. {t_end - t_start} (s) used")

    with np.load(f_name, 'rb') as loaded:
        print("obs mean: ")
        print(loaded['obs_mean'])
        print("obs std: ")
        print(loaded['obs_std'])

if __name__ == "__main__":
    main()