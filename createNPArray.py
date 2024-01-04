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
    # terminals are not used so they should be non 1.0
    terminals_file = np.zeros(len(observations_file))
    terminals_file[-1] = 1
#    terminals_file = np.random.randint(2, size=len(observations_file))
    return observations_file, actions_file, terminals_file



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--idir", type=str, default="data")
    parser.add_argument("-m", "--maxfiles", type=int, default=100)
    parser.add_argument("-o", "--odir", type=str, default="../data_np")
    args = parser.parse_args()

    # load the list of log files under the given directory
    # iterate over files in that directory
    files = sorted(os.listdir(args.idir))
    all_files = []
    train_data_files = []
    for name in files:
        f = os.path.join(args.idir, name)
        # checking if it is a file
        if os.path.isfile(f):
            train_data_files.append(f)
            if args.maxfiles > 0 and len(train_data_files) == args.maxfiles:
                all_files.append(train_data_files.copy())
                # reset the list of train data files
                train_data_files = []

    # add the last files
    if len(train_data_files) > 0:
        all_files.append(train_data_files)

    print(f"Files to load: {len(files)}")
    print(f"Divided into {len(all_files)} batches.")

    t1 = time.process_time()

    counter = 0
    for train_data_files in all_files:
        observations = []
        actions = []
        terminals = []
        # load data log and save it into .npz file
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_file, filename) for filename in train_data_files]
            for future in tqdm(as_completed(futures), desc=f'Batch {counter+1} - Loading MDP', unit="file"):
                result = future.result()
                observations_file, actions_file, terminals_file = result
                observations.append(observations_file)
                actions.append(actions_file)
                terminals.append(terminals_file)

        observations = np.concatenate(observations)
        actions = np.concatenate(actions)
        terminals = np.concatenate(terminals)
        # create the file
        f_name = '{:04d}'.format(counter)
        f_o = open(f"{args.odir}/{f_name}.npz", 'wb')
        np.savez_compressed(f_o, obs=observations, acts=actions, terms=terminals)
        f_o.close()
        # increase the counter
        counter = counter + 1

    t2 = time.process_time()
    print(f"Time for converting data log files: {t2 - t1}")

#    print(loaded['obs'][5])
#    print(loaded['acts'][5])
#    print(loaded['terms'][5])

if __name__ == "__main__":
    main()
