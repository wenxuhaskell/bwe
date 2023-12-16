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
    # calculate reward
#    rewards_file = np.array([self._reward_func(o) for o in observations_file])
    # terminals are not used so they should be non 1.0
#        terminals_file = np.zeros(len(observations_file))
    terminals_file = np.random.randint(2, size=len(observations_file))
    # timeout at the end of the file
    #timeouts_file = np.zeros(len(observations_file))
    #timeouts_file[-1] = 1.0
    return observations_file, actions_file, terminals_file



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str, default="data")
    parser.add_argument("-m", "--maxfiles", type=int, default=100)
    parser.add_argument("-o", "--outfile", type=str, default="array.npz")
    parser.add_argument("-c", "--outfile_comp", type=str, default="array_comp.npz")
    args = parser.parse_args()

    # load the list of log files under the given directory
    # iterate over files in that directory
    files = sorted(os.listdir(args.dir))
    train_data_files = []
    for name in files:
        f = os.path.join(args.dir, name)
        # checking if it is a file
        if os.path.isfile(f):
            train_data_files.append(f)
            if args.maxfiles > 0 and len(train_data_files) == args.maxfiles:
                break
    print(f"Files to load: {len(train_data_files)}")

    observations = []
    actions = []
    terminals = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file, filename) for filename in train_data_files]
        for future in tqdm(as_completed(futures), desc="Loading MDP", unit="file"):
            result = future.result()
            observations_file, actions_file, terminals_file = result
            observations.append(observations_file)
            actions.append(actions_file)
            terminals.append(terminals_file)

    observations = np.concatenate(observations)
    actions = np.concatenate(actions)
    terminals = np.concatenate(terminals)

    if args.outfile is not None:
        t1 = time.process_time()
        np.savez(args.outfile, obs=observations, acts=actions, terms=terminals)
        t2 = time.process_time()
        print(f"Time for saving file: {t2 - t1}")

    if args.outfile_comp is not None:
        t1 = time.process_time()
        np.savez_compressed(args.outfile_comp, obs=observations, acts=actions, terms=terminals)
        t2 = time.process_time()
        print(f"Time for saving compressed file: {t2-t1}")

#    t3 = time.process_time()
#    loaded = np.load(args.outfile_comp)
#    t4 = time.process_time()
#    print(f"Time for loading compressed file: {t4-t3}")

#    print(loaded['obs'][5])
#    print(loaded['acts'][5])
#    print(loaded['terms'][5])



if __name__ == "__main__":
    main()
