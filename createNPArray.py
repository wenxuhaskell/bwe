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
    parser.add_argument("-b", "--batchsize", type=int, default=100)
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
            if args.batchsize > 0 and len(train_data_files) == args.batchsize:
                all_files.append(train_data_files.copy())
                # reset the list of train data files
                train_data_files = []

    # add the last files
    if len(train_data_files) > 0:
        all_files.append(train_data_files)

    print(f"Files to load: {len(files)}")
    print(f"Divided into {len(all_files)} batches.")

    time_used = 0
    counter = 0
    for train_data_files in all_files:
        # name for new log file
        f_name = '{:04d}'.format(counter)
        f_path = f'{args.odir}/{f_name}.npz'
        # if the file already exists, skip generation.
        if os.path.isfile(f_path):
            print(f'{f_path} already exists, skip it!')
            counter = counter + 1
            continue

        t_start = time.process_time()
        observations = []
        actions = []
        terminals = []
        # load data log and save it into .npz file
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_file, filename) for filename in train_data_files]
            for future in tqdm(as_completed(futures), desc=f'Batch {counter+1} - Loading MDP', unit="file"):
                result = future.result()
                observations_file, actions_file, terminals_file = result
                # filter out bandwidth = 20000.0
                c = 0
                for a in actions_file:
                    if a == 20000.0:
                        c += 1
                    else:
                        break
                observations_file = observations_file[c:]
                actions_file = actions_file[c:]
                terminals_file = terminals_file[c:]

                observations.append(observations_file)
                actions.append(actions_file)
                terminals.append(terminals_file)

        observations = np.concatenate(observations)
        actions = np.concatenate(actions)
        terminals = np.concatenate(terminals)

        # create the file
        f_o = open(f_path, 'wb')
        np.savez_compressed(f_o, obs=observations, acts=actions, terms=terminals)
        f_o.close()
        # increase the counter
        counter = counter + 1
        # elapsed time
        t_end = time.process_time()
        print(f"Time (s) for converting data log file {f_path}: {t_end - t_start}")
        time_used = time_used + (t_end - t_start)

    print(f"Time for converting all data log files: {time_used}")

if __name__ == "__main__":
    main()
