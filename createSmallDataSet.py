import os
import time
import argparse
import json
from typing import Dict, Optional, Tuple
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from BweReward import RewardFunction, Feature, MI

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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

def process_feature_pca(feature: np.ndarray,
                        dim: int) -> np.ndarray:
    # scale the feature dataset
    scaling = StandardScaler()
    scaling.fit(feature)
    scaled_data = scaling.transform(feature)
    # dimensionality reduction using PCA
    principal = PCA(n_components=dim)
    principal.fit(scaled_data)
    x = principal.transform(scaled_data)

    return x


def process_feature_average(feature: np.ndarray) -> np.ndarray:
    # average the features
    new_feature = []

    for i in range(Feature.PROB_PKT_PROB):
        ave_value_s = np.sum(feature[:, i * 10 + MI.SHORT_60 - 1 : i * 10 + MI.SHORT_300], axis=1)/5
        ave_value_l = np.sum(feature[:, i * 10 + MI.LONG_600 - 1: i * 10 + MI.LONG_3000], axis=1)/5
        new_feature.append(ave_value_s)
        new_feature.append(ave_value_l)

    new_feature = np.array(new_feature).transpose()
    # scale the feature dataset
#    scaling = StandardScaler()
#    scaling.fit(new_feature)
#    x = scaling.transform(new_feature)

    return new_feature


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--idir", type=str, default="data")
    parser.add_argument("-b", "--batchsize", type=int, default=20)
    parser.add_argument("-d", "--dim", type=int, default=8)
    parser.add_argument("-o", "--odir", type=str, default="../data_np_small")
    parser.add_argument("-r", "--rewardfunc", type=str, default="QOE_V1")
    parser.add_argument("-a", "--algo", type=str, default="pca")

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

    counter = 0
    time_used = 0
    reward_func = RewardFunction(args.rewardfunc)
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
        rewards = []
        # load data log and save it into .npz file
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_file, filename) for filename in train_data_files]
            for future in tqdm(as_completed(futures), desc=f'Batch {counter + 1} - Loading MDP', unit="file"):
                result = future.result()
                observations_file, actions_file, terminals_file = result
                c = 0
                for a in actions_file:
                    if a==20000.0:
                        c += 1
                    else:
                        break

                observations_file = observations_file[c:]
                actions_file = actions_file[c:]
                terminals_file = terminals_file[c:]
                # calculate rewards
                rewards_file = [reward_func(o) for o in observations_file]
                rewards_file = np.append(rewards_file[1:], 0)
                # PCA dimensionality reduction of the feature
                if args.algo.upper() == 'PCA':
                    observations_file = process_feature_pca(observations_file, args.dim)
                elif args.algo.upper() == 'AVE':
                    observations_file = process_feature_average(observations_file)
                # save all data from the single data log file
                observations.append(observations_file)
                actions.append(actions_file)
                terminals.append(terminals_file)
                rewards.append(rewards_file)

        observations = np.concatenate(observations)
        actions = np.concatenate(actions)
        terminals = np.concatenate(terminals)
        rewards = np.concatenate(rewards)

        # create the file
        f_o = open(f_path, 'wb')
        np.savez_compressed(f_o, obs=observations, acts=actions, terms=terminals, rws=rewards)
        f_o.close()
        # increase the counter
        counter = counter + 1
        # elapsed time
        t_end = time.process_time()
        print(f"Time (s) for converting data log file {f_path}: {t_end - t_start}")
        time_used = time_used + (t_end - t_start)

    print(f'Total time (s) used: {time_used}')


if __name__ == "__main__":
    main()
