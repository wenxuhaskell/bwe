import argparse
import tkinter
from tkinter import filedialog, Tk, Button, Label, messagebox
from typing import Any, Dict, List
import d3rlpy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sklearn.linear_model
import torch.cuda
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.feature_selection import RFE, RFECV
from d3rlpy.models.encoders import register_encoder_factory

import BweReward
from BweUtils import load_test_data, load_train_data_from_file
from BweReward import Feature, MI, MIType, reward_qoe_v1, reward_r3net

model_filename = ''
data_filenames =[]

# short-term MI
def smi_features(feature_vec: List[float]) -> float:
    # use the 5 recent short MIs. (TODO: refinement)
    receive_rate = np.sum(feature_vec[(Feature.RECV_RATE - 1) * 10 : (Feature.RECV_RATE - 1) * 10 + 5]) / 5
    queuing_delay = np.sum(feature_vec[(Feature.QUEUING_DELAY - 1) * 10 : (Feature.QUEUING_DELAY - 1) * 10 + 5]) / 5
    pkt_loss_rate = np.sum(feature_vec[(Feature.PKT_LOSS_RATIO - 1) * 10 : (Feature.PKT_LOSS_RATIO - 1) * 10 + 5]) / 5

    return 0.6 * np.log(4 * receive_rate + 1) - queuing_delay / 1000 - 10 * pkt_loss_rate

# long-term MI
def lmi_features(feature_vec: List[float]) -> float:
    # use the 5 recent short MIs. (TODO: refinement)
    receive_rate = np.sum(feature_vec[(Feature.RECV_RATE - 1) * 10 : (Feature.RECV_RATE - 1) * 10 + 5]) / 5
    queuing_delay = np.sum(feature_vec[(Feature.QUEUING_DELAY - 1) * 10 : (Feature.QUEUING_DELAY - 1) * 10 + 5]) / 5
    pkt_loss_rate = np.sum(feature_vec[(Feature.PKT_LOSS_RATIO - 1) * 10 : (Feature.PKT_LOSS_RATIO - 1) * 10 + 5]) / 5

    return 0.6 * np.log(4 * receive_rate + 1) - queuing_delay / 1000 - 10 * pkt_loss_rate


def rewards_bwe(observation: List[float], rf_params: Dict[str, Any]=None) -> float:
    # use the 5 recent short MIs.
    # to adward
    video_pkt_prob = np.sum(observation[(Feature.VIDEO_PKT_PROB - 1) * 10: (Feature.VIDEO_PKT_PROB - 1) * 10 + 5]) / 5
    receive_rate = np.sum(observation[(Feature.RECV_RATE - 1) * 10 : (Feature.RECV_RATE - 1) * 10 + 5]) / 5

    award = (0.5*video_pkt_prob + 0.5*receive_rate)

    # to punish
    audio_pkt_prob = np.sum(observation[(Feature.AUDIO_PKT_PROB - 1) * 10 : (Feature.AUDIO_PKT_PROB - 1) * 10 + 5]) / 5
    pkt_interarrival = np.sum(observation[(Feature.PKT_INTERARRIVAL - 1) * 10 : (Feature.PKT_INTERARRIVAL - 1) * 10 + 5]) / 5
    pkt_jitter = np.sum(observation[(Feature.PKT_JITTER - 1) * 10 : (Feature.PKT_JITTER - 1) * 10 + 5]) / 5
    pkt_loss_rate = np.sum(observation[(Feature.PKT_LOSS_RATIO - 1) * 10: (Feature.PKT_LOSS_RATIO - 1) * 10 + 5]) / 5
    queuing_delay = np.sum(observation[(Feature.QUEUING_DELAY - 1) * 10 : (Feature.QUEUING_DELAY - 1) * 10 + 5]) / 5

    fine = (0.35*audio_pkt_prob + 0.35*pkt_interarrival + 0.1*pkt_jitter + 0.1*pkt_loss_rate + 0.1*queuing_delay)

    return (award - 0.8*fine + 0.2)*5


def get_feature_by_index(
    observation: List[float],
    feature_index: Feature,
    mi: MI = MI.LONG_600,
) -> float:
    return observation[(feature_index - 1) * 10 + mi - 1]


class eval_features:
    def __init__(self, initdir: str):
        self.__initdir = initdir
        self.__data_filenames = []

    def run(self):
        window = Tk()

        window.title("Feature correlation")

        window.geometry("600x400")

        window.config(background="grey")

        self.select_data_file()

        data_file_names = ''.join([o + '\n' for o in self.__data_filenames])
        label_data = Label(text=data_file_names)

        self.show_corr()

        window.mainloop()

    def select_data_file(self):
        self.__data_filenames = filedialog.askopenfilenames(initialdir=self.__initdir,
                                                title="Select data files",
                                                filetypes=(("Json files", "*.json"), ("Npz files", "*.npz")))

    def show_corr(self):
        if self.__data_filenames == []:
            print("Please select data files!")
            return

        # register your own encoder factory

        observations = []
        predictions = []
        f_rwds = []
        inner_params = {
            "MAX_RATE": dict(zip(MI, [0.0] * len(MI))),
            "MAX_DELAY": dict(zip(MI, [0.0] * len(MI))),
        }

        for filename in self.__data_filenames:
            result = load_train_data_from_file(filename)
            observations_file, preds_file, _, _ = result
            predictions.append(preds_file)

            observations_file = np.reshape(observations_file, [len(observations_file), 15, 10])
            observations_file = observations_file[:,:,5]
            observations.append(observations_file)

        #
        observations = np.concatenate(observations)
        predictions = np.concatenate(predictions)
        predictions = np.reshape(predictions, [-1,1])
        # scaling
        obs_scaler = StandardScaler()
        pred_scaler = MinMaxScaler()

        observations = obs_scaler.fit_transform(observations)
        predictions = pred_scaler.fit_transform(predictions)

        x_train, x_test, y_train, y_test = train_test_split(observations, predictions, test_size=0.3, random_state=23)
        # Recursively eliminate features
#        model = MLPRegressor(hidden_layer_sizes=(128,128,), random_state=23, max_iter=1000)
        model = LinearRegression()
        rfe = RFECV(model, step=1)
        x_rfe = rfe.fit(x_train, y_train)
        print(x_rfe.support_)
        print(x_rfe.ranking_)

        # show correlations among features
        observations = np.concatenate((observations, predictions), axis=1)

        column_names = [f'f{f.value}' for f in Feature]
        column_names.append('p')

        df = pd.DataFrame(data=observations, columns=column_names)
        df.head()

        plt.figure(figsize=(200,200))
        cor = df.corr()
        sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
        plt.show()


def main() -> None:
    e = eval_features('..')
    e.run()


if __name__ == "__main__":
    main()
