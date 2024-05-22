import argparse
import pandas as pd
import tkinter
from tkinter import filedialog, Tk, Button, Label, messagebox
from typing import Any, Dict, List
import d3rlpy
import matplotlib.pyplot as plt
import numpy as np
import torch.cuda
from sklearn.preprocessing import MinMaxScaler
from d3rlpy.models.encoders import register_encoder_factory

from BweEncoder import LSTMEncoderFactory, ACEncoderFactory
from BweUtils import load_train_data_from_file
from BweReward import Feature, MI, MIType, reward_qoe_v1, reward_r3net, reward_qoe_v2, reward_qoe_v3, reward_qoe_v4, reward_qoe_v5, process_feature_qoev3, process_feature_qoev4, process_feature_qoev5, process_feature_r3net


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


class eval_model:
    def __init__(self, initdir: str):
        self.__initdir = initdir
        self.__data_filenames = []
        self.__model_filename = ''
        self.__reward_func = ''
        self.__plot_path = ''
        self.__show_plot = False
        self.__plot_log = False
        self.__plot_reward = False

    def run(self):
        window = Tk()

        window.title("Evaluator")

        window.geometry("600x400")

        window.config(background="grey")

        self.select_model_file()

        label_model = Label(text="Model file: " + self.__model_filename)

        self.select_data_file()

        data_file_names = ''.join([o + '\n' for o in self.__data_filenames])
        label_data = Label(text=data_file_names)

        self.select_output_directory()

        self.ask_plot_reward()

        self.ask_show_plot()

        self.eval_model()

        label_model.grid(column=1, row=1)
        label_data.grid(column=1, row=2)
        window.mainloop()

    def select_model_file(self):
        self.__model_filename = filedialog.askopenfilename(initialdir=self.__initdir,
                                              title="Select a model file",
                                              filetypes=(("Model file", "*.d3"), ("All files", "*.*")))

    def select_data_file(self):
        self.__data_filenames = filedialog.askopenfilenames(initialdir=self.__initdir,
                                                title="Select data files",
                                                filetypes=(("Json files", "*.json"), ("Npz files", "*.npz")))

    def select_output_directory(self):
        self.__plot_path = filedialog.askdirectory()
        print(self.__plot_path)

    def ask_show_plot(self):
        self.__show_plot = messagebox.askyesno("Show each plot?")

    def ask_plot_reward(self):
        self.__plot_reward = messagebox.askyesno("Plot rewards?")

    def calc_action_diff(self, pred, bwpred):
        # percentiles:
        # up1 - 95%, low1 - 5%
        # up2 - 90%, low2 - 10%
        # up3 - 85%, low3 - 15%
        # up4 - 80%, low4 - 20%
        # up5 - 75%, low5 - 25%
        per_index = [95,90,85,80,75,70,65,60,55,50,45,40,35,30,25,20,15,10,5]
        self.__pred_per = np.round(np.percentile(pred, per_index)/1000, 1)
        self.__bwpred_per = np.round(np.percentile(bwpred, per_index)/1000, 1)

        self.__diff_per = self.__pred_per - self.__bwpred_per

        row_index = ['95%', '90%', '85%', '80%', '75%', '70%', '65%', '60%', '55%', '50%', '45%', '40%', '35%', '30%', '25%', '20%', '15%', '10%', '5%']

        self.__diffs = (pred - bwpred)/1000

        data = {
            'Model': self.__pred_per,
            'Baseline': self.__bwpred_per,
            'Diff': self.__diff_per
        }

        df = pd.DataFrame(data, row_index)

        print('Percentiles in kbps \n')
        print(df)

        print("Average difference [unit: kbps] \n    ")
        ave = np.sum(self.__diffs)/len(self.__diffs)
        print(ave)

    def eval_model(self):
        if self.__model_filename == '':
            print("Please select the model file!")
            return

        if self.__data_filenames == []:
            print("Please select data files!")
            return

        if self.__plot_path == '':
            print("Please select the output folder!")
            return

        print(f"Load the pre-trained model from the file {self.__model_filename}")

        # register your own encoder factory
        register_encoder_factory(ACEncoderFactory)
        register_encoder_factory(LSTMEncoderFactory)

        algo = d3rlpy.load_learnable(self.__model_filename, device='cuda:0' if torch.cuda.is_available() else 'cpu:0')
        # load the list of log files under the given directory
        # iterate over files in that directory

        inner_params = {
            "MAX_RATE": dict(zip(MI, [0.0] * len(MI))),
            "MAX_DELAY": dict(zip(MI, [0.0] * len(MI))),
        }

        for filename in self.__data_filenames:
            result = load_train_data_from_file(filename)
            observations, bw_predictions, r, t, videos, audios, capacity, lossrate = result
            # for r3net
#            f_rwds = [reward_r3net(o, inner_params) for o in observations]
            # for qoe_v3
            f_rwds = [reward_qoe_v5(o, inner_params, v, a) for (o, v, a) in zip(observations, videos, audios)]
            observations = process_feature_qoev5(observations)
            # for qoe_v4
#            f_rwds = np.array([reward_qoe_v4(o, inner_params, v, a) for (o, v, a) in zip(observations, videos, audios)])
            # exclude reward of NANs
#            indices = [i for i, x in enumerate(f_rwds) if not np.isnan(x)]
#            f_rwds = f_rwds[indices]
#            bw_predictions = bw_predictions[indices]
#            observations = observations[indices]
#            observations = process_feature_qoev4(observations)

            predictions = []
            # predict the bandwidth
            for observation in observations:
                # add batch dimension for prediction
                observation = observation.reshape((1, len(observation))).astype(np.float32)
                prediction = algo.predict(observation)[0]
                predictions.append(prediction)

            f_rwds = np.append(f_rwds[1:], f_rwds[-1])

            # plot the predictions
            x = range(len(predictions))
            self.calc_action_diff(predictions, bw_predictions)
            predictions_scaled = [x / 1000000 for x in predictions]
            bw_predictions_scaled = [x / 1000000 for x in bw_predictions]

            algo_name = self.__model_filename.split('/')[-1].split('.')[0].replace('model', '')
            log_file_name = filename.split('/')[-1]
            log_file_name_prefix = log_file_name.split('.')[0]

            if not self.__plot_reward:
                plt.clf()
                plt.plot(x, predictions_scaled, linewidth=0.8, label="estimate")
                plt.plot(x, bw_predictions_scaled, linewidth=0.8, label="baseline")
                plt.legend()
                plt.ylabel("Bandwidth [mbps]")
                plt.xlabel("Step")
                plt.title(f'{algo_name} \n {log_file_name}')
                plt.savefig(f'{self.__plot_path}/{log_file_name_prefix}_e.png')
                if self.__show_plot:
                    plt.show()
            else:
                plt.clf()
                plt.subplot(2, 1, 1)
                plt.plot(x, predictions_scaled, linewidth=0.8, label="estimate")
                plt.plot(x, bw_predictions_scaled, linewidth=0.8, label="baseline")
                plt.legend()
                plt.ylabel("Bandwidth")
                plt.xlabel("Step")
                plt.title(f'{algo_name} \n {log_file_name}')

                plt.subplot(2, 1, 2)
                plt.plot(x, f_rwds, linewidth=1, label="reward")
                plt.legend()
                plt.ylabel('Reward')
                algo_name = self.__model_filename.split('/')[-1]
                log_file_name = filename.split('/')[-1]
                plt.xlabel('Step')
                plt.savefig(f'{self.__plot_path}/{log_file_name_prefix}_e_rw.png')
                if self.__show_plot:
                    plt.show()

        print("Evaluation finishes!")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--model", type=str)
    parser.add_argument("-p", "--parameters", type=str)
    parser.add_argument("-d", "--dir", type=str, default="emudata")
    parser.add_argument("-m", "--maxfiles", type=int, default=3)
    args = parser.parse_args()

    e = eval_model('..')
    e.run()


if __name__ == "__main__":
    main()
