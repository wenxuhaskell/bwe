import argparse
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
from BweUtils import load_test_data, load_train_data_from_file
from BweReward import Feature, MI, MIType, reward_qoe_v1, reward_r3net, reward_qoe_v2, reward_qoe_v3, process_feature_qoev3

model_filename = ''
data_filenames =[]


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
        self.__model_filename1 = ''
        self.__model_filename2 = ''
        self.__plot_path = ''
        self.__show_plot = False
        self.__plot_log = False
        self.__plot_reward = False

    def run(self):
        window = Tk()

        window.title("Compare Q Learning with Decision Transformer")

        window.geometry("600x400")

        window.config(background="grey")

        self.select_model_file1()

        label_model1 = Label(text="Model file: " + self.__model_filename1)

        self.select_model_file2()

        label_model2 = Label(text="Model file: " + self.__model_filename1)

        self.select_data_file()

        data_file_names = ''.join([o + '\n' for o in self.__data_filenames])
        label_data = Label(text=data_file_names)

        self.select_output_directory()

        self.ask_plot_reward()

        self.ask_show_plot()

        self.eval_model()

        label_model1.grid(column=1, row=1)
        label_model2.grid(column=1, row=2)
        label_data.grid(column=1, row=3)
        window.mainloop()

    def select_model_file1(self):
        self.__model_filename1 = filedialog.askopenfilename(initialdir=self.__initdir,
                                              title="Select a Q model file",
                                              filetypes=(("Model file", "*.d3"), ("All files", "*.*")))

    def select_model_file2(self):
        self.__model_filename2 = filedialog.askopenfilename(initialdir=self.__initdir,
                                              title="Select a DT model file",
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

    def ask_plot_log(self):
        self.__plot_log = messagebox.askyesno("Plot features?")

    def ask_plot_reward(self):
        self.__plot_reward = messagebox.askyesno("Plot rewards?")

    def eval_model(self):
        if self.__model_filename1 == '' or self.__model_filename2 == '':
            print("Please select two model files!")
            return

        if self.__data_filenames == []:
            print("Please select data files!")
            return

        if self.__plot_path == '':
            print("Please select the output folder!")
            return

        print(f"Load the pre-trained models {self.__model_filename1} and {self.__model_filename2}")

        # register your own encoder factory
        register_encoder_factory(ACEncoderFactory)
        register_encoder_factory(LSTMEncoderFactory)

        algo1 = d3rlpy.load_learnable(self.__model_filename1, device='cuda:0' if torch.cuda.is_available() else 'cpu:0')

        algo2 = d3rlpy.load_learnable(self.__model_filename2, device='cuda:0' if torch.cuda.is_available() else 'cpu:0')
        actor = algo2.as_stateful_wrapper(target_return=0)

        # load the list of log files under the given directory
        # iterate over files in that directory

        inner_params = {
            "MAX_RATE": dict(zip(MI, [0.0] * len(MI))),
            "MAX_DELAY": dict(zip(MI, [0.0] * len(MI))),
        }


        for filename in self.__data_filenames:
            result = load_train_data_from_file(filename)
            observations, bw_predictions, r, t, videos, audios, capacity, lossrate = result

            # extract rewards
            #            f_rwds = [reward_r3net(o, inner_params) for o in observations]
            f_rwds = [reward_qoe_v3(o, inner_params, v, a) for (o, v, a) in zip(observations, videos, audios)]

            observations = process_feature_qoev3(observations)

            predictions1 = []
            predictions2 = []
            # returns greedy action
            for observation, f_reward in zip(observations, f_rwds):
                # add batch dimension for prediction
                #observation = observation.reshape((1, len(observation))).astype(np.float32)
                prediction2 = actor.predict(observation, f_reward)
                predictions2.append(prediction2)

                observation = observation.reshape((1, len(observation))).astype(np.float32)
                prediction1 = algo1.predict(observation)[0]
                predictions1.append(prediction1)

            f_rwds = np.append(f_rwds[1:], f_rwds[-1])

            # plot the predictions
            x = range(len(predictions1))
            predictions1_scaled = [x / 1000000 for x in predictions1]
            predictions2_scaled = [x / 1000000 for x in predictions2]
            bw_predictions_scaled = [x / 1000000 for x in bw_predictions]

            algo_name1 = self.__model_filename1.split('/')[-1].split('.')[0].replace('model', '')
            algo_name2 = self.__model_filename2.split('/')[-1].split('.')[0].replace('model', '')
            log_file_name = filename.split('/')[-1]
            log_file_name_prefix = log_file_name.split('.')[0]

            if not self.__plot_reward:
                plt.clf()
                plt.plot(x, predictions1_scaled, linewidth=0.8, label=algo_name1)
                plt.plot(x, predictions2_scaled, linewidth=0.8, label=algo_name2)
                plt.plot(x, bw_predictions_scaled, linewidth=0.8, label="baseline")
                plt.legend()
                plt.ylabel("Bandwidth [mbps]")
                plt.xlabel("Step")
                plt.title(f'{algo_name1} vs {algo_name2} \n Log: {log_file_name}')
                plt.savefig(f'{self.__plot_path}/{log_file_name_prefix}_e.png')
                if self.__show_plot:
                    plt.show()
            else:
                plt.clf()
                plt.subplot(2, 1, 1)
                plt.plot(x, predictions1_scaled, linewidth=0.8, label=algo_name1)
                plt.plot(x, predictions2_scaled, linewidth=0.8, label=algo_name2)
                plt.plot(x, bw_predictions_scaled, linewidth=0.8, label="baseline")
                plt.legend()
                plt.ylabel("Bandwidth [mbps]")
                plt.xlabel("Step")
                plt.title(f'{algo_name1} vs {algo_name2} \n {log_file_name}')

                plt.subplot(2, 1, 2)
                plt.plot(x, f_rwds, linewidth=1, label="reward")
                plt.legend()
                plt.ylabel('Reward')
                plt.xlabel(f'Evaluate on {log_file_name}')
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
