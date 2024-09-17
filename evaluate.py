import argparse
import time

import pandas as pd
import tkinter as tk
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
from BweReward import Feature, MI, MIType, reward_qoe_v1, reward_r3net, reward_qoe_v2, reward_qoe_v3, reward_qoe_v4, reward_qoe_v5, process_feature_qoev3, process_feature_qoev4, process_feature_qoev5, process_feature_r3net, process_feature_qoev3_compact


model_filename = ''
data_filenames =[]


class eval_model:
    def __init__(self, initdir: str):
        self.__initdir = initdir
        self.__data_filenames = []
        self.__model_filename = ''
        self.__reward_func = ''
        self.__plot_path = ''
        self.__show_plot = False
        self.__plot_capacity = False
        self.__plot_reward = False
        self.__mse = 0
        self.__pred_err_rate = 0
        self.__over_err = 0
        self.__mse_baseline = 0
        self.__pred_err_rate_baseline = 0
        self.__over_err_baseline = 0

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

        self.ask_plot_capacity()

        label_model.grid(column=1, row=1)
        label_data.grid(column=1, row=2)

        self.eval_model()
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

    def ask_plot_capacity(self):
        self.__plot_capacity = messagebox.askyesno("Plot true capacity?")

    def calc_pred_err_rate(self, pred, bwpred):
        p = np.squeeze(np.array(pred))
        b = np.array(bwpred)
        x = p - b
        y = np.abs(x)
        k = np.ones(len(pred))
        y = y / b
        z = np.minimum(k, y)
        err_rate = np.nansum(z) / len(pred)
        return err_rate

    def calc_overestimate_err(self, pred, bwpred):
        p = np.squeeze(np.array(pred))
        b = np.array(bwpred)
        x = p - b
        y = np.zeros(len(pred))
        x = x/b
        z = np.maximum(y, x/b)
        over_err = np.nansum(z) / len(pred)
        return over_err

    def calc_mse(self, pred, bwpred):
        p = np.squeeze(np.array(pred))
        b = np.array(bwpred)
        x = p - b
        y = np.square(x)
        z = np.nansum(y)
        mse = z / len(pred)
        return mse
        #mse = np.sum(np.square(np.squeeze(np.array(pred))-np.array(bwpred)))/ len(pred)

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

        total_records = 0
        total_inference_time = 0

        for filename in self.__data_filenames:
            result = load_train_data_from_file(filename)

            t1 = time.process_time()

            observations, bw_predictions, r, t, videos, audios, capacity, lossrate = result
            # for r3net
#            f_rwds = [reward_r3net(o, inner_params) for o in observations]
            # for qoe_v3
#            f_rwds = [reward_qoe_v3(o, inner_params, v, a) for (o, v, a) in zip(observations, videos, audios)]
#            observations = process_feature_qoev3(observations)
            # for qoe_v3 of long MIs only
            f_rwds = [reward_qoe_v3(o, inner_params, v, a) for (o, v, a) in zip(observations, videos, audios)]
            observations = process_feature_qoev3_compact(observations)
            # for qoe_v5
#            f_rwds = [reward_qoe_v5(o, inner_params, v, a) for (o, v, a) in zip(observations, videos, audios)]
#            observations = process_feature_qoev5(observations)
            # for qoe_v4
#            f_rwds = np.array([reward_qoe_v4(o, inner_params, v, a) for (o, v, a) in zip(observations, videos, audios)])
            # exclude reward of NANs
#            indices = [i for i, x in enumerate(f_rwds) if not np.isnan(x)]
#            f_rwds = f_rwds[indices]
#            bw_predictions = bw_predictions[indices]
#            observations = observations[indices]
#            observations = process_feature_qoev4(observations)
#            if np.any(capacity):
#                capacity = capacity[indices]

            predictions = []
            # predict the bandwidth
            for observation in observations:
                # add batch dimension for prediction
                observation = observation.reshape((1, len(observation))).astype(np.float32)
                prediction = algo.predict(observation)[0]
                predictions.append(prediction)

            t2 = time.process_time()
            print(f'Inference time: {t2 - t1} s')
            total_inference_time = total_inference_time + (t2 - t1)
            total_records = total_records + len(predictions)

            f_rwds = np.append(f_rwds[1:], f_rwds[-1])

            # plot the predictions
            x = range(len(predictions))
            predictions_scaled = [x / 1000000 for x in predictions]
            bw_predictions_scaled = [x / 1000000 for x in bw_predictions]
            capacity_scaled = None
            if np.any(capacity):
                capacity_scaled = [x / 1000000 for x in capacity]
                mse = self.calc_mse(predictions_scaled, capacity_scaled)
                self.__mse = self.__mse + mse
                print("mse")
                print(self.__mse)
                err_rate = self.calc_pred_err_rate(predictions_scaled, capacity_scaled)
                self.__pred_err_rate = self.__pred_err_rate + err_rate
                print("predction error")
                print(self.__pred_err_rate)
                over_err = self.calc_overestimate_err(predictions_scaled, capacity_scaled)
                self.__over_err = self.__over_err + over_err
                print("overestimate error")
                print(self.__over_err)

                mse = self.calc_mse(bw_predictions_scaled, capacity_scaled)
                self.__mse_baseline = self.__mse_baseline + mse
                print(filename.split('/')[-1])
                print("mse-baseline")
                print(self.__mse_baseline)
                err_rate = self.calc_pred_err_rate(bw_predictions_scaled, capacity_scaled)
                self.__pred_err_rate_baseline = self.__pred_err_rate_baseline + err_rate
                print("predction error-baseline")
                print(self.__pred_err_rate_baseline)
                over_err = self.calc_overestimate_err(bw_predictions_scaled, capacity_scaled)
                self.__over_err_baseline = self.__over_err_baseline + over_err
                print("overestimate error-baseline")
                print(self.__over_err_baseline)


            algo_name = self.__model_filename.split('/')[-1].split('.')[0].replace('model', '')
            log_file_name = filename.split('/')[-1]
            log_file_name_prefix = log_file_name.split('.')[0]

            if not self.__plot_reward:
                plt.clf()
                plt.plot(x, predictions_scaled, linewidth=0.8, label="estimate")
                plt.plot(x, bw_predictions_scaled, linewidth=0.8, label="baseline")
                if self.__plot_capacity and np.any(capacity_scaled):
                    plt.plot(x, capacity_scaled, linewidth=0.8, label="true capacity")
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

        if self.__plot_capacity and np.any(capacity):
            number_of_data_files = len(self.__data_filenames)
            print(f'Number of data files: {number_of_data_files}')
            print("--- Model performance ---")
            print("MSE: ")
            print(self.__mse/number_of_data_files)
            print("Prediction error rate: ")
            print(self.__pred_err_rate/number_of_data_files)
            print("Overestimation rate: ")
            print(self.__over_err/number_of_data_files)

            print("--- Baseline performance ---")
            print("MSE-baseline: ")
            print(self.__mse_baseline/number_of_data_files)
            print("Prediction error rate-baseline: ")
            print(self.__pred_err_rate_baseline/number_of_data_files)
            print("Overestimation rate-baseline: ")
            print(self.__over_err_baseline/number_of_data_files)

        print(f'total number of records: {total_records}')
        print(f'total amount of inference time: {total_inference_time} s')
        print(f'inference time per 1000 records: {total_inference_time*1000/total_records} s')
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
