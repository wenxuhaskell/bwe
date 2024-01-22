import argparse
import tkinter
from tkinter import filedialog, Tk, Button, Label

import d3rlpy
import matplotlib.pyplot as plt
import numpy as np
import torch.cuda
from d3rlpy.models.encoders import register_encoder_factory

from BweEncoder import LSTMEncoderFactory, ACEncoderFactory
from BweUtils import load_test_data, load_train_data_from_file

model_filename = ''
data_filenames =[]


class eval_model:
    def __init__(self, initdir: str):
        self.__initdir = initdir
        self.__data_filenames = []
        self.__model_filename = ''

    def run(self):
        window = Tk()

        window.title("Evaluator")

        window.geometry("600x400")

        window.config(background="grey")

        label_model = Label(text="Model file: " + self.__model_filename)

        self.select_model_file()

        label_model = Label(text="Model file: " + self.__model_filename)

        self.select_data_file()

        data_file_names = ''.join([o + '\n' for o in self.__data_filenames])
        label_data = Label(text=data_file_names)

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

    def eval_model(self):
        if self.__model_filename == '':
            print("Please select the model file!")
            return

        if self.__data_filenames == []:
            print("Please select data files!")
            return

        print(f"Load the pre-trained model from the file {self.__model_filename}")

        # register your own encoder factory
        register_encoder_factory(ACEncoderFactory)
        register_encoder_factory(LSTMEncoderFactory)

        algo = d3rlpy.load_learnable(self.__model_filename, device='cuda:0' if torch.cuda.is_available() else 'cpu:0')
        # load the list of log files under the given directory
        # iterate over files in that directory

        predictions = []
        bw_predictions = []
        for filename in self.__data_filenames:
            result = load_train_data_from_file(filename)
            observations, bw_preds, _, _ = result
            bw_predictions.append(bw_preds)
            # returns greedy action
            for observation in observations:
                # add batch dimension
                observation = observation.reshape((1, len(observation))).astype(np.float32)
                prediction = algo.predict(observation)[0]
                predictions.append(prediction)

        bw_predictions = np.concatenate(bw_predictions)
        # plot the predictions
        x = range(len(predictions))

        plt.plot(x, predictions, label="model estimate")
        plt.plot(x, bw_predictions, label="existing estimate")
        plt.legend()
        plt.xlabel('step')
        plt.ylabel('bandwidth')
        plt.title('Evaluation on training dataset')
        plt.show()


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
