import argparse
import json
from typing import Dict, Optional, Tuple
import onnxruntime as ort
import os
import random
import matplotlib.pyplot as plt
import math
import d3rlpy
import torch.cuda

import numpy as np

from d3rlpy.models.encoders import register_encoder_factory
from BweEncoder import LSTMEncoderFactory, ACEncoderFactory
from BweUtils import load_multiple_files, load_test_data

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--model", type=str)
    parser.add_argument("-p", "--parameters", type=str)
    parser.add_argument("-d", "--dir", type=str, default="emudata")
    parser.add_argument("-m", "--maxfiles", type=int, default=3)
    args = parser.parse_args()

    if args.model is None:
        print("Please provide the model file!")
        return

    # if there is already a pre-trained model, load it
    if not os.path.exists(args.model):
        print("There is no pre-trained model for evaluation!\n")
        return

    print(f"Load the pre-trained model from the file {args.model} for evaluation with emulated data!")

    # register your own encoder factory
    register_encoder_factory(ACEncoderFactory)
    register_encoder_factory(LSTMEncoderFactory)

    algo = d3rlpy.load_learnable(args.model, device='cuda:0' if torch.cuda.is_available() else 'cpu:0')

    # load the params.json
#    if not os.path.exists(args.parameters):
#        print("There is no parameters for the model evaluation!\n")
#        return

#    f = open(args.parameters, 'r')
#    data = json.load(f)
#    action_min = data['config']['params']['action_scaler']['params']['minimum'][0]
#    action_max = data['config']['params']['action_scaler']['params']['maximum'][0]

    # load the list of log files under the given directory
    # iterate over files in that directory
    eval_data_files = load_multiple_files(args.dir, args.maxfiles)
    predictions = []
    bw_predictions = []
    capacities = []
    for filename in eval_data_files:
        result = load_test_data(filename)
        observations, bw_preds, v_quality, a_quality, capacity, _ = result
        bw_predictions.append(bw_preds)
        capacities.append(capacity)
        # returns greedy action
        for observation in observations:
            # add batch dimension
            observation = observation.reshape((1, len(observation))).astype(np.float32)
            prediction = algo.predict(observation)[0]
#            if 0 < prediction < 1:
#                prediction = np.floor(prediction * (action_max - action_min))
            predictions.append(prediction)

    bw_predictions = np.concatenate(bw_predictions)
    capacities = np.concatenate(capacities)
    # plot the predictions
    x = range(len(predictions))

    plt.plot(x, predictions, label="model estimate")
    plt.plot(x, bw_predictions, label="existing estimate")
    plt.plot(x, capacities, label="true capacity")
    plt.legend()
    plt.xlabel('step')
    plt.ylabel('bandwidth')

    plt.show()

if __name__ == "__main__":
    main()
