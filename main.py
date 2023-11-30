import argparse
import time

import json
import d3rlpy
import numpy as np

import os
import BweModels
import onnxruntime as ort

import BweReward
import BweUtils
import BweLogger

# incomplete
def evaluatePolicy(modelFileName):
    # export as ONNX
    algo = d3rlpy.load_learnable(modelFileName, device="cpu:0")
    algo.save_policy("policy.onnx")

    # load ONNX policy via onnxruntime
    ort_session = ort.InferenceSession('policy.onnx', providers=["CPUExecutionProvider"])

    # to obtain observations from the dataset or environment (TODO)
    observation = []
    # returns greedy action
    action = ort_session.run(None, {'input_0': observation})
    print(action)
    assert action[0].shape == (1, 1)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, default="cqlconf.json")
    args = parser.parse_args()

    # load the configuration parameters
    f = open(args.conf, "r")
    params = json.load(f)
    f.close()

    if params['algorithmName'] == 'CQL':
        algo = BweModels.createCQL(params)
    elif params['algorithmName'] == "SAC":
        algo = BweModels.createSAC(params)
    elif params['algorithmName'] == "BCQ":
        algo = BweModels.createBCQ(params)
    elif params['algorithmName'] == "DT":
        algo = BweModels.createDT(params)
    else:
        print("Please provide a configuration file with a valid algorithm name!\n")
        return

    # train the model
    bwe = BweModels.BweDrl(params, algo)
    bwe.train_model()

if __name__ == "__main__":
    main()