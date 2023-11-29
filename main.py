import argparse
import time

import json
import d3rlpy
import numpy as np

import os
import BweDRL
#import ExpDrl
import onnxruntime as ort


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, default="cqlconf.json")
    args = parser.parse_args()

    f = open(args.conf, "r")
    params = json.load(f)
    f.close()

    if params['algorithmName'] == 'CQL':
        if False:
            # refactoring ongoing (incomplete!)
            logdir, modelfilefullname = ExpDrl.createCQL(params)
            bwe = ExpDrl.BweDrl(params, logdir, modelfilefullname)
            bwe.train_model()
        else:
            bwe = BweDRL.BweCQL(params)
            bwe.train_model()
    elif params['algorithmName'] == "BCQ":
        bwe = BweDRL.BweBCQ(params)
        bwe.train_model()
    elif params['algorithmName'] == "SAC":
        bwe = BweDRL.BweSAC(params)
        bwe.train_model()
    elif params['algorithmName'] == "DT":
        bwe = BweDRL.BweDT(params)
        bwe.train_model()
    else:
        print("Please provide a configuration file with a valid algorithm name!\n")


if __name__ == "__main__":
    main()