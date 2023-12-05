import argparse
import json
import numpy as np

import BweModels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--conf", type=str, default="cqlconf.json")
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

    bwe.evaluate_model_offline()


if __name__ == "__main__":
    main()
