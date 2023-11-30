import os
import json
import numpy as np

def load_data(datafile):
    if datafile == '':
        print("Please give a valid log file name! \n")
    else:
        f = open(datafile)
        d = json.load(f)
        f.close()

        bwe_data = np.array(d['bandwidth_predictions'])
        # to make it as type of float32
        # this is required for creating continuous actions space
        bwe_data = bwe_data + 0.01
        ob_data = np.array(d['observations'])
        # return states and actions
        return ob_data, bwe_data
