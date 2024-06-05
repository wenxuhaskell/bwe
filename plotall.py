import subprocess
import os
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
#
from typing import Optional

def plotall(
    path: str,
    metrics: Optional[str],
    titles: Optional[str],
    plottitle: Optional[str]
) -> None:

    # print params.json
    if os.path.exists(os.path.join(path, "params.json")):
        with open(os.path.join(path, "params.json"), "r") as f:
            params = json.loads(f.read())
        print("")
        for k, v in params.items():
            print(f"{k}={v}")

    metrics_names = []
    for i in metrics:
        t = os.path.join(path, f"{i}.csv")
        metrics_names.append(t)


    n_cols = int(np.ceil(len(metrics_names) ** 0.5))
    n_rows = int(np.ceil(len(metrics_names) / n_cols))

    plt.figure(figsize=(12, 7))

    for i in range(n_rows):
        for j in range(n_cols):
            index = j + n_cols * i
            if index >= len(metrics_names):
                break

            plt.subplot(n_rows, n_cols, index + 1)

            data = np.loadtxt(metrics_names[index], delimiter=",")

            plt.plot(range(1, len(data[:,2])+1, 1), data[:, 2])
            plt.title(os.path.basename(titles[index]))
            plt.xlabel("epoch")
            plt.ylabel("value")

    if plottitle:
        plt.suptitle(plottitle)

    plt.tight_layout()
    plt.show()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="d3rlpylogs")
    parser.add_argument ("--config", type=str, default="plotconf.json")
    args = parser.parse_args()

    print("path", args.path)

    print("config file", args.config)

    f = open(args.config, "r")
    conf = json.load(f)
    f.close()

#    print(conf.values())
    metrics = []
    titles = []
    for i in conf['metric_plots']:
        print(i)
        if i['include'] == 1:
            metrics.append(i['metric'])
            titles.append(i['plot_title'])

    plotall(path=args.path, metrics=metrics, titles=titles, plottitle="Data analysis")


if __name__ == "__main__":
    main()