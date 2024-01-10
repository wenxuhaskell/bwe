import argparse
import json
import os
import d3rlpy
import torch
import torch.distributed as dist

import BweModels


def get_device(is_ddp: bool = False) -> str:
    is_cuda = torch.cuda.is_available()
    world_size = dist.get_world_size() if is_ddp else 1

    if is_cuda:
        rank = d3rlpy.distributed.init_process_group("nccl") if is_ddp else 0
        device = f"cuda:{rank}"
    else:
        rank = 0
        device = "cpu:0"

    print(f"Training on {device} with rank {rank} and world_size {world_size}")
    return device, rank, world_size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--conf", type=str, default="cqlconf.json")
    parser.add_argument("-d", "--ddp", default=False, action="store_true")
    args = parser.parse_args()

    # load the configuration parameters
    f = open(args.conf, "r")
    params = json.load(f)
    f.close()
    # add device
    params['ddp'] = args.ddp
    # get devices for training (overwrite the "device" parameter in json file)
    params['device'], params['rank'], params['world_size'] = get_device(args.ddp)

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
    bwe.train_model_gradually()

    # disable it for now
    if False:
        bwe.evaluate_model_offline()

    if params['ddp'] == True:
        d3rlpy.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
