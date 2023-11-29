#!/bin/sh
docker run --gpus all -it --name "wen-container" --network=host -v /home/nsmirnov/bandwidth_challenge:/home/code/bandwidth_challenge/data wen:latest bash