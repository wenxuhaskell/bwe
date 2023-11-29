#!/bin/sh
docker run --gpus all -it --name "wen-container" --network=host wen:latest bash
