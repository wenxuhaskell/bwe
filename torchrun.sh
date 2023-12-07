#!/bin/sh
torchrun \
   --nnodes=1 \
   --nproc_per_node=3 \
   --rdzv_id=100 \
   --rdzv_backend=c10d \
   --rdzv_endpoint=localhost:29400 \
   --tee=3 \
   main.py -c cqlconf_docker.json -d