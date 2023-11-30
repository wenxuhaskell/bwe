#Bandwidth estimation

## List of algorithms
CQL, BCQ, SAC, <sub>DecisionTransformer (not fully working yet)</sub>

## Configuration for DRL algorithms
It is in the form of json file. For example cqlconf.json, bcqconf.json, sacconf.json.

If there is gpu available, one can set configuration as "device":"cuda:0" where 0 refers to the gpu 0. Otherwise "cpu:0" allows training without using gpu.

## Configuration for plotting the metrics from model training
It is also in the form of json file. For example cqlplotconf.json, bcqplotconf.json, sacplotconf.json.

## How to train a model
/> python3 main.py --conf cqlconf.json

The trained model and metrics log will be placed under a folder CQL_{YYYYMMDDHHMMSS}. For example CQL_20231129233045. The model file extension is "d3".

## How to export a onnx policy from a trained model
It is implemented alreay but is commented out due to ongoing refactoring. For now one can use command line tool of d3rlpy to export onnx policy (as below).
/>d3rlpy export CQL_20231129224315/model.d3 policy.onnx

How to plot the saved metrics
/>python3 plotall.py --path CQL_{YYYYMMDDHHMMSS} --conf cqlplotconf.json
