#Bandwidth estimation

## List of algorithms
CQL, BCQ, DDPG, SAC, CRR, TD3+BC, DecisionTransformer

The implementation can be found in BweModels.py

## List of reward functions
OnRL: inspired from OnRL.
R3Net: inspired from R3NET.
QOEV1: It is QOE taken from Nikita's paper. Full set of features. Both short Moniter Intervals (MI) and long MIs.
QOEV2: weighted sum of objective audio/video signals. Full set of features. Both short and long MIs.
QOEV3: Same QOE as above. Reduced set of features (10 features). Both short and long MIs.
QOEV3_compact: Same QOE as above. Reduced set of features (10 features). Long MIs only.
QOEV4: Objective video quality only. Full set of features. Both short and long MIs.
QOEV5: Same QOE as QOEV1. Reduced set of features (7 features). Long MIs only.

The definitions can be found in BweReward.py.

## Configuration for DRL algorithms
It is in the form of json file. For example cqlconf.json, bcqconf.json, sacconf.json.

If there is gpu available, one can set configuration as "device":"cuda:0" where 0 refers to the gpu 0. Otherwise "cpu:0" allows training without using gpu.

## Configuration for plotting the metrics from model training
It is also in the form of json file. For example cqlplotconf.json, bcqplotconf.json, sacplotconf.json.

## How to train a model
/> python3 main.py --conf cqlconf.json

The trained model and metrics log will be placed under a folder CQL_{YYYYMMDDHHMMSS}. For example CQL_20231129233045. The model file extension is "d3".

## How to use Optuna to tune hyperparameters of a model.
/> python3 main.py --conf cqlconf.json

In the conf file, "finetune" needs to be set as "true" and "tune_evaluator" as one of "action_diff", "td_error". The found hyper parameters and tuning logs will be saved under "trails/CQL_params.json, trails/CQL_attrs.json".

## How to export a onnx policy from a trained model
It is implemented alreay but is commented out due to ongoing refactoring. For now one can use command line tool of d3rlpy to export onnx policy (as below).
/>d3rlpy export CQL_20231129224315/model.d3 policy.onnx

## How to plot the saved metrics
/>python3 plotall.py --path CQL_{YYYYMMDDHHMMSS} --conf cqlplotconf.json


## How to evaluate a trained model (Q-learning based)
/> python3 evaluate.py

This script includes GUI. Since the trained model does not contain information on the reward function, the reward function for evaluation is hard coded in the script (line 163-184).
The first dialogue asks for the model (.d3) to be evaluated. 
The 2nd dialogue asks for the log files to be used for evaluation (it is possible to select multiple log files). 
The 3rd dialogue asks for the output folder for plots generated during evaluation. 
The 4th dialogue asks for whether to include reward or not.
The 5th dialogue asks for whether to display the plot after evaluating each log file.
The 6th diagoue asks for wether to inlcude true capacity or not.


## How to evaluate a trained model (Decision Transformer)
/> python3 evaluatedt.py

This script includes GUI. Since the trained model does not contain information on the reward function, the reward function for evaluation is hard coded in the script (line 173-191).
The first dialogue asks for the model (.d3) to be evaluated. 
The 2nd dialogue asks for the log files to be used for evaluation (it is possible to select multiple log files). 
The 3rd dialogue asks for the output folder for plots generated during evaluation.
The 4th dialogue asks for whether to include reward or not.
The 5th dialogue asks for whether to display the plot after evaluating each log file.
The 6th diagoue asks for wether to inlcude true capacity or not.

## Create bar chart from evaluation results of models
/> python3 bar_chart_evaluation_results_all.py

The Microsoft dataset includes To reduce the number of file retrieval during training, This script creates a bar chart using the evaluation results (need to be hard coded in the beginning of the script). 

## Create larger data log files
/>python3 createNPArray.py -i data -b 50 -o data_small

This script creates larger data log file from a number of data log files.
Arguments:
  -i, or --idir: the folder to the data log files
  -b, or --batchsize: the number of data log files to be included in a large data log file
  -o, or --odir: the output folder for large data log files.

## Create larger data log files including pre-calculated rewards and feature reduction
/>python3 createSmallDataSet.py -i data -b 50 -d 8 -o data_small -r R3NET -a pca

This script creates larger data log file from a number of data log files. The reward can be calculated during the creation and be included in the target data log file. It can also use different algorithms such as PCA for feature deduction.
Arguments:
  -i, or --idir: the folder to the data log files
  -b, or --batchsize: the number of data log files to be included in a large data log file
  -d, or --dim: the targeted feature dimension for PCA feature deduction.
  -o, or --odir: the output folder for large data log files.
  -r, or --rewardfunc: the reward function to be used for calculating rewards.
  -a, or --algo: the algorithm for feature deduction.

## Calculate the min/max values of all features in the dataset
/>python3 findMinMax.py -i data -e emudata -o output

This script iterate all data logs in train dataset and find out the min/max values of each feature. The results are saved in a file minmaxvalues.npz and printed in terminal.
Arguments:
  -i or -idir: the folder to the data log files (captured during real world calls)
  -e or -edir: the folder tot he data log files (captured during emulated calls)
  -o or -odir: the folder for output results.

## Calculate the mean/std values of all features in the dataset
/>python3 findMeanStd.py -i data -e emudata -o output

This script iterate all data logs in train dataset and find out the mean/std of each feature. The results are saved in a file meanstdvalues.npz and printed in terminal.
Arguments:
  -i or -idir: the folder to the data log files (captured during real world calls)
  -e or -edir: the folder tot he data log files (captured during emulated calls)
  -o or -odir: the folder for output results.

## Feature correlation analysis
/>python3 feature_select.py

This script uses Recursive Feature Elimination with Cross Validation (in Scikit-Learn) to propose the features that can be eliminated. It also conducts correlation analysis and plot a heatmap based on the results.

## Baseline MLP implementation
/>python3 learn_qoe.py

This script implement a simple Multi-layer perception ([256,256,256,64]) for bandwidth estimation. The purpose of it is to test whether it is possible to learn using only the objective video/audio signals in the dataset. Training log for each epoch is saved under "QOE/qoe_{timestamp}.json".
Arguments:
  -i, or --idir: the folder to the data log files
  -b, or --batchsize: the batch size for training neural network
  -m, or --maxfiles: the maximum number of training files (can be useful during testing)
  -e, or --numepochs: the number of epochs
  -p, or --plot: whether to plot the results or not

## Compare two Q-learning based models
/>python3 comp_q_q.py

This script compares two Q-learning based models. Since the trained model does not contain information on the reward function, the reward function for evaluation is hard coded in the script (line 173-175).

The 1st dialogue asks for the first model file.
The 2nd dialogue asks for the second model file.
The 3rd dialogue asks for the log files to be used for evaluation (it is possible to select multiple log files). 
The 4th dialogue asks for the output folder for results.
The 5th dialogue asks for whether to include reward or not.
The 6th dialogue asks for whether to display the plot after evaluating each log file.
The 7th diagoue asks for wether to inlcude true capacity or not.


## Compare a Q-learning based model with a Decision Transformer model
/>python3 comp_q_dt.py

This script compares a Q-learning based model with a Decision Transformer model. Since the trained model does not contain information on the reward function, the reward function for evaluation is hard coded in the script (line 139-141).

The 1st dialogue asks for the Q-learning based model.
The 2nd dialogue asks for the DecisionTransformer model.
The 3rd dialogue asks for the log files to be used for evaluation (it is possible to select multiple log files). 
The 4th dialogue asks for the output folder for results.
The 5th dialogue asks for whether to include reward or not.
The 6th dialogue asks for whether to display the plot after evaluating each log file.