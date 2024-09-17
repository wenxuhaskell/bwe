# This is duplicated to bar_chart_evaluation_results_all.py. (predecessor)
# Usage
# This script is to draw a bar chart of evaluation results of multiple models
# The evaluation results are hardcoded in
# models_all: list of model names to be included in the bar chart
#  mse_all: list of MSEs, in the same order as in models_all
#  pred_err_all: list of prediction errors, in the same order as in models_all
#  overest_err_all: list overestimation errors, in the same order as in models_all
#

from matplotlib import pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 140
plt.rcParams['figure.figsize'] = (5,3)

models_all = ['TD3+BC', 'DT', 'Baseline']


# evaluation results for td_tuned TD3+BC, DT (2 heads 4 layers) and baseline
mse_all         = [0.79, 0.75, 0.72]
pred_err_all  = [0.40, 0.19, 0.13]
overest_err_all = [0.30, 0.27, 0.02]


x = np.arange(len(models_all))
width = 0.2

mse_x = x
pred_err_x = x + width
overest_err_x = x + 2*width



plt.bar(mse_x, mse_all, width = width, color = 'skyblue', label = 'MSE')
plt.bar(pred_err_x, pred_err_all, width = width, color = 'moccasin', label = 'Prediction Error')
plt.bar(overest_err_x, overest_err_all, width = width, color = 'lightsalmon', label = 'Overestimation Error')


plt.xticks(x+width, labels=models_all, fontsize = 15)

for i in range(len(models_all)):
    plt.text(mse_x[i], mse_all[i], mse_all[i], va = 'bottom', ha = 'center', fontsize = 12)
    plt.text(pred_err_x[i], pred_err_all[i], pred_err_all[i], va = 'bottom', ha = 'center', fontsize = 12)
    plt.text(overest_err_x[i], overest_err_all[i], overest_err_all[i], va = 'bottom', ha = 'center', fontsize = 12)


plt.legend(loc = 'best', fontsize = 15)

plt.show()