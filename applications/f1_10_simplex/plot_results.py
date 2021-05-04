"""
TODO
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set(style="darkgrid")


def plot_results(file_name, title):
    data = pd.read_csv(file_name, header=0, names=['steps', 'time', 'eval_steps', 'reward', 'done', 'exit'])
    fig = sns.relplot(x="steps", y="reward", kind="line", data=data)
    fig.fig.suptitle(title)

    return fig


if __name__ == '__main__':
    fig1 = plot_results(
        '/home/nate/rl_library/applications/f1_10_simplex/ars/final_models/training_performance.csv',
        title='ARS Training Curve')

    plt.show()


