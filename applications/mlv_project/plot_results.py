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
        '/Users/nphamilton/rl_library/applications/mlv_project/inverted_pendulum/ddpg/episode_performance.csv',
        title='Inverted Pendulum DDPG')

    fig2 = plot_results(
        '/Users/nphamilton/rl_library/applications/mlv_project/inverted_pendulum/ars/training_performance.csv',
        title='Inverted Pendulum ARS')

    fig3 = plot_results(
        '/Users/nphamilton/rl_library/applications/mlv_project/cart_pole/ddpg/episode_performance.csv',
        title='Cartpole DDPG')

    fig4 = plot_results(
        '/Users/nphamilton/rl_library/applications/mlv_project/cart_pole/ars/training_performance.csv',
        title='Cartpole ARS')

    fig5 = plot_results(
        '/Users/nphamilton/rl_library/applications/gym_training/ddpg/episode_performance.csv',
        title='Inverted Pendulum DDPG unmodified')

    fig6 = plot_results(
        '/Users/nphamilton/rl_library/applications/gym_training/ars/training_performance.csv',
        title='Inverted Pendulum ARS unmodified')

    fig7 = plot_results(
        '/Users/nphamilton/rl_library/applications/gym_training/ppo/episode_performance.csv',
        title='Inverted Pendulum PPO unmodified')

    plt.show()


