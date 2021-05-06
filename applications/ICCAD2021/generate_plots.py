"""
TODO header
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter1d
import seaborn as sns

if __name__ == '__main__':
    # Load data from .csv files
    print("Loading data to generate plots...")
    avg_dynamics_train_data = pd.read_csv("./data/avg_dynamics/episode_performance.csv", header=0,
                                          names=['training steps', 'time', 'steps in evaluation', 'reward', 'done',
                                                 'exit'])
    hybrid_dynamics_train_data = pd.read_csv("./data/hybrid_dynamics/episode_performance.csv", header=0,
                                             names=['training steps', 'time', 'steps in evaluation', 'reward', 'done',
                                                    'exit'])
    avg_dynamics_eval_data = pd.read_csv("./data/avg_dynamics_evaluations.csv")
    hybrid_dynamics_eval_data = pd.read_csv("./data/hybrid_dynamics_evaluations.csv")

    # Plot the training curve from training in averaged dynamics environment
    fig1 = sns.relplot(x="training steps", y="reward", kind="line", data=avg_dynamics_train_data)
    fig1.fig.suptitle("Training Curve with Averaged Dynamics")
    fig1.set(xlim=[0, 100000], ylim=[-11000, 1000])
    plt.savefig('./figures/figure1_avg_dynamics_training_curve.pdf')

    # Plot the training curve from training in hybrid dynamics environment
    fig2 = sns.relplot(x="training steps", y="reward", kind="line", data=hybrid_dynamics_train_data)
    fig2.fig.suptitle("Training Curve with Hybrid Dynamics")
    fig2.set(xlim=[0, 100000], ylim=[-11000, 1000])
    plt.savefig('./figures/figure2_hybrid_dynamics_training_curve.pdf')

    # Plot the evaluation simulation results within the averaged dynamics environment
    fig3 = sns.relplot(x="time", y="voltage", kind="line", hue="method", data=avg_dynamics_eval_data)
    plt.savefig('./figures/figure3a_eval_voltage_vs_time.pdf')
    fig4 = sns.relplot(x="time", y="accumulated reward", kind="line", hue="method", data=avg_dynamics_eval_data)
    plt.savefig('./figures/figure3b_eval_reward_vs_time.pdf')
    fig5 = sns.relplot(x="time", y="current", kind="line", hue="method", data=avg_dynamics_eval_data)
    plt.savefig('./figures/figure3c_eval_current_vs_time.pdf')
    fig6 = sns.relplot(x="current", y="voltage", kind="line", sort=False, hue="method", data=avg_dynamics_eval_data)
    plt.savefig('./figures/figure3d_eval_voltage_vs_current.pdf')

    # Plot the evaluation simulation results within the averaged dynamics environment
    fig7 = sns.relplot(x="time", y="voltage", kind="line", hue="method", data=hybrid_dynamics_eval_data)
    plt.savefig('./figures/figure4a_eval_voltage_vs_time.pdf')
    fig8 = sns.relplot(x="time", y="accumulated reward", kind="line", hue="method", data=hybrid_dynamics_eval_data)
    plt.savefig('./figures/figure4b_eval_reward_vs_time.pdf')
    fig9 = sns.relplot(x="time", y="current", kind="line", hue="method", data=hybrid_dynamics_eval_data)
    plt.savefig('./figures/figure4c_eval_current_vs_time.pdf')
    fig10 = sns.relplot(x="current", y="voltage", kind="line", sort=False, hue="method", data=hybrid_dynamics_eval_data)
    plt.savefig('./figures/figure4d_eval_voltage_vs_current.pdf')

