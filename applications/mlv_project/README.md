# Machine Learning Verification - CS 8395 - Spring 2020
This folder contains all of the code I have written for the following project. The purpose and explanation for this 
project are explained below. If you encounter any issues using or trying to replicate the results explained in the 
report, please contact me at nathaniel.p.hamilton@vanderbilt.edu.

# Is Safe Reinforcement Learning Safer? A Brief Look into How Safe Reinforcement Learning Methods Compare to “Unsafe” Methods
Deep Reinforcement Learning has enabled Neural Network controllers to achieve state-of-the-art performance on many 
high-dimensional control tasks. However, RL allows agents to learn via trial and error, exploring any behavior during 
the learning process. In many realistic domains, this level of freedom is unacceptable. Consider the example of an 
industrial robot arm learning to place objects in a factory. Some behaviors could cause it to damage itself, the plant, 
or nearby workers. As a result, the realm of Safe Reinforcement Learning (SRL) is extremely important. However, little 
to no work has been done to formally verify these SRL methods actually create safe policies.

For this project, I implemented and experimented with 2 “unsafe” and used one safe RL methods in 3 different 
environments. I verify the safety constraints on the learned policies using [nnv](https://github.com/verivital/nnv).

# Algorithms

## "Unsafe"

### Augmented Random Search (ARS)
This method uses random search for training static, linear policies. While it is not meant for training NNs, the 
performance of the learned linear policies matches, or exceeds, state-of-the-art DRL methods. ARS was introduced in 
[Mania et al. 2018](https://papers.nips.cc/paper/7451-simple-random-search-of-static-linear-policies-is-competitive-for-reinforcement-learning).

### Deep Deterministic Policy Gradient (DDPG)
This off-policy, deterministic actor-critic method was developed in [Lillicrap et al. 2015](https://arxiv.org/abs/1509.02971).  
It uses random exploration to estimate a state-action value function. The state-action value function is then used to 
influence the decisions made by the policy function to select the action resulting in the highest reward.

## "Safe"

### Shielding via inductive synthesis
Shielding enforces safety by restricting the action space for the agent at each time step to only actions that have 
been determined safe. The shield determines these safe actions differently depending on the method. In this specific 
method, proposed in [Zhu et al. 2019](https://arxiv.org/abs/1907.07273), the shield by verifying a synthesized program 
treated as a substitute for the NN policy. The synthesized program is easier to verify and the safety is enforced using 
a counterexample-guided inductive synthesis (CEGIS) loop. If the system is about to be driven out of the safety 
boundary, the synthesized program is used to take an action that is guaranteed to be safe.

# Environments

## Cart-Pole
This environment consists of a cart on a track that can move left and right. The cart has a pole attached to the middle 
on a bracket that allows it to move about a fixed point. The goal of the agent is to keep the pole up as close to 
vertical as possible without moving outside a set of boundaries. This is a commonly used baseline for learning 
algorithms with a known plant model. Having a known plant model helps with the verification process. While an 
implementation of this environment exists within OpenAI's Gym suite, it uses only discrete actions. DDPG requires that 
the action space is continous, so agents are trained and evaluated on an LQR representation of the environment provided 
by [Zhu et al. 2019](https://github.com/rowangithub/VRL_CodeReview/blob/master/cartpole_continuous.py).

## Inverted Pendulum
This is a commonly used baseline for learning algorithms with a known plant model. Having a known plant model helps 
with the verification process.

## Linear Quadratic Regulator (LQR) with unknown dynamics
This test environment was highlighted in [Mania et al. 2018](https://papers.nips.cc/paper/7451-simple-random-search-of-static-linear-policies-is-competitive-for-reinforcement-learning)
as being a rigorous, difficult, and applicable benchmarking environment. It is simple to find the optimal solution if 
the dynamics are known, but when the dynamics are unknown, the solution is more difficult to find. This provides a good 
baseline to compare all tested algorithms against and helps identify when learners fail to converge on the optimal 
solution. Also, the plant model is simple and known, which aids in the verification process.

 
