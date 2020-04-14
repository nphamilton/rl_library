% File:   replay_buffer.py
% Author: Nathaniel Hamilton
%
% Description: This class implements a replay buffer where the relevant information of past experiences is stored and can
%              be sampled from.
%
clear all;

%% System Plant Model
A = [0.0, 1.0, 0.0, 0.0; 0.0, 0.0, 0.716, 0.0; 0.0, 0.0, 0.0, 1.0; 0.0, 0.0, 15.76, 0.0];
B = [0.0; 0.9755; 0.0; 1.46] * 15;
C = eye(4);
D = zeros(4,1); %* 0.0005;
Ts = 0.01;

lb = [-0.05; -0.1; -0.05; -0.05];
ub = [0.05; 0.1; 0.05; 0.05];
init_set = Star(lb, ub);

plant = DLinearODE(A, B, C, D, Ts);

%% Controller Model
load('/Users/nphamilton/rl_library/utils/test.mat')

L1 = LayerS(W{1, 1}, b{1, 1}', 'poslin');
L2 = LayerS(W{1, 2}, b{1, 2}', 'poslin');
L3 = LayerS(W{1, 3}, b{1, 3}', 'tansig');

control = FFNNS([L1 L2 L3]);

%% NNCS
% feedbackMap = 1; %ones(4,1);
% ncs = DNonlinearNNCS(control, plant, feedbackMap);

%% Compute reachability set
N = 2;
n_cores = 1;

reachPRM.numCores = n_cores ;
reachPRM.init_set = init_set ;
reachPRM.ref_input = [] ;
reachPRM.numSteps = N ;
reachPRM.reachMethod = 'approx-star';

% [P1, reachTime1] = ncs.reach(reachPRM);

%% Verify
% unsafe_matrix
% unsafe_vector
% unsafeRegion = Halfspace(unsafe_matrix, unsafe_vector);
% 
% [safe, checkingTime] = check_safety(unsafe_mat, unsafe_vec, numOfCores)