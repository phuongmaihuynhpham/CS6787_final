#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 12:57:23 2023

@author: phuongmaihuynhpham
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import jacobian
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(
    0, "/Users/ckaramanli/Documents/LBL internship/codes/RLAdaptiveOpt/iteratorCodes"
)
sys.path.insert(0, "/Users/maihuynh/Desktop/RLAdaptiveOpt/iteratorCodes")
sys.path.insert(0, "/home/wild/oldcomp/repos/RLAdaptiveOpt/iteratorCodes")
from quadraticProblem import *
from iterator import *
from policyNN import *


### MAIN CODE STARTS FROM HERE:
### USER INPUTS
# specify problem parameters:
d_x = 15
tau = 100
x_bar = np.zeros(d_x)
# specify action space:
h_space = [10**-10, 10**-9, 10**-8, 10**-7, 10**-6]  # for now
# h_space = [10**-10,10**-5,10**0,10**5,10**10]
eta_space = [2**-3, 2**-2, 2**-1, 2**0, 2**1]  # for now
alpha_space = [10**-6, 10**-4, 10**-2, 10**-1]
# alpha_space = [10**-4] # for now
alpha = alpha_space[0]

# specify the indices of the fixed h and eta values:
# if a negative value is selected, then the policy will not be fixed for that action.
fixed_h_index = 1
fixed_eta_index = 1

### USER INPUTS PART ENDS
# set values:
default_h_k = 10 ** (-6)
# default_h_k = 10**5
default_eta_k = 1
seed = 3
np.random.seed(seed)
x_init = np.random.normal(size=d_x)
delta_k = 1
p = int(d_x / 2)
epsilon = 10 ** (-8)
T = 5
d_input = T * (d_x + p + 6) + 1
d_layer1 = (d_x + p + 6) + 1
d_h_space = len(h_space)
d_eta_space = len(eta_space)

# create prob object in iterator object:
# initialize problem:
prob_config = {
    "d_x": d_x,
    "alpha": alpha,
    "x_bar": x_bar,
    "seed": seed,
    "is_stoch": True,
}
prob = quadraticProblem(**prob_config)
k = 0
rho_k = 0
b_k = 0
x_k = x_init
f_k = prob.obj_func(x_k)
Q_sample = np.random.normal(size=(d_x, p))
Q_sample, _ = np.linalg.qr(Q_sample)
g_hat_k = prob.g_hat_func(x_k, f_k, default_h_k, Q_sample, p)
# put dummy values for h_k and eta_k:
result_init = result(
    default_h_k, default_eta_k, k, f_k, rho_k, delta_k, b_k, x_k, g_hat_k
)

# create iterator object:
iterator_object = iterator(p, T, prob, result_init)

# create NN:
torch.manual_seed(seed)
net = Net(d_x, p, T, d_layer1, h_space, eta_space)

# save the initial network:
net_init = copy.deepcopy(net)

### MODIFY THE NN TO GET FIXED POLICY:
net = fixedPolicyNN(net, fixed_h_index, fixed_eta_index)

# test the fixed policy:
dataset = iterator_object.dataGenerator(net, tau)

plt.figure()
(line1,) = plt.semilogy(dataset[:, T], label="h_k", linewidth="3")
(line2,) = plt.semilogy(dataset[:, 2 * T], label="eta_k", linewidth="3")
plt.title("Fixed Policy NN Test", fontsize="17", weight="bold")
plt.xlabel("RL iteration", fontsize="15")
plt.ylabel("actions", fontsize="15")
leg = plt.legend(loc="best", fontsize="15")
plt.savefig("fixed_policy_NN_test_actions.png", dpi=600)
plt.show()

# INDICES PLOT:
# h_indices_list = [];
# eta_indices_list = [];
# for idx in range(dataset.shape[0]):
#     h_indices_list.append(h_space.index(dataset[idx,T]))
#     eta_indices_list.append(h_space.index(dataset[idx,2*T]))

# plt.figure()
# (line1,) = plt.plot(h_indices_list, label="h indices", linewidth="3")
# (line2,) = plt.plot(eta_indices_list, label="eta indices", linewidth="3")
# plt.title("Fixed Policy NN Test", fontsize="17", weight="bold")
# plt.xlabel("RL iteration", fontsize="15")
# plt.ylabel("action indices", fontsize="15")
# leg = plt.legend(loc="best", fontsize="15")
# plt.savefig("fixed_policy_NN_test_action_indices.png", dpi=600)
# plt.show()
