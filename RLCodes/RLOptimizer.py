#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 19:51:38 2023

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
from copy import deepcopy

from threading import Thread, Lock

# Mai
sys.path.insert(0, "/Users/viethungle/Desktop/RLAdaptiveOpt/iteratorCodes")
sys.path.insert(0, "/Users/viethungle/Desktop/RLAdaptiveOpt/NNCodes")
sys.path.insert(0, "/Users/viethungle/Desktop/RLAdaptiveOpt")
# sys.path.insert(0, "/Users/maihuynh/Desktop/RLAdaptiveOpt/iteratorCodes")
# sys.path.insert(0, "/Users/maihuynh/Desktop/RLAdaptiveOpt/NNCodes")

# Cem
# sys.path.insert(
#     0, "/Users/ckaramanli/Documents/LBL internship/codes/RLAdaptiveOpt/iteratorCodes"
# )
# sys.path.insert(
#     0, "/Users/ckaramanli/Documents/LBL internship/codes/RLAdaptiveOpt/NNCodes"
# )

# Stefan
# sys.path.insert(0, "/home/wild/oldcomp/repos/RLAdaptiveOpt/iteratorCodes")
# sys.path.insert(0, "/home/wild/oldcomp/repos/RLAdaptiveOpt/NNCodes")

from iteratorCodes.quadraticProblem import quadraticProblem
from iteratorCodes.iterator import iterator, result
from NNCodes.policyNN import Net, FullNet, flattenWeights, updateNN

# from quadraticProblem import *
# from iterator import *
# from policyNN import *


def gradOfValueFuncOld(net, tau, dataset, total_reward):
    # this function calculates the gradients based on single trajectory of data
    # total_reward here is the total reward of a single trajectory
    T = net.T
    output_data = dataset[:, (T, 2 * T)]
    for i in range(1, tau - 1):
        # take forward step in the network:
        net.zero_grad()
        inputs = dataset[i, :]
        inputs = net.dataPrep(inputs)
        outputs = net(inputs)

        h_index = net.h_space.index(output_data[i + 1, 0])
        eta_index = net.eta_space.index(output_data[i + 1, 1])
        h_output = outputs[0, h_index]
        eta_output = outputs[1, eta_index]

        grad_h = torch.tensor([])
        grad_eta = torch.tensor([])

        net.zero_grad()
        h_output.backward(retain_graph=True)
        for params in net.parameters():
            grad_h = torch.cat((grad_h, params.grad.flatten()))
            # TODO! Check zero_grad.
        net.zero_grad()
        eta_output.backward(retain_graph=True)
        for params in net.parameters():
            grad_eta = torch.cat((grad_eta, params.grad.flatten()))

        # divide by the output to calculate the derivative of the log expression
        grad_h /= max(h_output, np.finfo(np.float64).eps)
        grad_eta /= max(eta_output, np.finfo(np.float64).eps)

        if i == 1:
            grad_w = (grad_h + grad_eta) / 2.0
        else:
            grad_w += (grad_h + grad_eta) / 2.0
    # TODO!! Use a more general reward function
    return grad_w * total_reward


def gradOfValueFunc(net: Net | FullNet, tau, dataset, total_reward):
    # this function calculates the gradients based on single trajectory of data
    # total_reward here is the total reward of a single trajectory

    # T = net.T
    # output_data = dataset[:, (T, 2 * T)]
    for i in range(1, tau - 1):
        # take forward step in the network:
        net.zero_grad()
        inputs = dataset[i, :]
        inputs = torch.from_numpy(inputs)
        outputs = net(inputs)

        grad_h = torch.tensor([])
        grad_eta = torch.tensor([])

        net.zero_grad()
        outputs[0].backward(retain_graph=True)
        for params in net.parameters():
            grad_h = torch.cat((grad_h, params.grad.flatten()))
            # TODO! Check zero_grad.
        net.zero_grad()
        outputs[1].backward(retain_graph=True)
        for params in net.parameters():
            grad_eta = torch.cat((grad_eta, params.grad.flatten()))

        # divide by the output to calculate the derivative of the log expression
        grad_h /= max(outputs[0], np.finfo(np.float64).eps)
        grad_eta /= max(outputs[1], np.finfo(np.float64).eps)

        if i == 1:
            grad_w = (grad_h + grad_eta) / 2.0
        else:
            grad_w += (grad_h + grad_eta) / 2.0
    # TODO!! Use a more general reward function
    return grad_w * total_reward


def RLOptimizerOld(
    net,
    iterator_object,
    alpha_space,
    beta,
    tau=100,
    loss_threshold=10 ** (-8),
    max_num_of_iter=200,
):
    # beta: learning rate
    i = 1
    avg_grad_w_norm = sys.maxsize  # dummy initialization
    avg_grad_w_norm_hist = []
    func_val_hist = []
    while avg_grad_w_norm > loss_threshold and i < max_num_of_iter:
        for alpha_idx in range(len(alpha_space)):
            alpha = alpha_space[alpha_idx]
            # initialize result and then initialize iterator:
            # get the information from existing iterator_object
            T = iterator_object.T
            p = iterator_object.p
            result_init = deepcopy(iterator_object.result_init)
            prob = deepcopy(iterator_object.prob)
            # change alpha in prob and then change parts of result_init that are affected
            prob.alpha = alpha
            f_k = prob.obj_func(result_init.x_k)
            Q_sample = np.random.normal(size=(prob.d_x, p))
            Q_sample, _ = np.linalg.qr(Q_sample)
            g_hat_k = prob.g_hat_func(
                result_init.x_k, f_k, result_init.h_k, Q_sample, p
            )
            # incorporate the changes:
            result_init.f_k = f_k
            result_init.g_hat_k = g_hat_k
            # create iterator object:
            iterator_object = iterator(p, T, prob, result_init)
            # call iterator to generate input output lists:
            dataset = iterator_object.dataGenerator(net, tau)
            # TODO!! We skip the reward function here:
            total_reward = dataset[:, 15]
            # IMPORTANT!! Take the exponential of the reward function
            # so that it does not get too small
            # TODO!! We use -1* as reward and it is gradient ascent now
            # total_reward = np.exp(-total_reward)
            # Mai's reward function:
            total_reward = dataset[:, 15]
            total_reward = 1 / (np.sum(total_reward) / dataset[0, 15])
            # feed the data to calculate the gradient:
            grad_w = gradOfValueFunc(net, tau, dataset, total_reward)
            # grad_w = gradOfValueFuncOld(net, tau, dataset, total_reward)
            if alpha_idx == 0:
                avg_total_reward = total_reward / len(alpha_space)
                avg_grad_w = grad_w / len(alpha_space)
            else:
                avg_grad_w += grad_w / len(alpha_space)
                avg_total_reward += total_reward / len(alpha_space)
        avg_grad_w_norm = avg_grad_w.norm()
        flat_w = flattenWeights(net)
        # take the gradient step:
        new_flat_w = flat_w + beta * avg_grad_w
        net = updateNN(net, new_flat_w)  # update network params
        # record history:
        avg_grad_w_norm_hist.append(avg_grad_w_norm.item())
        func_val_hist.append(avg_total_reward)
        print("iter: ", i)
        print("grad_norm: ", avg_grad_w_norm.item())
        print("reward: ", avg_total_reward)
        i += 1
    return net, func_val_hist, avg_grad_w_norm_hist


def RLOptimizer(
    net,
    iterator_object,
    alpha_space,
    beta,
    tau=100,
    loss_threshold=10 ** (-8),
    max_num_of_iter=200,
):
    # beta: learning rate
    i = 1
    avg_grad_w_norm = sys.maxsize  # dummy initialization
    avg_grad_w_norm_hist = []
    func_val_hist = []

    lock = Lock()
    while avg_grad_w_norm > loss_threshold and i < max_num_of_iter:
        threads = []
        total_rewards = [0] * len(alpha_space)
        grad_ws = [0] * len(alpha_space)

        for alpha_idx in range(len(alpha_space)):
            alpha = alpha_space[alpha_idx]
            # initialize result and then initialize iterator:
            # get the information from existing iterator_object
            threads.append(
                Thread(
                    target=run_thread,
                    args=(
                        alpha,
                        iterator_object,
                        net,
                        total_rewards,
                        grad_ws,
                        alpha_idx,
                        tau,
                        lock,
                    ),
                )
            )

        for j in range(len(alpha_space)):
            threads[j].start()

        for j in range(len(alpha_space)):
            threads[j].join()

        avg_total_reward = sum(total_rewards) / len(alpha_space)
        print(total_rewards)
        avg_grad_w = sum(grad_ws) / len(alpha_space)
        print(avg_grad_w)
        avg_grad_w_norm = avg_grad_w.norm()
        flat_w = flattenWeights(net)
        # take the gradient step:
        new_flat_w = flat_w + beta * avg_grad_w
        net = updateNN(net, new_flat_w)  # update network params
        # record history:
        avg_grad_w_norm_hist.append(avg_grad_w_norm.item())
        func_val_hist.append(avg_total_reward)
        print("iter: ", i)
        print("grad_norm: ", avg_grad_w_norm.item())
        print("reward: ", avg_total_reward)
        i += 1
    return net, func_val_hist, avg_grad_w_norm_hist


def run_thread(
    alpha, iterator_object: iterator, net: Net | FullNet, total_rewards, grad_ws, alpha_idx, tau, lock: Lock
):
    T = iterator_object.T
    p = iterator_object.p
    result_init = deepcopy(iterator_object.result_init)
    prob: quadraticProblem = deepcopy(iterator_object.prob)
    # change alpha in prob and then change parts of result_init that are affected
    prob.alpha = alpha
    f_k = prob.obj_func(result_init.x_k)
    Q_sample = np.random.normal(size=(prob.d_x, p))
    Q_sample, _ = np.linalg.qr(Q_sample)
    g_hat_k = prob.g_hat_func(result_init.x_k, f_k, result_init.h_k, Q_sample, p)
    # incorporate the changes:
    result_init.f_k = f_k
    result_init.g_hat_k = g_hat_k
    # create iterator object:
    iterator_object = iterator(p, T, prob, result_init)
    # call iterator to generate input output lists:
    dataset = iterator_object.dataGenerator(net, tau)
    # TODO!! We skip the reward function here:
    # total_reward = dataset[:, 15]
    # IMPORTANT!! Take the exponential of the reward function
    # so that it does not get too small
    # TODO!! We use -1* as reward and it is gradient ascent now
    # total_reward = np.exp(-total_reward)
    # Mai's reward function:
    total_reward = dataset[:, 15]
    total_reward = 1 / (np.sum(total_reward) / dataset[0, 15])
    # feed the data to calculate the gradient:
    lock.acquire()
    grad_w = gradOfValueFunc(net, tau, dataset, total_reward)
    lock.release()
    # grad_w = gradOfValueFuncOld(net, tau, dataset, total_reward)

    total_rewards[alpha_idx] = total_reward
    grad_ws[alpha_idx] = grad_w


# ### MAIN CODE STARTS FROM HERE:

# ### USER INPUTS
# # specify problem parameters:
d_x = 15
beta = 10 ** (-3)
tau = 100
lt = 10 ** (-3)
# max_iter = 200
max_iter = 50
x_bar = np.zeros(d_x)
# specify action space:
h_space = [10**-10, 10**-9, 10**-8, 10**-7, 10**-6]  # for now
# h_space = [10**-10,10**-5,10**0,10**5,10**10]
eta_space = [2**-3, 2**-2, 2**-1, 2**0, 2**1]  # for now
alpha_space = [10**-6, 10**-4, 10**-2, 10**-1]
# alpha_space = [10**-4] # for now
alpha = alpha_space[0]

# ### USER INPUTS PART ENDS
# # set values:
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
# d_input = T * (d_x + p + 6) + 1
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
result_init = result(default_h_k, default_eta_k, k, f_k, rho_k, delta_k, b_k, x_k, g_hat_k)

# create iterator object:
iterator_object = iterator(p, T, prob, result_init)

# create NN:
torch.manual_seed(seed)
net = FullNet(d_x, p, T, h_space, eta_space)
# net = Net(d_x, p, T,  d_layer1,h_space, eta_space)

# save the initial network:
net_init = deepcopy(net)

# train NN using RL:
net, func_val_hist, grad_w_norm_hist = RLOptimizer(
    net, iterator_object, alpha_space, beta, tau, lt, max_iter
)

plt.figure()
plt.plot(func_val_hist, linewidth="3")
plt.title("Reward over iterations", fontsize="17", weight="bold")
plt.xlabel("RL iteration", fontsize="15")
plt.ylabel("reward", fontsize="15")
plt.savefig("reward_over_iter.png", dpi=600)
plt.show()

plt.figure()
plt.semilogy(grad_w_norm_hist, linewidth="3")
plt.title("Gradient of the rewards w.r.t network's parameters", fontsize="13", weight="bold")
plt.xlabel("RL iteration", fontsize="15")
plt.ylabel(r"$ \nabla_{\mathbf{w}}$ network", fontsize="15")
plt.savefig("grad_of_net_over_iter.png", dpi=600)
plt.show()

# compare initial network and final network:
dataset_initial = iterator_object.dataGenerator(net_init, tau)
dataset_final = iterator_object.dataGenerator(net, tau)

plt.figure()
(line1,) = plt.semilogy(dataset_initial[:, T], label="h_k, initial NN", linewidth="3")
(line2,) = plt.semilogy(
    dataset_initial[:, 2 * T], label="eta_k, initial NN", linewidth="3"
)
(line3,) = plt.semilogy(dataset_final[:, T], label="h_k, final NN", linewidth="3")
(line4,) = plt.semilogy(dataset_final[:, 2 * T], label="eta_k, final NN", linewidth="3")
plt.title("Change in the actions before and after RL", fontsize="17", weight="bold")
plt.xlabel("Iterator iteration", fontsize="15")
plt.ylabel("actions", fontsize="15")
leg = plt.legend(loc="best", fontsize="15")
plt.savefig("change_in_actions.png", dpi=600)
plt.show()
