#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 12:59:35 2023

@author: ckaramanli
@coauthor: phuongmaihuynhpham
"""


import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import sys

sys.path.append("..")
from quadraticProblem import *

# specify problem parameters:
d_x = 15
alpha_space = [
    10 ** (-12),
    10 ** (-10),
    10 ** (-8),
    10 ** (-6),
    10 ** (-4),
    10 ** (-2),
    10 ** (-1),
]
seeds = [i for i in range(10)]
x_bar = np.zeros(d_x)

# set values:
h_space = [10**i for i in range(-20, 24, 2)]
np.random.seed(3)
x_init = np.random.normal(size=d_x)
delta_0 = 1
p = int(d_x / 2)
epsilon = 10 ** (-16)


# function definitions:
class result:
    def __init__(self, h_k, eta_k, k, f_k, rho_k, delta_k, b_k, x_k, g_hat_k):
        self.h_k = h_k
        self.eta_k = eta_k
        self.k = k
        self.f_k = f_k
        self.rho_k = rho_k
        self.delta_k = delta_k
        self.b_k = b_k
        self.x_k = x_k
        self.g_hat_k = g_hat_k


def TRDFO(prob, total_iter_num=50, h_k=10 ** (-10), eta_k=1):
    result_list = []
    x_k = x_init
    delta_k = delta_0
    for k in range(total_iter_num):
        Q_sample = np.random.normal(size=(d_x, p))
        Q_sample, _ = np.linalg.qr(Q_sample)
        f_k = prob.obj_func(x_k)
        g_hat_k = prob.g_hat_func(x_k, f_k, h_k, Q_sample, p)
        # take max between machine eps and norm g_hat_k to avoid NaNs
        g_hat_norm = max(np.finfo(np.float64).eps, np.linalg.norm(g_hat_k))
        s_prop = ((-1) * g_hat_k / g_hat_norm) * delta_k
        x_prop = np.dot(Q_sample, s_prop)
        rho_k = (f_k - prob.obj_func(x_prop)) / (g_hat_norm * delta_k)
        b_k = 0
        if rho_k >= eta_k:
            x_k = x_prop
            b_k = 1
        else:
            delta_k /= 2
        # record keeping:
        result_k = result(h_k, eta_k, k, f_k, rho_k, delta_k, b_k, x_k, g_hat_k)
        result_list.append(result_k)
    return result_list


def exper_h_eta(prob, h, eta=1):
    result_list = TRDFO(prob, h_k=h, eta_k=eta)
    k_hist = []
    f_k_hist = []
    b_k_hist = []
    for i in range(len(result_list) - 1):
        k_hist.append(result_list[i].k)
        f_k_hist.append(result_list[i].f_k)
        b_k_hist.append(result_list[i].b_k)
    return f_k_hist, b_k_hist


# store median values of each alpha
sumReward = []
lastReward = []

# store min (max) reward value of each h at different alpha, size len(alpha_space) x len(h_space)
minSum = []
maxSum = []

minLast = []
maxLast = []

for j in range(len(alpha_space)):
    rewards_sum = []
    rewards_last = []
    for seed in seeds:
        # initialize problem:
        prob_config = {
            "d_x": d_x,
            "alpha": alpha_space[j],
            "x_bar": x_bar,
            "seed": seed,
            "is_stoch": True,
        }

        prob = quadraticProblem(**prob_config)

        sum_fk = []
        last_fk = []

        for i in range(len(h_space)):
            f_k_hist, b_k_hist = exper_h_eta(prob, h_space[i])
            last_fk.append(1 / (f_k_hist[-1] / f_k_hist[0]))
            sum_fk.append(1 / (sum(f_k_hist) / f_k_hist[0]))

        rewards_sum.append(sum_fk)
        rewards_last.append(last_fk)

    rewards_sum = np.array(rewards_sum)
    rewards_last = np.array(rewards_last)

    medSum = []
    medLast = []

    sumMin = []
    sumMax = []

    lastMin = []
    lastMax = []

    # find the median reward values of each alpha
    for h_idx in range(len(h_space)):
        medSum.append(np.median(rewards_sum[:, h_idx]))
        medLast.append(np.median(rewards_last[:, h_idx]))

        sumMin.append(np.min(rewards_sum[:, h_idx]))
        sumMax.append(np.max(rewards_sum[:, h_idx]))

        lastMin.append(np.min(rewards_last[:, h_idx]))
        lastMax.append(np.max(rewards_last[:, h_idx]))

    sumReward.append(medSum)
    lastReward.append(medLast)

    minSum.append(sumMin)
    maxSum.append(sumMax)
    minLast.append(lastMin)
    maxLast.append(lastMax)

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
figure()
for i in range(len(alpha_space)):
    plt.loglog(
        h_space, sumReward[i], label="{:.0e}".format(alpha_space[i]), color=colors[i]
    )
    for j in range(len(h_space)):
        plt.loglog(
            [h_space[j], h_space[j]], [minSum[i][j], maxSum[i][j]], color=colors[i]
        )
        plt.loglog(h_space[j], minSum[i][j], "_", color=colors[i])
        plt.loglog(h_space[j], maxSum[i][j], "_", color=colors[i])
plt.title(r"Reward values (sum $f_k$) for different $\alpha$, varying seed")
plt.xlabel(r"Parameter $h = h_k$ for all $k$")
plt.ylabel("reward value")
plt.legend(loc="best")
plt.show()

figure()
for i in range(len(alpha_space)):
    plt.loglog(
        h_space, lastReward[i], label="{:.0e}".format(alpha_space[i]), color=colors[i]
    )
    for j in range(len(h_space)):
        plt.loglog(
            [h_space[j], h_space[j]], [minLast[i][j], maxLast[i][j]], color=colors[i]
        )
        plt.loglog(h_space[j], minLast[i][j], "_", color=colors[i])
        plt.loglog(h_space[j], maxLast[i][j], "_", color=colors[i])
plt.title(r"Reward values (last $f_k$) for different $\alpha$, varying seed")
plt.xlabel(r"Parameter $h = h_k$ for all $k$")
plt.ylabel("reward value")
plt.legend(loc="best")
plt.show()
