#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 12:59:35 2023

@author: ckaramanli
@coauthor: phuongmaihuynhpham
"""


import numpy as np
import matplotlib.pyplot as plt
import sys
from pylab import *

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
x_bar = np.zeros(d_x)

# set values:
# h_space = np.linspace(10**-20, 10**20, 102)  # change the last param to change # h's
# h_space = [10**-20, 10**-10, 10**0, 10**10, 10**20, 10**30]
h_space = [10**i for i in range(-20, 22, 2)]
# h_space = [10**i for i in range(-40,35,5)]
# h_space = [10**-10, 10**-9, 10**-8, 10**-7, 10**-6, 1, 10, 100, 1000] # 10**-10 -> 10**3
eta_space = [2**-3, 2**-2, 2**-1, 2**0, 2**1]  # for now
# h_k = 10 ** (-6)
# eta_k = 1
seed = 3
np.random.seed(seed)
x_init = np.random.normal(size=d_x)
delta_0 = 1
p = int(d_x / 2)
epsilon = 10 ** (-16)


# if ~os.path.isfile('./iterator codes/L_bar.csv'):
#    L_bar = np.tril(np.random.rand(d,d)) # CHECK! Diagonal exists is this fine?
#    L_bar = L_bar/np.linalg.norm(L_bar,ord = 'fro') # normalize the matrix
#    # CHECK! This is using Frobenius norm is this fine?
#    dataframe = pd.DataFrame(L_bar)
#    dataframe.to_csv(r"./iterator codes/L_bar.csv")
# else:
#    df = pd.read_csv('./iterator codes/L_bar.csv')
#    L_bar = df.to_numpy()
rewards_sum = []
rewards_last = []
rewards_best = []


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
        # Q = ortho_group.rvs(dim=d)
        # indices = np.random.choice(d, p, replace=False)
        # Q_sample = Q[:,indices]
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


for alpha in alpha_space:
    # initialize problem:
    seed = 3
    prob_config = {
        "d_x": d_x,
        "alpha": alpha,
        "x_bar": x_bar,
        "seed": seed,
        "is_stoch": True,
    }

    prob = quadraticProblem(**prob_config)

    f_k = []
    sum_fk = []
    last_fk = []
    best_fk = []
    b_k = []

    # fig = plt.figure()
    # gs = fig.add_gridspec(len(h_space)//3, 3)
    # axs = gs.subplots(sharex=True, sharey=True)
    # fig.suptitle('f_k with diff h')

    for i in range(len(h_space)):
        f_k_hist, b_k_hist = exper_h_eta(prob, h_space[i])
        f_k.append(f_k_hist)
        b_k.append(b_k_hist)
        last_fk.append(1 / (f_k_hist[-1] / f_k_hist[0]))
        sum_fk.append(1 / (sum(f_k_hist) / f_k_hist[0]))
        # best_fk.append(min(f_k_hist)/f_k_hist[0])
    #     axs[i//3, i%3].semilogy(f_k_hist)
    #     axs[i//3, i%3].set_title('h = {:.0e}'.format(h_space[i]))
    # plt.show()

    # figure()
    # for f_k_hist in f_k:
    #     plt.semilogy(f_k_hist)
    # title('f_k with diff h')
    # show()

    # figure()
    # plt.loglog(h_space,sum_fk)
    # title('Sum of best f_k over 50 iters')
    # show()

    rewards_sum.append(sum_fk)
    rewards_last.append(last_fk)
    # rewards_best.append(best_fk)

# reward func: sum of best fk for all k (k=50)
# sum_fk = np.matrix(rewards_sum)
# avg_reward_sum = sum_fk.sum(axis = 0)/len(alpha_space)
figure()
# plt.loglog(h_space, avg_reward_sum.transpose(), label = 'Average')
for i in range(len(rewards_sum)):
    plt.loglog(
        h_space, rewards_sum[i], label="{:.0e}".format(alpha_space[i]), linewidth="3"
    )
plt.title(
    r"Reward values (sum $f_k$) for different $\alpha$", fontsize="17", weight="bold"
)
plt.xlabel(r"Parameter $h = h_k$ for all $k$", fontsize="17")
plt.ylabel("reward value", fontsize="17")
plt.legend(loc="best", ncol=2, fontsize="15")
plt.show()

# reward func: last fk
# last_fk = np.matrix(rewards_last)
# avg_reward_last = last_fk.sum(axis = 0)/len(alpha_space)
figure()
# plt.loglog(h_space, avg_reward_last.transpose(), label = 'Average')
for i in range(len(rewards_last)):
    plt.loglog(
        h_space, rewards_last[i], label="{:.0e}".format(alpha_space[i]), linewidth="3"
    )
plt.title(
    r"Reward values (last $f_k$) for different $\alpha$", fontsize="17", weight="bold"
)
plt.xlabel(r"Parameter $h = h_k$ for all $k$", fontsize="17")
plt.ylabel("reward value", fontsize="17")
plt.legend(loc="best", fontsize="15")
plt.show()

# # reward func: best fk
# best_fk = np.matrix(rewards_best)
# avg_reward_best = best_fk.sum(axis = 0)/len(alpha_space)
# figure()
# plt.loglog(h_space, avg_reward_best.transpose(), label = 'Avg over 5 probs')
# for i in range(len(rewards_best)):
#     plt.loglog(h_space, rewards_best[i], label = '{:.0e}'.format(alpha_space[i]))
#     title(r'Reward funcs (best fk) for diff $\alpha$')
# plt.legend(loc = 'best', ncol = 2)
# show()
