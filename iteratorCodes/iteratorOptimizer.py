#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 12:59:35 2023

@author: ckaramanli
@coauthor: phuongmaihuynhpham
"""


import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from scipy.stats import ortho_group
import os.path
import sys

sys.path.append("..")
from quadraticProblem import *

# specify problem parameters:
d_x = 15
alpha = 10 ** (4)
x_bar = np.zeros(d_x)
seed = 3
# set values:
h_k = 10 ** (2)
eta_k = 1
x_init = np.random.normal(size=d_x)
delta_0 = 1
p = int(d_x / 2)
epsilon = 10 ** (-6)


# if ~os.path.isfile('./iterator codes/L_bar.csv'):
#    L_bar = np.tril(np.random.rand(d,d)) # CHECK! Diagonal exists is this fine?
#    L_bar = L_bar/np.linalg.norm(L_bar,ord = 'fro') # normalize the matrix
#    # CHECK! This is using Frobenius norm is this fine?
#    dataframe = pd.DataFrame(L_bar)
#    dataframe.to_csv(r"./iterator codes/L_bar.csv")
# else:
#    df = pd.read_csv('./iterator codes/L_bar.csv')
#    L_bar = df.to_numpy()

# initialize problem:
prob_config = {
    "d_x": d_x,
    "alpha": alpha,
    "x_bar": x_bar,
    "seed": seed,
    "is_stoch": True,
}

prob = quadraticProblem(**prob_config)


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


def TRDFO(total_iter_num=50):
    result_list = []
    x_k = x_init
    delta_k = delta_0
    for k in range(0, total_iter_num):
        # Q = ortho_group.rvs(dim=d)
        # indices = np.random.choice(d, p, replace=False)
        # Q_sample = Q[:,indices]
        Q_sample = np.random.normal(size=(d_x, p))
        Q_sample, _ = np.linalg.qr(Q_sample)
        f_k = prob.obj_func(x_k)
        g_hat_k = prob.g_hat_func(x_k, f_k, h_k, Q_sample, p)
        g_hat_norm = np.linalg.norm(g_hat_k)
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


# plot results:
from pylab import *

result_list = TRDFO()
k_hist = []
f_k_hist = []
b_k_hist = []
for i in range(len(result_list) - 1):
    k_hist.append(result_list[i].k)
    f_k_hist.append(result_list[i].f_k)
    b_k_hist.append(result_list[i].b_k)
figure()
semilogy(f_k_hist)
ylabel("f_k", fontsize=16)
xlabel("k", fontsize=16)
show()

# figure()
# plot(b_k_hist)
# ylabel('b_k',fontsize=16)
# xlabel('k',fontsize=16)
# show()
