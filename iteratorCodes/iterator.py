#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 12:59:35 2023

@author: phuongmaihuynhpham
"""

import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from scipy.stats import ortho_group
import copy
import os.path
import sys

sys.path.insert(0, "/Users/ckaramanli/Documents/LBL internship/codes/RLAdaptiveOpt")
sys.path.insert(0, "/Users/maihuynh/Desktop/RLAdaptiveOpt/NNCodes")
sys.path.insert(0, "/home/wild/oldcomp/repos/RLAdaptiveOpt/NNCodes")
from quadraticProblem import *
from policyNN import *


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


class iterator:
    def __init__(self, p, T, prob, result_init):
        self.p = p
        self.T = T
        self.prob = prob
        self.result_init = copy.deepcopy(result_init)
        self.result_k = copy.deepcopy(result_init)

    # the iterator function just iterates the algorithm for 1 time step.
    def oneStepIterator(self, h_k=10**-6, eta_k=1):
        # get the info from initial result class:
        x_k = self.result_k.x_k
        delta_k = self.result_k.delta_k

        Q_sample = np.random.normal(size=(self.prob.d_x, self.p))
        Q_sample, _ = np.linalg.qr(Q_sample)
        f_k = self.prob.obj_func(x_k)
        g_hat_k = self.prob.g_hat_func(x_k, f_k, h_k, Q_sample, self.p)
        g_hat_norm = np.linalg.norm(g_hat_k)
        s_prop = ((-1) * g_hat_k / g_hat_norm) * delta_k
        x_prop = np.dot(Q_sample, s_prop)
        rho_k = (f_k - self.prob.obj_func(x_prop)) / (g_hat_norm * delta_k)
        b_k = 0
        if rho_k >= eta_k:
            x_k = x_prop
            b_k = 1
        else:
            delta_k /= 2
        # record keeping:
        k = self.result_k.k + 1
        output_result = result(h_k, eta_k, k, f_k, rho_k, delta_k, b_k, x_k, g_hat_k)
        return output_result

    def dataGenerator(self, net, num_rows):
        # first calls the iterator T times to create one instance of data,
        # then calls num_rows-1 times more to generate num_rows rows of data in total.
        # first part:
        # TODO!! We always start from the same initial result struct:
        self.result_k = copy.deepcopy(self.result_init)
        # TODO!! Fix the seed to get deterministic results:
        np.random.seed(self.prob.seed)
        T = self.T
        result_list = []
        result_list.append(self.result_k)
        for k in range(0, T):
            # Run default h_k,eta_k values for the initialization
            self.result_k = self.oneStepIterator(self.result_k.h_k, self.result_k.eta_k)
            result_list.append(self.result_k)
        d_data = T * (6 + self.prob.d_x + self.p) + 1
        first_row = np.zeros(d_data - T * self.prob.d_x - T * self.p)
        x_k_stack = np.zeros(0)
        g_hat_k_stack = np.zeros(0)
        first_row[0] = self.result_k.k
        index = 0
        for i in range(0, T):
            first_row[1 + i + index * T] = result_list[i].h_k
            index += 1
            first_row[1 + i + index * T] = result_list[i].eta_k
            index += 1
            first_row[1 + i + index * T] = result_list[i].f_k
            index += 1
            first_row[1 + i + index * T] = result_list[i].rho_k
            index += 1
            first_row[1 + i + index * T] = result_list[i].delta_k
            index += 1
            first_row[1 + i + index * T] = result_list[i].b_k
            index = 0
            x_k_stack = np.hstack((x_k_stack, result_list[i].x_k))
            g_hat_k_stack = np.hstack((g_hat_k_stack, result_list[i].g_hat_k))
        first_row = np.hstack((first_row, x_k_stack, g_hat_k_stack))
        # second part:
        data = np.zeros((num_rows, d_data))
        data[0, :] = first_row
        for row in range(0, num_rows - 1):
            # Get h_k,eta_k info using NN
            inputs = data[row, :]
            h_k, eta_k = net.getActions(inputs)
            self.result_k = self.oneStepIterator(h_k, eta_k)
            data[row + 1, 0] = self.result_k.k
            data[row + 1, 1:T] = data[row, 2 : T + 1]
            data[row + 1, T] = self.result_k.h_k
            data[row + 1, T + 1 : 2 * T] = data[row, T + 2 : 2 * T + 1]
            data[row + 1, 2 * T] = self.result_k.eta_k
            data[row + 1, 2 * T + 1 : 3 * T] = data[row, 2 * T + 2 : 3 * T + 1]
            data[row + 1, 3 * T] = self.result_k.f_k
            data[row + 1, 3 * T + 1 : 4 * T] = data[row, 3 * T + 2 : 4 * T + 1]
            data[row + 1, 4 * T] = self.result_k.rho_k
            data[row + 1, 4 * T + 1 : 5 * T] = data[row, 4 * T + 2 : 5 * T + 1]
            data[row + 1, 5 * T] = self.result_k.delta_k
            data[row + 1, 5 * T + 1 : 6 * T] = data[row, 5 * T + 2 : 6 * T + 1]
            data[row + 1, 6 * T] = self.result_k.b_k
            x_k_stack = np.hstack((x_k_stack[self.prob.d_x :], self.result_k.x_k))
            g_hat_k_stack = np.hstack((g_hat_k_stack[self.p :], self.result_k.g_hat_k))
            stack = np.hstack((x_k_stack, g_hat_k_stack))
            data[row + 1, 6 * T + 1 :] = stack

        return data

    def resultsToArray(self, result_list):
        # input: result_list which consists of T number of results
        # output: one array of input for the NN
        d_data = self.T * (6 + self.prob.d_x + self.p) + 1
        row = np.zeros(d_data - self.T * self.prob.d_x - self.T * self.p)
        x_k_stack = np.zeros(0)
        g_hat_k_stack = np.zeros(0)
        row[0] = self.result_k.k
        index = 0
        for i in range(0, self.T):
            row[1 + i + index * T] = result_list[i].h_k
            index += 1
            row[1 + i + index * T] = result_list[i].eta_k
            index += 1
            row[1 + i + index * T] = result_list[i].f_k
            index += 1
            row[1 + i + index * T] = result_list[i].rho_k
            index += 1
            row[1 + i + index * T] = result_list[i].delta_k
            index += 1
            row[1 + i + index * T] = result_list[i].b_k
            index = 0
            x_k_stack = np.hstack((x_k_stack, result_list[i].x_k))
            g_hat_k_stack = np.hstack((g_hat_k_stack, result_list[i].g_hat_k))
        row = np.hstack((row, x_k_stack, g_hat_k_stack))
        return row


# # test:
# # specify problem parameters:
# d_x = 5
# alpha = 1
# x_bar = np.zeros(d_x)
# seed = 3
# # set values:
# h_k = 10**(-6)
# eta_k = 1
# x_init = np.random.normal(size=d_x)
# delta_k = 1
# p = int(d_x/2)
# epsilon = 10**(-6)
# T = 5


# #if ~os.path.isfile('./iterator codes/L_bar.csv'):
# #    L_bar = np.tril(np.random.rand(d,d)) # CHECK! Diagonal exists is this fine?
# #    L_bar = L_bar/np.linalg.norm(L_bar,ord = 'fro') # normalize the matrix
# #    # CHECK! This is using Frobenius norm is this fine?
# #    dataframe = pd.DataFrame(L_bar)
# #    dataframe.to_csv(r"./iterator codes/L_bar.csv")
# #else:
# #    df = pd.read_csv('./iterator codes/L_bar.csv')
# #    L_bar = df.to_numpy()

# # initialize problem:
# prob_config = {
#     'd_x'   : d_x,
#     'alpha' : alpha,
#     'x_bar' : x_bar,
#     'seed'  : seed,
#     'is_stoch' : True
# }

# prob = quadraticProblem(**prob_config)
# # evaluate initial values:
# k=0;rho_k=0;b_k = 0
# x_k = x_init
# f_k = prob.obj_func(x_k)
# Q_sample = np.random.normal(size=(d_x, p))
# Q_sample,_ = np.linalg.qr(Q_sample)
# g_hat_k = prob.g_hat_func(x_k, f_k,h_k, Q_sample,p)
# # test:
# result_init = result(h_k,eta_k,k,f_k,rho_k,delta_k,b_k,x_k,g_hat_k)
# iterator_object = iterator(p,T,prob,result_init)
# tau = 100
# dataset = iterator_object.dataGenerator(tau)
