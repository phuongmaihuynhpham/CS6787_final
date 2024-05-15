#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 13:09:57 2023

@author: ckaramanli
"""

import numpy as np


class quadraticProblem:
    def __init__(self, **kwargs):
        self.seed = kwargs.get("seed", 3)
        self.d_x = kwargs.get("d_x", 3)
        self.x_bar = kwargs.get("x_bar", np.zeros(self.d_x))
        self.alpha = kwargs.get("alpha", 1)
        self.is_stoch = kwargs.get("is_stoch", True)

        np.random.seed(self.seed)
        # normalize the matrix
        self.L_bar = np.tril(np.random.rand(self.d_x, self.d_x))
        self.L_bar = self.L_bar / np.linalg.norm(self.L_bar, ord="fro")
        # store matrix vector product:
        self.L_bar_x_bar = np.dot(self.L_bar, self.x_bar)

    def obj_func(self, x_k):
        if self.is_stoch == True:
            noise = np.random.normal(0, self.alpha, size=self.d_x)
            y = self.L_bar_x_bar + noise
            return (1 / 2) * (np.linalg.norm(np.dot(x_k, self.L_bar) - y)) ** 2
        else:
            x_diff = x_k - self.x_bar
            return (
                (1 / 2) * self.alpha * (np.linalg.norm(np.dot(x_diff, self.L_bar))) ** 2
            )

    def g_hat_func(self, x_k, f_k, h_k, Q_sample, p):
        g_hat = np.zeros(p)
        for i in range(0, p):
            g_hat[i] = (self.obj_func(x_k + h_k * Q_sample[:, i]) - f_k) / h_k
        return g_hat
