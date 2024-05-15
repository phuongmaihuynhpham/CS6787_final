#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This defines the Watson function
d_x is usually 6 or 9 but can be 15
"""

import numpy as np


class Watsonprob:
    def __init__(self, **kwargs):
        self.seed = kwargs.get("seed", 3)
        self.d_x = kwargs.get("d_x", 3)
        self.x_bar = kwargs.get("x_bar", np.zeros(self.d_x))
        self.alpha = kwargs.get("alpha", 0.001)
        self.is_stoch = kwargs.get("is_stoch", True)

        np.random.seed(self.seed)

    def obj_func(self, x_k):
        y = 0
        fvec = np.zeros(31)
        x = x_k

        for i in range(29):
            div = (i + 1) / 29
            s1 = 0
            dx = 1
            for j in range(1, self.d_x):
                s1 = s1 + j * dx * x[j]
                dx = div * dx
            s2 = 0
            dx = 1
            for j in range(self.d_x):
                s2 = s2 + dx * x[j]
                dx = div * dx
            fvec[i] = s1 - s2 * s2 - 1

        fvec[29] = x[0]
        fvec[30] = x[1] - x[0] * x[0] - 1

        if self.is_stoch == True:
            noise = np.random.normal(0, self.alpha, size=31)

            for i in range(31):
                y = y + (fvec[i] + noise[i]) ** 2

            return y
        else:
            for i in range(31):
                y = y + fvec[i] ** 2

            return y

    def g_hat_func(self, x_k, f_k, h_k, Q_sample, p):
        g_hat = np.zeros(p)
        for i in range(0, p):
            g_hat[i] = (self.obj_func(x_k + h_k * Q_sample[:, i]) - f_k) / h_k
        return g_hat


# a good input:
xx = np.array(
    [
        0.0000,
        1.0000,
        0.0002,
        0.3317,
        0.0003,
        0.1833,
        -0.2293,
        0.4518,
        -0.1011,
        -0.5940,
        0.8590,
        -0.3621,
        -0.0176,
        0.0089,
        0.0264,
    ]
)
