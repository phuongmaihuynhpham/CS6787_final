#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 22:36:20 2023

@author: ckaramanli
"""

# policy neural network code
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

# from torchviz import make_dot

torch.autograd.set_detect_anomaly(True)

# fully connected:
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(32 * 32 * 3, 500)
#         self.fc2 = nn.Linear(500, 10)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         return self.fc2(x)


# not fully connected:
class FullNet(nn.Module):
    def __init__(self, d_x, p, T, h_space, eta_space):
        torch.set_default_dtype(torch.float64)
        super(FullNet, self).__init__()
        self.d_x = d_x
        self.p = p
        self.T = T
        self.d_input = T * (d_x + p + 6) + 1
        self.h_space = h_space
        self.eta_space = eta_space
        self.d_h_space = len(h_space)
        self.d_eta_space = len(eta_space)
        self.l11 = nn.Linear(self.d_input, self.T)
        self.l21 = nn.Linear(self.T, self.d_h_space)
        self.l22 = nn.Linear(self.T, self.d_eta_space)
        self.output = torch.tensor([])

    def forward(self, x):
        # input x is torch
        # use activation functions:
        x = F.relu(self.l11(x))

        x1 = self.l21(x)
        x2 = self.l22(x)
        x = torch.vstack((x1, x2))
        x = torch.softmax(x, 1)
        x1 = 10 ** torch.matmul(x[0, :], torch.log10(torch.tensor(self.h_space)))
        x2 = torch.matmul(x[1, :], torch.tensor(self.eta_space))
        x = torch.vstack((x1, x2))
        return x

    def getActions(self, inputs):
        # input is 1d numpy array
        inputs = torch.from_numpy(inputs)
        outputs = self.forward(inputs)
        # h_index = int(torch.argmax(outputs, dim=1)[0])
        # eta_index = int(torch.argmax(outputs, dim=1)[1])
        # h_k = self.h_space[h_index]
        # eta_k = self.eta_space[eta_index]
        # new h_k and eta_k calculation:
        h_k = outputs[0].item()
        eta_k = outputs[1].item()
        return h_k, eta_k


class Net(nn.Module):
    def __init__(self, d_x, p, T, d_layer1, h_space, eta_space):
        torch.set_default_dtype(torch.float64)
        super(Net, self).__init__()
        self.d_x = d_x
        self.p = p
        self.T = T
        self.d_layer1 = d_layer1
        self.h_space = h_space
        self.eta_space = eta_space
        self.d_h_space = len(h_space)
        self.d_eta_space = len(eta_space)
        self.l11 = nn.Linear(
            self.d_layer1, 1
        )  # TODO!! Try to generalize it to a general T
        self.l12 = nn.Linear(self.d_layer1, 1)
        self.l13 = nn.Linear(self.d_layer1, 1)
        self.l14 = nn.Linear(self.d_layer1, 1)
        self.l15 = nn.Linear(self.d_layer1, 1)
        self.l21 = nn.Linear(self.T, self.d_h_space)
        self.l22 = nn.Linear(self.T, self.d_eta_space)
        self.output = torch.tensor([])

    def dataPrep(self, x):
        # input x is 1d numpy array
        d_x = self.d_x
        p = self.p
        T = self.T
        # create x1,x2,x3,x4,x5:
        x1 = np.hstack(
            (
                x[[0, 1, T + 1, 2 * T + 1, 3 * T + 1, 4 * T + 1, 5 * T + 1]],
                x[6 * T + 1 : 6 * T + 1 + d_x],
                x[6 * T + 1 + T * d_x : 6 * T + 1 + T * d_x + p],
            )
        )
        x2 = np.hstack(
            (
                x[[0, 2, T + 2, 2 * T + 2, 3 * T + 2, 4 * T + 2, 5 * T + 2]],
                x[6 * T + 1 * d_x + 1 : 6 * T + 1 + 2 * d_x],
                x[6 * T + 1 + T * d_x + p : 6 * T + 1 + T * d_x + 2 * p],
            )
        )
        x3 = np.hstack(
            (
                x[[0, 3, T + 3, 2 * T + 3, 3 * T + 3, 4 * T + 3, 5 * T + 3]],
                x[6 * T + 2 * d_x + 1 : 6 * T + 1 + 3 * d_x],
                x[6 * T + 1 + T * d_x + 2 * p : 6 * T + 1 + T * d_x + 3 * p],
            )
        )
        x4 = np.hstack(
            (
                x[[0, 4, T + 4, 2 * T + 4, 3 * T + 4, 4 * T + 4, 5 * T + 4]],
                x[6 * T + 3 * d_x + 1 : 6 * T + 1 + 4 * d_x],
                x[6 * T + 1 + T * d_x + 3 * p : 6 * T + 1 + T * d_x + 4 * p],
            )
        )
        x5 = np.hstack(
            (
                x[[0, 5, T + 5, 2 * T + 5, 3 * T + 5, 4 * T + 5, 5 * T + 5]],
                x[6 * T + 4 * d_x + 1 : 6 * T + 1 + 5 * d_x],
                x[6 * T + 1 + T * d_x + 4 * p : 6 * T + 1 + T * d_x + 5 * p],
            )
        )

        x1 = torch.from_numpy(x1)  # TODO!! Float results in data loss
        x2 = torch.from_numpy(x2)
        x3 = torch.from_numpy(x3)
        x4 = torch.from_numpy(x4)
        x5 = torch.from_numpy(x5)

        xo = torch.vstack((x1, x2, x3, x4, x5))
        return xo

    def forward(self, x):
        # input x is torch
        # use activation functions:
        x1 = F.relu(self.l11(x[0, :]))
        x2 = F.relu(self.l12(x[1, :]))
        x3 = F.relu(self.l13(x[2, :]))
        x4 = F.relu(self.l14(x[3, :]))
        x5 = F.relu(self.l15(x[4, :]))
        x = torch.cat((x1, x2, x3, x4, x5), dim=0)

        x1 = self.l21(x)
        x2 = self.l22(x)
        x = torch.vstack((x1, x2))
        x = torch.softmax(x, 1)
        self.outputs = x
        return x

    def getActions(self, inputs):
        # input is 1d numpy array
        inputs = self.dataPrep(inputs)
        outputs = self.forward(inputs)
        h_index = int(torch.argmax(outputs, dim=1)[0])
        eta_index = int(torch.argmax(outputs, dim=1)[1])
        h_k = self.h_space[h_index]
        eta_k = self.eta_space[eta_index]
        return h_k, eta_k


def flattenWeights(net):
    flat_w = torch.tensor([0])
    for param in net.parameters():
        flat_w = torch.cat((flat_w, param.flatten()))
        flat_w = flat_w.flatten()
    flat_w = flat_w[1:]
    return flat_w


def updateNN(net, new_flat_w):
    i = 0
    ## IMPORTANT!! This operation changes the original network parameters
    # update weights in NN
    for param in net.parameters():
        if len(param.data.shape) > 1:
            nr = param.data.shape[0]
            nc = param.data.shape[1]
            for r in range(nr):
                param.data[r, :] = new_flat_w[i : i + nc]
                i += nc
        else:
            vec_len = param.data.shape[0]
            param.data = new_flat_w[i : i + vec_len]
            i += vec_len
    return net


def fixedPolicyNN(net, fixed_h_index, fixed_eta_index):
    # assume 2*5 tensor for the fixed_policy
    # row 1 corr. to h and row 2 corr. to eta.
    # change self.l21 and self.l22 values of the network to get fixed h_policy.
    eps = 10 ** (-6)
    i = 0
    for name, param in net.named_parameters():
        if name == "l21.bias":
            if fixed_h_index >= 0:
                param.data = torch.zeros(param.data.shape[0])
                param.data[fixed_h_index] = eps
        if name == "l22.bias":
            if fixed_eta_index >= 0:
                param.data = torch.zeros(param.data.shape[0])
                param.data[fixed_eta_index] = eps
        elif name == "l21.weight":
            param.data = torch.ones(param.data.shape[0], param.data.shape[1]) * eps
            if fixed_h_index >= 0:
                param.data[fixed_h_index, fixed_h_index] = 1
        elif name == "l22.weight":
            param.data = torch.ones(param.data.shape[0], param.data.shape[1]) * eps
            if fixed_eta_index >= 0:
                param.data[fixed_eta_index, fixed_eta_index] = 1
    return net


# initialize problem:
# prob_config = {
#     'd_x'   : d_x,
#     'alpha' : alpha,
#     'x_bar' : x_bar,
#     'seed'  : seed,
#     'is_stoch' : True
# }

# prob = quadraticProblem(**prob_config)
# # evaluate initial values:
# k=0;rho_k=0;b_k = 0;
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


# # testing the NN:
# # define total loss function:
# def totalLoss(ground_truth_outputs_tensor,outputs_tensor):
#     return torch.norm(ground_truth_outputs_tensor - outputs_tensor)

# torch.manual_seed(4)
# net = Net(d_x,p,T,d_layer1,h_space,eta_space)
# theta = torch.tensor([0])
# for param in net.parameters():
#     theta = torch.cat((theta,param.flatten()))
# theta_star = theta

# # get the labels:
# ground_truth_outputs_tensor = torch.zeros(len(dataset), 2,d_h_space)
# outputs_tensor = torch.zeros(len(dataset), 2,d_h_space)
# ground_truth_labels_tensor = torch.zeros(len(dataset), 2)
# for i in range(0,len(dataset)):
#     inputs = dataset[i,:]
#     inputs = net.dataPrep(inputs)
#     outputs = net(inputs)
#     ground_truth_outputs_tensor[i,:,:] = outputs
#     ground_truth_labels_tensor[i,:] = torch.argmax(outputs,1)

# # create the net again with different random weights:
# net = Net(d_x,p,T,d_layer1,h_space,eta_space)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(net.parameters(), lr=3e-4)
# tL_list = []
# for epoch in range(50): # loop over the dataset multiple times
#     for i in range(0,len(dataset)):
#         # Get the inputs
#         inputs = dataset[i,:]
#         ground_truth_outputs = ground_truth_outputs_tensor[i,:,:]
#         # Zero the parameter gradients
#         optimizer.zero_grad()
#         # Forward + backward + optimize
#         inputs = net.dataPrep(inputs)
#         outputs = net(inputs)
#         outputs_tensor[i,:,:] = outputs
#         loss = criterion(outputs, ground_truth_outputs)
#         loss.backward(retain_graph=True)
#         optimizer.step()
#     tL = totalLoss(ground_truth_outputs_tensor,outputs_tensor).item()
#     tL_list.append(tL)
#     print("epoch: ",epoch,", tL = ",tL)

# plt.plot(tL_list)

# Visualize the NN
# nn_visual = make_dot(outputs.mean(), params=dict(net.named_parameters()))
# nn_visual.format = 'jpg'
# nn_visual.render(filename='nonCompactVisual', directory='/Users/maihuynh/Desktop/RLAdaptiveOpt/NNCodes/nnVisual/nonCompact',view = True, format='jpg')

# Plot total loss
# figure()
# plot(tL_list)
# show()
