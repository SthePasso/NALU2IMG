"""The Model Implementation of Neural Arithmetic Logical Unit"""
import argparse

import numpy as np
import matplotlib.pyplot as plt

import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn


class NAC(nn.Block):
    def __init__(self, in_units, units):
        super(NAC, self).__init__()
        self.W_hat = self.params.get('W_hat', shape=(in_units, units))
        self.M_hat = self.params.get('M_hat', shape=(in_units, units))

    def forward(self, x):
        if len(x.shape) > 2:  # If input is 2D, flatten it
            x = nd.flatten(x)

        W = nd.tanh(self.W_hat.data()) * nd.sigmoid(self.M_hat.data())
        return nd.dot(x, W)

class NALU(nn.Block):
    def __init__(self, in_units, units):
        super(NALU, self).__init__()

        self.W0_hat = self.params.get('W0_hat', shape=(in_units, units))
        self.M0_hat = self.params.get('M0_hat', shape=(in_units, units))
        self.dependent_G = True  # whether the gate is dependent on the input

        if self.dependent_G:
            self.G = self.params.get('G', shape=(in_units, units))
        else:
            self.G = self.params.get('G', shape=(units,))

    def forward(self, x):
        if len(x.shape) > 2:  # If input is 2D, flatten it
            x = nd.flatten(x)

        if self.dependent_G:
            g = nd.sigmoid(nd.dot(x, self.G.data()))
        else:
            g = nd.sigmoid(self.G.data())

        W0 = nd.tanh(self.W0_hat.data()) * nd.sigmoid(self.M0_hat.data())
        a = nd.dot(x, W0)
        m = nd.exp(nd.dot(nd.log(nd.abs(x) + 1e-10), W0))
        y = g * a + (1 - g) * m

        return y

class NALU2M(nn.Block): #replicate state of art
    def __init__(self, in_units, units):
        super(NALU2M, self).__init__()

        self.W0_hat = self.params.get('W0_hat', shape=(in_units, units))
        self.M0_hat = self.params.get('M0_hat', shape=(in_units, units))
        self.W1_hat = self.params.get('W1_hat', shape=(in_units, units))
        self.M1_hat = self.params.get('M1_hat', shape=(in_units, units))

        self.dependent_G = True  # whether the gate is dependent on the input

        if self.dependent_G:
            self.G = self.params.get('G', shape=(in_units, units))
        else:
            self.G = self.params.get('G', shape=(units,))

    def forward(self, x):
        if self.dependent_G:
            g = nd.sigmoid(nd.dot(x, self.G.data()))
        else:
            g = nd.sigmoid(self.G.data())

        W0 = nd.tanh(self.W0_hat.data()) * nd.sigmoid(self.M0_hat.data())
        W1 = nd.tanh(self.W1_hat.data()) * nd.sigmoid(self.M1_hat.data())
        a = nd.dot(x, W0)
        m = nd.exp(nd.dot(nd.log(nd.abs(x) + 1e-10), W1))
        y = g * a + (1 - g) * m

        return y

class NALUIG(nn.Block):
    def __init__(self, in_units, units):
        super(NALUIG, self).__init__()

        self.W0_hat = self.params.get('W0_hat', shape=(in_units, units), init=mx.init.Xavier())
        self.M0_hat = self.params.get('M0_hat', shape=(in_units, units), init=mx.init.Xavier())
        self.dependent_G = False  # whether the gate is dependent on the input

        if self.dependent_G:
            self.G = self.params.get('G', shape=(in_units, units), init=mx.init.Xavier())
        else:
            self.G = self.params.get('G', shape=(units,), init=mx.init.Normal())  # Adjust initializer for 1D

    def forward(self, x):
        if len(x.shape) > 2:  # If input is 2D, flatten it
            x = nd.flatten(x)

        if self.dependent_G:
            g = nd.sigmoid(nd.dot(x, self.G.data()))
        else:
            g = nd.sigmoid(self.G.data())

        W0 = nd.tanh(self.W0_hat.data()) * nd.sigmoid(self.M0_hat.data())
        a = nd.dot(x, W0)
        m = nd.exp(nd.dot(nd.log(nd.abs(x) + 1e-10), W0))
        y = g * a + (1 - g) * m
        #print("NALUIG: ",y.shape)
        return y


class NALU2MIG(nn.Block):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(NALU2MIG, self).__init__()

        # Define convolutional weight matrices for W0 and W1
        self.W0_hat = self.params.get('W0_hat', shape=(out_channels, in_channels, kernel_size, kernel_size), init=mx.init.Xavier())
        self.M0_hat = self.params.get('M0_hat', shape=(out_channels, in_channels, kernel_size, kernel_size), init=mx.init.Xavier())
        self.W1_hat = self.params.get('W1_hat', shape=(out_channels, in_channels, kernel_size, kernel_size), init=mx.init.Xavier())
        self.M1_hat = self.params.get('M1_hat', shape=(out_channels, in_channels, kernel_size, kernel_size), init=mx.init.Xavier())

        self.dependent_G = False  # Gate not dependent on input by default

        if self.dependent_G:
            self.G = self.params.get('G', shape=(out_channels, in_channels, kernel_size, kernel_size), init=mx.init.Xavier())
        else:
            self.G = self.params.get('G', shape=(out_channels,), init=mx.init.Normal())  # single value for the entire channel

    def forward(self, x):
        if len(x.shape) == 2:  # Reshape if input is flattened (2D)
            batch_size = x.shape[0]
            x = x.reshape((batch_size, -1, 1, 1))  # Reshape to 4D for convolution

        # Define the gating mechanism
        if self.dependent_G:
            g = nd.sigmoid(nd.Convolution(data=x, weight=self.G.data()))
        else:
            g = nd.sigmoid(self.G.data())  # use a single gate value

        # Perform operations for W0 and W1 using convolution instead of dot product
        W0 = nd.tanh(self.W0_hat.data()) * nd.sigmoid(self.M0_hat.data())
        W1 = nd.tanh(self.W1_hat.data()) * nd.sigmoid(self.M1_hat.data())

        # Apply convolution instead of dot product
        a = nd.Convolution(data=x, weight=W0, num_filter=W0.shape[0], kernel=(3, 3), pad=(1, 1), no_bias=True)
        m = nd.exp(nd.Convolution(data=nd.log(nd.abs(x) + 1e-10), weight=W1, num_filter=W1.shape[0], kernel=(3, 3), pad=(1, 1), no_bias=True))

        y = g * a + (1 - g) * m

        return y

