import pandas as pd
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F




def heaviside_theta(x, mu, r):
    """Heaviside theta function with parameters mu and r.

    Args:
        x (torch.Tensor): Input tensor.
        mu (float): Center of the function.
        r (float): Width of the function.
    
    Returns:
        torch.Tensor: Output tensor.
    """
    x = x - mu
    return (torch.clamp(x + r, 0, r) - torch.clamp(x, 0, r)) / r

def _linear_interpolation(x, X, Y):
    """Linear interpolation function.

    Note: This function is used to apply the linear interpolation to one element of the input tensor.
    For vectorized operations, use the linear_interpolation function.

    Args:
        x (torch.Tensor): Input tensor.
        X (torch.Tensor): X values.
        Y (torch.Tensor): Y values.

    Returns:
        torch.Tensor: Output tensor.
    """
    mu = X
    r = X[1] - X[0]
    F = torch.vmap(heaviside_theta, in_dims=(None, 0, None))
    y = F(x, mu, r).reshape(-1) * Y
    return y.sum()

def linear_interpolation(x, X, Y):
    """Linear interpolation function.

    Args:
        x (torch.Tensor): Input tensor.
        X (torch.Tensor): X values.
        Y (torch.Tensor): Y values.

    Returns:
        torch.Tensor: Output tensor.
    """
    shape = x.shape
    x = x.reshape(-1)
    return torch.vmap(_linear_interpolation, in_dims=(-1, None, None), out_dims=-1)(x, X, Y).reshape(shape)


def phi(x, w1, w2, b1, b2, n_sin, device):
    omega = (2 ** torch.arange(0, n_sin)).float().reshape(-1, 1).to(device)
    omega_x = F.linear(x, omega, bias=None)
    x = torch.cat([x, torch.sin(omega_x), torch.cos(omega_x)], dim=-1)
    
    x = F.linear(x, w1, bias=b1)
    x = F.silu(x)
    x = F.linear(x, w2, bias=b2)
    return x


class KANLayer(nn.Module):
    def __init__(self, dim_in, dim_out, device, fcn_hidden=32, fcn_n_sin=3):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(dim_in, dim_out, fcn_hidden, 1 + fcn_n_sin * 2))
        self.W2 = nn.Parameter(torch.randn(dim_in, dim_out, 1, fcn_hidden))
        self.B1 = nn.Parameter(torch.randn(dim_in, dim_out, fcn_hidden))
        self.B2 = nn.Parameter(torch.randn(dim_in, dim_out, 1))

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.fcn_hidden = fcn_hidden
        self.fcn_n_sin = torch.tensor(fcn_n_sin).long()
        self.device = device

        self.init_parameters()
        self.to(device)  # Move the model parameters to the specified device
    
    def init_parameters(self):
        nn.init.xavier_normal_(self.W1)
        nn.init.xavier_normal_(self.W2)
        nn.init.zeros_(self.B1)
        nn.init.zeros_(self.B2)
    
    def map(self, x):
        F = torch.vmap(
            torch.vmap(phi, (None, 0, 0, 0, 0, None, None), 0),
            (0, 0, 0, 0, 0, None, None), 0
        )
        return F(x.unsqueeze(-1), self.W1, self.W2, self.B1, self.B2, self.fcn_n_sin, self.device).squeeze(-1)

    def forward(self, x):
        x = x.to(self.device)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        batch, dim_in = x.shape
        assert dim_in == self.dim_in

        batch_f = torch.vmap(self.map, 0, 0)
        phis = batch_f(x)
        return phis.sum(dim=1)
    
    def take_function(self, i, j):
        def activation(x):
            return phi(x, self.W1[i, j], self.W2[i, j], self.B1[i, j], self.B2[i, j], self.fcn_n_sin, self.device)
        return activation

def smooth_penalty(model):
    p = 0
    if isinstance(model, KANInterpoLayer):
        dx = model.X[1] - model.X[0]
        grad = model.Y[:, :, 1:] - model.Y[:, :, :-1]
        return torch.norm(grad, 2) / dx

    for layer in model:
        if isinstance(layer, KANInterpoLayer):
            dx = layer.X[1] - layer.X[0]
            grad = layer.Y[:, :, 1:] - layer.Y[:, :, :-1]
            p += torch.norm(grad, 2) / dx
    return p


class GLU(nn.Module):
    #Gated Linear Unit
    def __init__(self, input_size, fcn_hidden, fcn_n_sin, device):
        super(GLU, self).__init__()
        
        self.fc1 = nn.Linear(input_size, input_size) #KANLayer(input_size, input_size, device, fcn_hidden, fcn_n_sin)
        self.fc2 = nn.Linear(input_size, input_size) #KANLayer(input_size, input_size, device, fcn_hidden, fcn_n_sin)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        sig = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return torch.mul(sig, x)

    

class GRN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout, device, fcn_hidden, fcn_n_sin, context_size=None):
        super(GRN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.context_size = context_size
        self.dropout = dropout

        self.fc1 = KANLayer(self.input_size, self.hidden_size, device, fcn_hidden, fcn_n_sin)
        self.elu1 = nn.ELU()

        self.fc2 = KANLayer(self.hidden_size, self.output_size, device, fcn_hidden, fcn_n_sin)
        self.elu2 = nn.ELU()

        self.dropout = nn.Dropout(self.dropout)
        self.bn = nn.BatchNorm1d(self.output_size)
        self.gate = GLU(self.output_size, fcn_hidden, fcn_n_sin, device)  

        if self.input_size != self.output_size:
            self.skip_layer = KANLayer(self.input_size, self.output_size, device)
        if self.context_size is not None:
            self.context = KANLayer(self.context_size, self.hidden_size, device)

    def forward(self, x, context=None):
        if self.input_size != self.output_size:
            residual = self.skip_layer(x)
        else:
            residual = x
        x = self.fc1(x)
        if context is not None:
            context = self.context(context)
            x = x + context

        x = self.elu1(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.gate(x)
        x = x + residual
        x = self.bn(x)

        return x

class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_size, num_cont_inputs, hidden_size, dropout, device, fcn_hidden, fcn_n_sin, context = None):        
        super(VariableSelectionNetwork, self).__init__()
        
        self.input_size = input_size
        self.num_cont_inputs = num_cont_inputs
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.context = context
        self.fcn_hidden = fcn_hidden
        self.fcn_n_sin = fcn_n_sin
        self.device = device
        
        #self.flattened_grn = GRN(self.num_cont_inputs * self.input_size, self.hidden_size, self.num_cont_inputs, self.dropout, self.device, self.fcn_hidden, self.fcn_n_sin)
        self.flattened_grn = KANLayer(self.num_cont_inputs * self.input_size, self.hidden_size,self.device, self.fcn_hidden, self.fcn_n_sin)
        
        self.single_variable_grns = nn.ModuleList()
        for i in range(self.num_cont_inputs):
            self.single_variable_grns.append(KANLayer(self.input_size, self.hidden_size, self.device, self.fcn_hidden, self.fcn_n_sin))
            #self.single_variable_grns.append(GRN(self.input_size, self.hidden_size, self.hidden_size, self.dropout, self.device, self.fcn_hidden, self.fcn_n_sin,))
        self.softmax = nn.Softmax()

    def forward(self, embedding, context=None):
        if context is not None:
            sparse_weights = self.flattened_grn(embedding, context)
        else:
            sparse_weights = self.flattened_grn(embedding)

        sparse_weights = self.softmax(sparse_weights).unsqueeze(1)
        var_outputs = []
        for i in range(self.num_cont_inputs):
            var_outputs.append(self.single_variable_grns[i](embedding[:, (i * self.input_size) : (i + 1) * self.input_size]))

        var_outputs = torch.stack(var_outputs, axis=-1)
        outputs = var_outputs * sparse_weights
        
        outputs = outputs.sum(axis=-1)
        return outputs, sparse_weights.squeeze(1).mean(0)
