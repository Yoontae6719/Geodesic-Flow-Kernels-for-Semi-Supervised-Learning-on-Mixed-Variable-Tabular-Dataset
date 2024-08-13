import pandas as pd
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class GLU(nn.Module):
    #Gated Linear Unit
    def __init__(self, input_size):
        super(GLU, self).__init__()
        
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        
        sig = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return torch.mul(sig, x)
    
class GRN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout, context_size = None):
        super(GRN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.context_size = context_size
        self.dropout = dropout
        
        
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.elu1 = nn.ELU()
        
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.elu2 = nn.ELU()
        
        self.dropout = nn.Dropout(self.dropout)
        
        self.bn = nn.BatchNorm1d(self.output_size)
        
        self.gate = GLU(self.output_size)
        
        if self.input_size!=self.output_size:
            self.skip_layer = nn.Linear(self.input_size, self.output_size)
        
        if self.context_size is not None:
            self.context = nn.Linear(self.context_size, self.hidden_size)
            
    def forward(self, x, context = None):
        if self.input_size!=self.output_size:
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
    def __init__(self, input_size, num_cont_inputs, hidden_size, dropout, context = None):        
        super(VariableSelectionNetwork, self).__init__()
        
        self.input_size = input_size
        self.num_cont_inputs = num_cont_inputs
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.context = context
        
        if self.context is not None:
            self.flattened_grn = GRN(self.num_cont_inputs * self.input_size, self.hidden_size, self.num_cont_inputs, self.dropout, self.context)
        else:
            self.flattened_grn = GRN(self.num_cont_inputs * self.input_size, self.hidden_size, self.num_cont_inputs, self.dropout)
            
        
        self.single_variable_grns = nn.ModuleList()
        for i in range(self.num_cont_inputs):
            self.single_variable_grns.append(GRN(self.input_size, self.hidden_size, self.hidden_size, self.dropout))

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

        return outputs, sparse_weights