import csv
import re, os
import pandas as pd
import numpy as np
import boxx
from boxx import *
import math
from collections import Counter
import DataProcess
import torch
import torch.nn as nn


class DnsCnn(nn.Module):
    '''
    
    simple cnn
    
    '''
    def __init__(self, dim,input_size, alpha_size, embedding_size,
                 kernel_size, num_of_classes, droupout,
                 optimizer = 'adam', loss='categorical_crossentropy'
                 ) -> None:
        '''
        input_size: dataset.shape
        alpha_size: alpha 
        embedding_size: embedding
        conv_layer: conv 
        fully_connected_layer: fully connect
        num_of_classes: class (2)
        droupout: dropout
        optimizer = 'adam', loss='categorical_crossentropy'
        '''
        super(DnsCnn, self).__init__()
        self.input_size = input_size
        self.in_channels = 128
        self.out_channels = 64
        self.alpha_size = alpha_size
        self.embedding_size = embedding_size
        self.kernel_size = kernel_size
        self.num_of_classes = num_of_classes
        self.droupout = droupout
        self.optimizer = optimizer
        self.loss = loss
        self.Sigmoid = nn.Sigmoid()
        self.tryconv = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size,  stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.dim = dim
        self.item = 1
        # (in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.conv = nn.Conv1d(self.out_channels, self.out_channels, self.kernel_size,  stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.bn = nn.BatchNorm1d(self.out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2, padding=1) 
        self.line = nn.Linear(((self.input_size//2)+1)*self.out_channels, self.num_of_classes, bias=True, device=None, dtype=None)
        self.dropout = nn.Dropout(p = 0.5)
        self.embedding = nn.Embedding(self.alpha_size+1, self.in_channels)

    def forward(self, x):
        # p/'network creating.........'
        outs = self.embedding(x)
        # outs = torch.unsqueeze(outs, dim = -2)
        # tree/outs
        outs = self.tryconv(outs)
        residual = outs
        for _ in range(self.item):
            outs = self.conv(outs)   
            outs = self.bn(outs) 
            outs = self.relu(outs) 
            outs = self.dropout(outs)
        outs = self.pool(outs) 
            # tree-outs
            # tree-out
            # tree-residual
            # out += residual
            # residual = out
        outs = outs.view(outs.size(0), -1)
        # tree-outs
        outs = self.line(outs)
        outs = self.Sigmoid(outs)
        # out = self.softmax(out)
        
        # tree-outs
        g()
        # p/'network creat finishing.........'
        return outs