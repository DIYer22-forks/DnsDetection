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


class DnsCNN(nn.module):
    '''
    
    simple cnn
    
    '''
    def __init__(self, input_size, alpha_size, embedding_size,
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
        super(DnsCNN, self).__init__()
        self.input_size = input_size
        self.in_channels = self.input_size
        self.out_channels = self.input_size
        self.alpha_size = alpha_size
        self.embedding_size = embedding_size
        self.kernel_size = kernel_size
        self.num_of_classes = num_of_classes
        self.droupout = droupout
        self.optimizer = optimizer
        self.loss = loss
        
        # (in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.conv = nn.ModuleList([nn.conv1d(self.in_channels, self.in_channels, self.kernel_size,  stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None),
                                   nn.BatchNorm1d(self.in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None),
                                   nn.MaxPool1d(self.kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False) ])
        self.line = nn.Linear(self.in_channels, self.out_channels, bias=True, device=None, dtype=None)
        self.softmax = nn.Softmax(dim=self.out_channels)


    def forward(self, x):
        residual = x
        
        for _ in range(3):
            for model in self.conv:
                x = model(x)    
            x += residual
            residual = x
        x = self.line(x)
        x = self.softmax(x)

        return x