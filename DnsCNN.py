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
        super(DnsCnn, self).__init__()
        self.input_size = input_size
        self.in_channels = self.input_size
        self.out_channels = 64
        self.alpha_size = alpha_size
        self.embedding_size = embedding_size
        self.kernel_size = kernel_size
        self.num_of_classes = num_of_classes
        self.droupout = droupout
        self.optimizer = optimizer
        self.loss = loss
        
        self.tryconv = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size,  stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
                                    
        # (in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.conv = nn.ModuleList([nn.Conv1d(self.out_channels, self.out_channels, self.kernel_size,  stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None),
                                    nn.BatchNorm1d(self.out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool1d(self.kernel_size, stride=2, padding=1) ])
        self.line = nn.Linear(self.out_channels*32, self.num_of_classes, bias=True, device=None, dtype=None)


    def forward(self, x):
        p/'network creating.........'
        outs = self.tryconv(x)
        residual = outs
        for _ in range(2):
            for model in self.conv:
                
                outs = model(outs)   
            # tree-out
            # tree-residual
            # out += residual
            # residual = out
        outs = outs.view(outs.size(0), -1)
        outs = self.line(outs)
        # out = self.softmax(out)
        
        tree-outs
        g()
        p/'network creat finishing.........'
        return outs