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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DnsCnn(nn.Module):
    '''
    
    simple cnn
    
    '''
    def __init__(self, dim,input_size, alpha_size, embedding_size,
                 kernel_size, num_of_classes, droupout
                 ) -> None:
        '''
        input_ssize: alpha 
        embedding_size: embedding
        conv_layer: conv 
        fully_connected_layer: fully connect
        num_of_classes: class (2)
        droupout: dropout
        optimizer = 'adam', loss='categorical_crossentropy'
        '''
        super(DnsCnn, self).__init__()
        self.input_size = input_size
        self.in_channels = dim
        self.connsize = dim
        self.out_channels = 64
        self.alpha_size = alpha_size
        self.embedding_size = embedding_size
        self.kernel_size = kernel_size
        self.num_of_classes = num_of_classes
        # self.droupout = droupout
        self.Sigmoid = nn.Sigmoid()
        self.dim = dim
        # self.item = 1
        # (in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

        self.bn = nn.BatchNorm1d(self.in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=DEVICE, dtype=None)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.PReLU(num_parameters=1, init=0.25)
        self.lrelu = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2, padding=1) 
        # self.line2 = nn.Linear( self.in_channels, self.num_of_classes, bias=True, device=DEVICE, dtype=None)
        self.dropout = nn.Dropout(p = droupout)
        
        
        self.embedding = nn.Embedding(self.alpha_size+1, self.in_channels)
        self.convlist = []
        for kernel in ([7,3,5,3]):
            self.convlist.append(nn.Conv1d(self.in_channels, self.in_channels, kernel,  stride=1, padding = kernel//2, dilation=1, groups=1, bias=True, padding_mode='zeros', device=DEVICE, dtype=None))
            self.connsize = self.connsize//2+1
        self.line1 = nn.Linear( self.in_channels*self.connsize, self.num_of_classes, bias=True, device=None, dtype=None)
        
        self.line = nn.Linear( in_features = self.dim, out_features = self.num_of_classes, bias=True, device=None, dtype=None)
        
        #单层线性回归
        self.linehuigui = nn.Linear( in_features = self.dim*self.dim, out_features = self.num_of_classes, bias=True, device=None, dtype=None)
        

    def forward(self, x):
        # p/'network creating.........'
        # tree-x
        outs = self.embedding(x)
        # outs = torch.unsqueeze(outs, dim = -2)
        # tree/outs    #256,256,256
        
        # residual = outs
        for conv in self.convlist:
            outs = conv(outs)   
            outs = self.bn(outs) 
            outs = self.lrelu(outs) 
            
            # tree-outs
            outs = self.pool(outs) 
            # tree-outs

            # outs += residual 
        # outs = self.pool(outs) 
        # tree-outs
            # tree-out
            # tree-residual
            # out += residual
            # residual = out
        outs = outs.view(outs.size(0), -1)
        # tree-outs
        # outs = self.linehuigui(outs)
        outs = self.line1(outs)
        outs = self.dropout(outs)
        outs = self.lrelu(outs) 
        
        # outs = self.line2(outs)
        outs = self.Sigmoid(outs)
        # out = self.softmax(out)
        
        # tree-outs
        g()
        # p/'network creat finishing.........'
        return outs