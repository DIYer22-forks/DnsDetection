# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 09:51:38 2023

@author: unicom
"""
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
import DnsCNN




def main():
    alpha = "0123456789abcdefghijklmnopqrstuvwxyz-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    train_file = "dataset/dataset_train_tunnel.csv"
    test_file = "dataset/dataset_test_tunnel.csv"

    train_data = DataProcess(train_file, alpha, 64, 2)  # 256
    train_data.load_data()
    train_dataset, train_label = train_data.get_all_data()   # train_dataset.shape  --  (12986, 7, 64)    traing_label.shape  --  (12986, 2)


    test_data = DataProcess(test_file, alpha, 64, 2)  # 256
    test_data.load_data()
    test_dataset, test_label = train_data.get_all_data()   # train_dataset.shape  --  (12986, 7, 64)    traing_label.shape  --  (12986, 2)


    input_size = train_dataset.shape
    alpha_size = len(alpha)
    embedding_size = 0
    kernel_size = 3
    num_of_classes = 2
    droupout = 0.1

    DnsDetect = DnsCNN(input_size, alpha_size, embedding_size, kernel_size, num_of_classes, droupout, optimizer = 'adam', loss='categorical_crossentropy')
    



    print("it work!")



if __name__ =="__main__":
    main()