# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 09:51:38 2023

@author: unicom
"""

import logging
import csv
import re, os
import pandas as pd
from datetime import datetime
import numpy as np
import boxx
from boxx import *
import math
from collections import Counter
from DataProcess import DataProcess
from DnsCNN import DnsCnn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
import torch
import torch.nn as nn
# import DnsCNN

class Trainer():
    def __init__(self, model, loss, optimizer, batch_size) -> None:
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.batch_size = batch_size


    def train(self, train_dataloader,save_mode_path):

        epoch = len(train_dataloader)
        iterator = tqdm(range(epoch), ncols=70)
        # tree-train_dataloader
        # tree-train_dataload
        loss = 0
        for epoch_num in iterator: 
        # for epoch_num in range(epoch-4, epoch):
            for i_batch, sampled_batch in enumerate(train_dataloader):
                
                self.model.train()  
                data_batch, label_batch = sampled_batch[0], sampled_batch[1]
                # data_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                outputs = self.model(data_batch)
                # tree-outputs
                # tree-label_batch
                # what-outputs
                loss = self.loss(outputs, label_batch)
                
                # # 验证过程
                # model.eval()  # 设置模型为评估模式，关闭 Dropout 等训练特定操作
                # correct = 0
                # total = 0
            
                # with torch.no_grad():  # 在验证过程中不需要计算梯度
                #     for data in validation_loader:  # 遍历验证集数据
                #         inputs, labels = data
            
                #         # 前向传播
                #         outputs = model(inputs)
                #         predicted_classes = torch.argmax(outputs, dim=1)
            
                #         # 统计准确率
                #         total += labels.size(0)
                #         correct += (predicted_classes == labels).sum().item()
            
                # accuracy = correct / total
                
                # tree-sampled_batch
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print("-----------------loss:", loss)
            if epoch_num >= epoch -1:
                torch.save(self.model.state_dict(), save_mode_path)
                print("save model to {}".format(save_mode_path))
                iterator.close()
                break
        # p/loss
        p/' '
        p/"Training Finished!"
        return 
    
    
class Tester():
    def __init__(self, model, loss, optimizer, batch_size) -> None:
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.batch_size = batch_size


    def test(self, test_dataloader,save_mode_path):
        self.model.eval()
        self.model.load_state_dict(torch.load(save_mode_path))
        accuracy = 0
        file_path = './whyyyyyyy.txt'
        with open(file_path, 'w') as f:
            f.write(str(datetime.now()) + '\n')
        
        for i_batch, sampled_batch in tqdm(enumerate(test_dataloader)):
            data_batch, label_batch = sampled_batch[0], sampled_batch[1]
            outputs = self.model(data_batch)
            # p/outputs
            # tree-outputs
            predicted_classes = torch.argmax(outputs, dim=1)
            true_classes = torch.argmax(label_batch, dim=1)
            correct = (predicted_classes == true_classes).sum().item()
            
            outputs = outputs.tolist()
            label_batch = label_batch.tolist()
            
            with open(file_path, 'a') as f:
                f.write("the data_batch number is :----" + str(len(predicted_classes))+'\n')
                for i in range(len(predicted_classes)):
                    f.write(str(outputs[i]) +"--"+ str(predicted_classes[i]) +  '=============' + str(label_batch[i]) +'---' + str(true_classes[i])+'\n')

            accuracy += correct / len(label_batch)
        
            
        accuracy = accuracy/len(test_dataloader)
        print("-----------------准确率:", accuracy)

        return 

def main():
    
    alpha = "0123456789abcdefghijklmnopqrstuvwxyz-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    train_file = "dataset/dataset_train_tunnel.csv"
    test_file = "dataset/dataset_test_tunnel.csv"
    dim = 128
    save_mode_path = './final_model.pth'
        
    input_size = dim
    alpha_size = len(alpha)
    embedding_size = 0
    kernel_size = 3
    num_of_classes = 2
    droupout = 0.1
    batch_size = 128

    train_data = DataProcess('train', train_file, alpha, dim, 2)  # 128
    train_data.load_data()
    train_dataset, train_label = train_data.get_all_data()   # train_dataset.shape  --  (12986, 1, 128)    traing_label.shape  --  (12986, 2)
    train_dataload =  train_data.getloaddata(train_dataset, train_label)
    train_dataloader = DataLoader(train_dataload, batch_size=batch_size,shuffle=True)
    
    test_data = DataProcess('test', test_file, alpha, dim, 2)  # 128
    test_data.load_data()
    test_dataset, test_label = test_data.get_all_data()   # train_dataset.shape  --  (12986, 1, 128)    traing_label.shape  --  (12986, 2)
    test_dataload =  test_data.getloaddata(test_dataset, test_label)
    test_dataloader = DataLoader(test_dataload, batch_size=batch_size,shuffle=True)

    DnsDetect = DnsCnn(dim,input_size, alpha_size, embedding_size, kernel_size, num_of_classes, droupout, optimizer = 'adam', loss='categorical_crossentropy')
 
    loss = nn.BCELoss()
    optimizer = optim.SGD(DnsDetect.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)

    # trainF = Trainer(DnsDetect, loss, optimizer, batch_size)
    # trainF.train(train_dataloader,save_mode_path)
    
    testF = Tester(DnsDetect, loss, optimizer, batch_size)
    testF.test(test_dataloader,save_mode_path)



    print("it work!")
    g()



if __name__ =="__main__":
    main()