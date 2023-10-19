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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACC = 0



class Trainer():
    def __init__(self, model, loss, optimizer, batch_size) -> None:
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.batch_size = batch_size


    def train(self, train_dataloader,save_mode_path, num, test_dataloader, ACC):

        epoch = len(train_dataloader)
        epoch = 2
        iterator = tqdm(range(epoch), ncols=70)
        loss = 0
        temp_acc = 0
        
        self.model = self.model.to(DEVICE)
        for epoch_num in iterator: 
        # for epoch_num in range(epoch-1, epoch):
            for i_batch, sampled_batch in enumerate(train_dataloader):
                self.model.train()  
                data_batch, label_batch = sampled_batch[0], sampled_batch[1]
                data_batch = data_batch.to(DEVICE)
                label_batch = label_batch.to(DEVICE)

                outputs = self.model(data_batch)
                loss = self.loss(outputs, label_batch)
                
                # # 验证过程
                self.model.eval()  # 设置模型为评估模式，关闭 Dropout 等训练特定操作
                tp = 0
                tn = 0
                fp = 0
                fn = 0
            
                with torch.no_grad():  # 在验证过程中不需要计算梯度

                    for i_batch1, sampled_batch in tqdm(enumerate(test_dataloader)):
                        data_batch, label_batch = sampled_batch[0], sampled_batch[1]
                        data_batch = data_batch.to(DEVICE)
                        label_batch = label_batch.to(DEVICE)

                        outputs = self.model(data_batch)
                        predicted_classes = torch.argmax(outputs, dim=1).bool()
                        true_classes = torch.argmax(label_batch, dim=1).bool()
                        tn += torch.logical_and(predicted_classes == 0, true_classes == 0).sum().item()
                        tp += torch.logical_and(predicted_classes == 1, true_classes == 1).sum().item()
                        fp += torch.logical_and(predicted_classes == 1, true_classes == 0).sum().item()
                        fn += torch.logical_and(predicted_classes == 0, true_classes == 1).sum().item()

                        outputs = outputs.tolist()
                        label_batch = label_batch.tolist()
                    # 准确率 = TP / (TP + FP)
                    precision = round((tp+0.01)/(tp+fp+0.01),4)
                    # 召回率 = TP / (TP + FN)
                    recall = round((tp+0.01)/(tp+fn+0.01),4)

                    accuracy = round((tp+tn+0.01)/num,4)
                    print("the ",epoch_num, ' epoch ', i_batch, " batch:")
                    print("--tp:", tp, ' --tn:',tn, ' --fp:', fp, ' --fn:', fn)
                    print("--准确率:", accuracy, ' --查准率：',precision, ' --查全率：', recall)
                    
                    if accuracy >= ACC:
                        torch.save(self.model.state_dict(), save_mode_path)
                        print("save model to {}".format(save_mode_path))
                        ACC = accuracy
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # print("-----------------loss:", loss)
            # if epoch_num >= epoch -1:
            #     torch.save(self.model.state_dict(), save_mode_path)
            #     print("save model to {}".format(save_mode_path))
            #     iterator.close()
            #     break
        # p/loss
        p/' '
        p/"Training Finished!"
        return (tp, tn, fp, fn)
    
    
class Tester():
    def __init__(self, model, loss, optimizer, batch_size) -> None:
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.batch_size = batch_size


    def test(self, num, test_dataloader,save_mode_path):
        self.model.eval()
        self.model.load_state_dict(torch.load(save_mode_path))
        self.model = self.model.to(DEVICE)
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        

        file_path = './whyyyyyyy.txt'
        with open(file_path, 'w') as f:
            f.write(str(datetime.now()) + '\n')
        
        for i_batch, sampled_batch in tqdm(enumerate(test_dataloader)):
            data_batch, label_batch = sampled_batch[0], sampled_batch[1]
            data_batch = data_batch.to(DEVICE)
            label_batch = label_batch.to(DEVICE)

            outputs = self.model(data_batch)
            # p/outputs

            predicted_classes = torch.argmax(outputs, dim=1).bool()
            true_classes = torch.argmax(label_batch, dim=1).bool()
            # tree-predicted_classes
            # tree-true_classes

            tn += torch.logical_and(predicted_classes == 0, true_classes == 0).sum().item()
            tp += torch.logical_and(predicted_classes == 1, true_classes == 1).sum().item()
            fp += torch.logical_and(predicted_classes == 1, true_classes == 0).sum().item()
            fn += torch.logical_and(predicted_classes == 0, true_classes == 1).sum().item()

            outputs = outputs.tolist()
            label_batch = label_batch.tolist()

            with open(file_path, 'a') as f:
                f.write("the data_batch number is :----" + str(len(predicted_classes))+'\n')
                for i in range(len(predicted_classes)):
                    f.write(str(outputs[i]) +"--"+ str(predicted_classes[i]) +  '=============' + str(label_batch[i]) +'---' + str(true_classes[i])+'\n')

        
            
        
        # 准确率 = TP / (TP + FP)
        precision = round((tp+0.01)/(tp+fp+0.01),4)
        # 召回率 = TP / (TP + FN)
        recall = round((tp+0.01)/(tp+fn+0.01),4)
        
        accuracy = round((tp+tn+0.01)/num,4)
        print("--tp:", tp, ' --tn:',tn, ' --fp:', fp, ' --fn:', fn)
        print("--准确率:", accuracy, ' --查准率：',precision, ' --查全率：', recall)
        print('FNR:',fn/(tp+fn),'  FPR: ', fp/(fp+fn))

        return 

def main():

    
    alpha = "0123456789abcdefghijklmnopqrstuvwxyz-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    train_file = "dataset/dataset_train_tunnel.csv"
    test_file = "dataset/dataset_test_tunnel.csv"
    dim = 256
    save_mode_path = './final_model.pth'
        
    input_size = 256
    alpha_size = len(alpha)
    embedding_size = 0
    kernel_size = 3
    num_of_classes = 2
    droupout = 0.1
    batch_size = 1000


    DnsDetect = DnsCnn(dim,input_size, alpha_size, embedding_size, kernel_size, num_of_classes, droupout)
 

    times = 5
    
    # loss = nn.BCELoss()
    loss = nn.SmoothL1Loss()
    optimizer = optim.SGD(DnsDetect.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)    
        
    save_mode_path = 'final_model.pth'

    train_data = DataProcess('train', train_file, alpha, dim, 2)  # 128
    train_data.load_data()
    train_dataset, train_label = train_data.get_all_data()   # train_dataset.shape  --  (12986, 256)    traing_label.shape  --  (12986, 2)
    # numtr = train_dataset.shape[0]
    
    valid_data = DataProcess('train', train_file, alpha, dim, 2)  # 128
    valid_data.load_data()
    train_dataset, train_label = train_data.get_all_data() 

    tp = tn = fp = fn=0
    for time in range(times):
        data_train, label_train, data_valid, label_valid = train_data.get_k_fold(times, time, train_dataset, train_label)
        train_dataload =  train_data.getloaddata(data_train, label_train)
        train_dataloader = DataLoader(train_dataload, batch_size=batch_size,shuffle=True)
        
        valid_dataload =  train_data.getloaddata(data_valid, label_valid)
        valid_dataloader = DataLoader(valid_dataload, batch_size=batch_size,shuffle=True)
        
        # tree-data_train
        # tree-label_train

            
        trainF = Trainer(DnsDetect, loss, optimizer, batch_size)
        tp1, tn1, fp1, fn1 = trainF.train(train_dataloader, save_mode_path, num, valid_dataloader,ACC)
        tp += tp1
        tn += tn1
        fp += fp1
        fn += fn1
        
    precision = round((tp+0.01)/(tp+fp+0.01),4)
    # 召回率 = TP / (TP + FN)
    recall = round((tp+0.01)/(tp+fn+0.01),4)

    accuracy = round((tp+tn+0.01)/num,4)
    
    print('precision:',precision, 'recall:',recall,'accuary:', accuracy)

#     test_data = DataProcess('test', test_file, alpha, dim, 2)  # 128
#     test_data.load_data()
#     test_dataset, test_label = test_data.get_all_data()   # train_dataset.shape  --  (12986, 1, 128)    traing_label.shape  --  (12986, 2)
#     test_dataload =  test_data.getloaddata(test_dataset, test_label)
#     test_dataloader = DataLoader(test_dataload, batch_size=batch_size,shuffle=True)
#     numte = test_dataset.shape[0]
    
    

    
    
#     testF = Tester(DnsDetect, loss, optimizer, batch_size)
#     testF.test(numte, test_dataloader, save_mode_path)


    print("it work!")
    g()



if __name__ =="__main__":
    main()
    print(datetime.now())
    
    
    
    
    