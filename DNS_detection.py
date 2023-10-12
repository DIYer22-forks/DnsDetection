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
import torch.nn.functional as F
# import DnsCNN
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")



class FocalLoss(nn.Module):
    def __init__(self, gama=1.5, alpha=0.25, weight=None, reduction="mean") -> None:
        super().__init__() 
        self.loss_fcn = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)
        self.gama = gama 
        self.alpha = alpha 

    def forward(self, pre, target):
        logp = self.loss_fcn(pre, target)
        p = torch.exp(-logp) 
        loss = (1-p)**self.gama * self.alpha * logp
        return loss.mean()
    
    
    
class Trainer():
    def __init__(self, model, loss, optimizer, batch_size) -> None:
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.batch_size = batch_size


    def train(self, train_dataloader,save_mode_path, num, valid_dataloader):

        epoch = len(train_dataloader)
        epoch = 2
        iterator = tqdm(range(epoch), ncols=70)
        loss = 0
        temp_acc = 0
        epoch_best = 0
        batch_best = 0
        
        self.model = self.model.to(DEVICE)
        # torch.cuda.manual_seed(0)
        for epoch_num in iterator: 
        # for epoch_num in range(epoch-1, epoch):
            
            # torch.cuda.manual_seed(0)
            for i_batch, sampled_batch in enumerate(train_dataloader):
                self.model.train()  
                data_batch, label_batch = sampled_batch[0], sampled_batch[1]
                data_batch = data_batch.to(DEVICE)
                label_batch = label_batch.to(DEVICE)

                outputs = self.model(data_batch)
                
                loss = self.loss(outputs, label_batch)
                
                # # 验证过程
                # self.model.eval()  # 设置模型为评估模式，关闭 Dropout 等训练特定操作
                # tp = 0
                # tn = 0
                # fp = 0
                # fn = 0
            
                # with torch.no_grad():  # 在验证过程中不需要计算梯度

                #     for i_batch1, sampled_batch in tqdm(enumerate(valid_dataloader)):
                #         data_batch, label_batch = sampled_batch[0], sampled_batch[1]
                #         data_batch = data_batch.to(DEVICE)
                #         label_batch = label_batch.to(DEVICE)

                #         outputs = self.model(data_batch)
                #         predicted_classes = torch.argmax(outputs, dim=1).bool()
                #         true_classes = torch.argmax(label_batch, dim=1).bool()
                #         tn += torch.logical_and(predicted_classes == 0, true_classes == 0).sum().item()
                #         tp += torch.logical_and(predicted_classes == 1, true_classes == 1).sum().item()
                #         fp += torch.logical_and(predicted_classes == 1, true_classes == 0).sum().item()
                #         fn += torch.logical_and(predicted_classes == 0, true_classes == 1).sum().item()

                #         outputs = outputs.tolist()
                #         label_batch = label_batch.tolist()
                #     # 准确率 = TP / (TP + FP)
                #     precision = round((tp+0.01)/(tp+fp+0.01),4)
                #     # 召回率 = TP / (TP + FN)
                #     recall = round((tp+0.01)/(tp+fn+0.01),4)

                #     accuracy = round((tp+tn+0.01)/num,4)
                #     print("the ",epoch_num, ' epoch ', i_batch, " batch:")
                #     print("--tp:", tp, ' --tn:',tn, ' --fp:', fp, ' --fn:', fn)
                #     print("--准确率:", accuracy, ' --查准率：',precision, ' --查全率：', recall)
                    
                #     # if i_batch1 == epoch and accuracy >= temp_acc:
                #     #     epoch_best = epoch_num
                #     #     batch_best = i_batch
                #     #     torch.save(self.model.state_dict(), save_mode_path)
                #     #     print("save model to {}".format(save_mode_path))
                #     #     temp_acc = accuracy
                
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
        p/"Training Finished! "
        print(" the best epoch is",epoch_num ,' batch:', i_batch)
        return 
    
    
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
            
        # torch.cuda.manual_seed(0)
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

    # torch.cuda.manual_seed(0)
    alpha = "0123456789abcdefghijklmnopqrstuvwxyz-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    train_file = "dataset/dataset_train.csv"
    valid_file = "dataset/dataset_validation.csv"
    test_file = "dataset/dataset_test.csv"
    dim = 256
    save_mode_path = './final_model.pth'
        
    input_size = 256
    alpha_size = len(alpha)
    embedding_size = 0
    kernel_size = 3
    num_of_classes = 2
    # droupout = 0.2
    droupout = 0
    batch_size = 2048

    train_data = DataProcess('train', train_file, alpha, dim, 2)  # 128
    train_data.load_data()
    train_dataset, train_label = train_data.get_all_data()   # train_dataset.shape  --  (12986, 1, 128)    traing_label.shape  --  (12986, 2)
    train_dataload =  train_data.getloaddata(train_dataset, train_label)
    train_dataloader = DataLoader(train_dataload, batch_size=batch_size,shuffle=True)
    
    valid_data = DataProcess('valid', valid_file, alpha, dim, 2)  # 128
    valid_data.load_data()
    valid_dataset, valid_label = valid_data.get_all_data()   # train_dataset.shape  --  (12986, 1, 128)    traing_label.shape  --  (12986, 2)
    valid_dataload =  valid_data.getloaddata(valid_dataset, valid_label)
    valid_dataloader = DataLoader(valid_dataload, batch_size=batch_size,shuffle=True)

    test_data = DataProcess('test', test_file, alpha, dim, 2)  # 128
    test_data.load_data()
    test_dataset, test_label = test_data.get_all_data()   # train_dataset.shape  --  (12986, 1, 128)    traing_label.shape  --  (12986, 2)
    test_dataload =  test_data.getloaddata(test_dataset, test_label)
    test_dataloader = DataLoader(test_dataload, batch_size=batch_size,shuffle=True)

    DnsDetect = DnsCnn(dim,input_size, alpha_size, embedding_size, kernel_size, num_of_classes, droupout, optimizer = 'adam', loss='categorical_crossentropy')
 
    num = test_dataset.shape[0]
    # loss = nn.BCELoss()
    loss = nn.BCEWithLogitsLoss()
    # loss =  FocalLoss()
    optimizer = optim.SGD(DnsDetect.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)
    # optimizer = optim.Adagrad(DnsDetect.parameters(), lr=0.01, lr_decay=0, weight_decay=0)
    

    # 训练
    trainF = Trainer(DnsDetect, loss, optimizer, batch_size)
    trainF.train(train_dataloader, save_mode_path, num, valid_dataloader)

    #取参数
    # save_mode_path = '23.10.06 17.26_0.9647_model.pth'
    
    # 测试
    testF = Tester(DnsDetect, loss, optimizer, batch_size)
    testF.test(num, test_dataloader, save_mode_path)




    print("it work!")
    g()



if __name__ =="__main__":
    main()
    print(datetime.now())