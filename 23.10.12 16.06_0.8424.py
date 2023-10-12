23.10.12 16.06_0.8424.py.txtchange.py

import os
import csv
import pandas as pd
from boxx import *
import torch
import pandas as pd
import numpy as np

#调整测试集
# with open('dataset/dataset_test_tunnel.csv', 'w')as f1:
#     with open('dataset/dataset_test_tunnel-havecom.csv','r') as f2:
#         for line in f2:
#             print(line,'-------------')
#             stage1 = line.split(',')
#             stage2 = stage1[0].split('.')
#             stage2 = stage2[:-2]
#             final = ('.').join(stage2)
#             final = final + '.' + ',' + stage1[1]
#             print(final)
#             f1.write(final)

# #构建五折交叉验证数据集
# with open('dataset/dataset_all_tunnel.csv', 'w')as f:
#     with open('dataset/dataset_train_tunnel.csv','r') as f1: 
#         for line in f1:
#             f.write(line)

#     with open('dataset/dataset_test_tunnel.csv','r') as f2: 
#         next(f2)
#         for line in f2:
#             f.write(line)

# #构建验证集
# with open('dataset/dataset_train_tunnel.csv','r') as f1: 
#     sum = f1.readlines()
#     sum0 = (len(sum)//8)*3
#     sum1 = (len(sum)//8)*3
#     print('sum1:',sum1)
# with open('dataset/dataset_validation_tunnel.csv', 'w')as f:
#     with open('dataset/temp.csv','w') as f2: 
#         with open('dataset/dataset_train_tunnel.csv','r') as f1: 

#             # for line in f1:
#             #     p/line
#             p/f1.readline()
#             f.write(f1.readline())
#             p/f1.readline()
#             for line in f1:
#                 p/line
#                 if sum0 > 0:
#                     if int(line[-2]) == 1:
#                         f2.write(line)
#                         sum0 -= 1
#                         continue
#                 if sum1 > 0:    
#                     if int(line[-2]) == 0:
#                         f2.write(line)
#                         sum1 -= 1
#                         continue
#                 f.write(line)
                

# # with open('dataset/temp.csv', 'r', encoding='utf-8', ) as f:
# #     rr = csv.reader(f, delimiter=",", quotechar='"')
# rr = pd.read_csv('dataset/temp.csv')
# rr.info()
# # rr[rr['Label'] == 0].shape
# rr.duplicated(subset = 'Label' )

def get_k_fold_data(k, i, X, y):
    assert k >1
    fold_size = X.shape[0]//k
    X_train, y_train = None, None
    
    for j in range(k):
        p/j
        idx = slice(j * fold_size, (j + 1) * fold_size)
        p/idx
        
        
        X_part, y_part = X[idx,:],y[idx]
        
        if j == i: 
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
            
        else:
            X_train = torch.cat((X_train, X_part), dim=0) #dim=0增加行数，竖着连接
            y_train = torch.cat((y_train, y_part), dim=0)
        
    return X_train, y_train, X_valid, y_valid

# x = torch.randn((50,5))
# y = torch.randn((50,1))
# x = pd.read_csv("dataset/dataset_train_tunnel.csv")
# x = np.array(x, dtype = float32)
# x = torch.tensor(x)
y = pd.read_csv("dataset/dataset_test_tunnel.csv")

y = np.array(y)
# y = torch.tensor(y)
# X_train, y_train, X_valid, y_valid = get_k_fold_data(10, 0, x, y)

from torch import nn 
import torch 

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







DataProcess.py

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
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader




class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y

    def __len__(self):
        return len(self.data)

class DataProcess(object):
    def __init__(self, typet, data_file, alpha,
                 input_size, num_of_classes) -> None:
        '''
        args:
            data_file: csv file
            input_size: features
            num of classes: label
        '''
        self.typet = typet
        self.data_file = data_file
        self.alpha = alpha
        self.alpha_size = len(alpha)
        self.classes = num_of_classes
        self.dict = {}
        self.dict['unk'] = 0
        # one-hot编码
        for idx, alp in enumerate(self.alpha):
            self.dict[alp] = idx 
        self.size = input_size
        print(self.typet + "data loading..............")
    
    def caculate_entropy(self, s):
        lens = len(s)
        num_counts = Counter(s)
        entropy = 0
        for count in num_counts.values():
            probability = count/lens
            entropy += - probability * math.log2(probability)
        return int(entropy*10)


    def load_data(self):
        data = []
        with open(self.data_file, 'r', encoding='utf-8', ) as f:
            rr = csv.reader(f, delimiter=",", quotechar='"')
            next(rr)
            for line in rr:
                txt = ''
                for s in line:
                    label = int(line[1])
                    # 先去掉最后一个'.'
                    txt = line[0][:-1]  
                    # 如果有子域名，在这里处理！
                    txtsp = txt.split('.')
                    if len(txtsp) >= 2:
                        subdotxt = txtsp[-2]   #三级子域名
                    else:
                        subdotxt = ''
                    dotxt = txtsp[-1]   #二级子域名
                    subdosize = len(subdotxt)   #三级子域名长度
                    dosize = len(dotxt)    #二级子域名长度
                    entropy = self.caculate_entropy(dotxt)     #二级子域名熵值
                    Anum = sum(1 for char in dotxt if char.isdigit())   #数字个数
                    nnum = sum(1 for char in dotxt if not char.isalnum())   #符号+空格的个数
                    subentropy = self.caculate_entropy(subdotxt)

                data.append([label, dotxt, dosize, entropy, Anum, nnum, subdotxt, subdosize, subentropy])
        self.data = np.array(data)
        # p/data
        # p/self.data
        print(self.typet + "data load finishing..............")
        g()

    def str_to_indexes(self, s, size):
        """
        reverse abc to 123

        """
        s = s.lower()
        max_length = min(len(s), size)
        str2idx = np.zeros(size, dtype='int64')
        for i in range(max_length):
            str2idx[i] = self.dict[s[i]]
        return str2idx

    def get_all_data(self):
        '''
        get the np-type data, with one-hot label.    onebatch!
        la: label
        doa: domian 
        donum: len(domain) 
        doent: domain.entropy 
        do123,; domain_have_number 
        dosym: domain_have_symbol 
        subdo: sub-domain
        subnum: len(sub-domain)
        subdoent: subdomain_entropy
        '''
        print(self.typet + "data transfer starting...........")
        datasize = len(self.data)  
        start_index = 0
        end_index = datasize
        batch_txts = self.data[start_index:end_index]  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        batch_indices = []
        one_hot = np.eye(self.classes, dtype='int64')
        classes = []
        # what-self.data
        # what-batch_txts
        sizeeeeee = (self.size-6)//2
        
        for line in batch_txts:
            la, doa, donum, doent, do123, dosym, subdoa, subnum, subdoent = line
            # print(la, doa, donum, doent, do123, dosym, subdo, subnum,'\n')
            la = one_hot[int(la)]
            doa = self.str_to_indexes(doa, sizeeeeee)
            subdoa = self.str_to_indexes(subdoa, sizeeeeee)
            donum = np.int64([donum])
            doent = np.int64([0 if not doent else int(doent)])
            do123 = np.int64([do123])
            dosym = np.int64([dosym])
            subnum =  np.int64([subnum])
            subdoent = np.int64([0 if not subdoent else int(subdoent)])
            
            batch_indices.append(np.concatenate((doa, donum, doent, do123, dosym, subdoa, subnum, subdoent), axis = 0))
            # batch_indices.append([doa, donum, doent, do123, dosym, subdoa, subnum, subdoent])
            classes.append(la)
        # tree/batch_indices
        print(self.typet + "data transfer finishing...........")   #12986




        batch_indices = torch.tensor(batch_indices)
        # batch_indices = torch.unsqueeze(batch_indices, dim = -2)
        classes = torch.Tensor(classes)
        
        # batch_indices = torch.tensor(batch_indices).float()
        # batch_indices = torch.unsqueeze(batch_indices, dim = -2)
        # classes = torch.Tensor(classes).float()
        
        # mean = batch_indices.mean(dim=(0, 2))
        # std = batch_indices.std(dim=(0, 2))

        # normalize = transforms.Normalize(mean=mean, std=std)
        # batch_indices = normalize(batch_indices)
        g()
        return batch_indices, classes

    #变成torch的数据集形式，然后就可以加载到loader
    def getloaddata(self, train_dataset, train_label):
        dataset = CustomDataset(data=train_dataset, labels=train_label)
        return dataset
    
    
    def get_k_fold(self, k, i, train_dataset, train_label):
        fold_size = train_dataset.shape[0]//k
        
        data_train, label_train = None, None
        
        for j in range(k):
            
            idx = slice(j * fold_size, (j + 1) * fold_size)
            
            
            X_part, y_part = train_dataset[idx,:],train_label[idx, :]
            
            if j == i: 
                data_valid, label_valid = X_part, y_part
            elif data_train is None:
                data_train, label_train = X_part, y_part
                
            else:
                data_train = torch.cat((data_train, X_part), dim=0) #dim=0增加行数，竖着连接
                label_train = torch.cat((label_train, y_part), dim=0)
            
        return data_train, label_train, data_valid, label_valid
debug.py

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 14:47:26 2023

@author: unicom
"""

from boxx import *
import numpy as np



# dict = {}
# max_length = 9
# s = 'martineve'
# alpha = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
# for idx, alp in enumerate(alpha):
#     dict[alp] = idx + 1
# p/dict
# str2idx = np.zeros(60, dtype='int64')
# for i in range(max_length):
#     str2idx[i] = dict[s[i]]
#     p/str2idx
        
# for i in range(max_length):
#     str2idx[s[i]] = dict[s[i]]
#     p/str2idx


batch_txts =np.array(['1','b3d01c72045a640d302c0b49fa','26','3.6094963344256965','17','0','a544a5d92cd881a0ff6cf1b833419d7480f2453d0978799bac64f3194d1d','60'])
# for la, doa, donum, doent, do123, dosym, subdo, subnum  in batch_txts:
#     p/la
#     p/doa
la, doa, donum, doent, do123, dosym, subdo, subnum = batch_txts
print("la:", la)
print("doa:", doa)
DnsCNN.py

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
        self.in_channels = dim
        self.out_channels = 64
        self.alpha_size = alpha_size
        self.embedding_size = embedding_size
        self.kernel_size = kernel_size
        self.num_of_classes = num_of_classes
        # self.droupout = droupout
        self.optimizer = optimizer
        self.loss = loss
        self.Sigmoid = nn.Sigmoid()
        self.tryconv = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size,  stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.dim = dim
        self.item = 4
        # (in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.conv = nn.Conv1d(self.out_channels, self.out_channels, self.kernel_size,  stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.bn = nn.BatchNorm1d(self.out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
        self.relu = nn.ReLU(inplace=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2, padding=1) 
        self.line = nn.Linear(((self.input_size//2)+1)*self.out_channels, self.num_of_classes, bias=True, device=None, dtype=None)
        self.dropout = nn.Dropout(p = droupout)
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
            outs = self.lrelu(outs) 

            # outs += residual 
        outs = self.pool(outs) 
            # tree-outs
            # tree-out
            # tree-residual
            # out += residual
            # residual = out
        outs = outs.view(outs.size(0), -1)
        # tree-outs
        outs = self.line(outs)
        outs = self.dropout(outs)
        outs = self.Sigmoid(outs)
        # out = self.softmax(out)
        
        # tree-outs
        g()
        # p/'network creat finishing.........'
        return outs
DNS_detection.py

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
DEVICE = torch.device("cpu")



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
DNS_detection_times.py

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


    DnsDetect = DnsCnn(dim,input_size, alpha_size, embedding_size, kernel_size, num_of_classes, droupout, optimizer = 'adam', loss='categorical_crossentropy')
 

    times = 10
    
    # loss = nn.BCELoss()
    loss = nn.SmoothL1Loss()
    optimizer = optim.SGD(DnsDetect.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)    
        
    save_mode_path = 'final_model.pth'

    # train_data = DataProcess('train', train_file, alpha, dim, 2)  # 128
    # train_data.load_data()
    # train_dataset, train_label = train_data.get_all_data()   # train_dataset.shape  --  (12986, 256)    traing_label.shape  --  (12986, 2)
    # numtr = train_dataset.shape[0]
    

    
    # for time in range(times):
    #     data_train, label_train, data_valid, label_valid = train_data.get_k_fold(times, time, train_dataset, train_label)
    #     train_dataload =  train_data.getloaddata(data_train, label_train)
    #     train_dataloader = DataLoader(train_dataload, batch_size=batch_size,shuffle=True)
        
    #     valid_dataload =  train_data.getloaddata(data_valid, label_valid)
        # valid_dataloader = DataLoader(valid_dataload, batch_size=batch_size,shuffle=True)
          # numva = train_dataload.shape[0]
        

            
    #     trainF = Trainer(DnsDetect, loss, optimizer, batch_size)
    #     trainF.train(train_dataloader, save_mode_path, numva, valid_dataloader,ACC)
    
    test_data = DataProcess('test', test_file, alpha, dim, 2)  # 128
    test_data.load_data()
    test_dataset, test_label = test_data.get_all_data()   # train_dataset.shape  --  (12986, 1, 128)    traing_label.shape  --  (12986, 2)
    test_dataload =  test_data.getloaddata(test_dataset, test_label)
    test_dataloader = DataLoader(test_dataload, batch_size=batch_size,shuffle=True)
    numte = test_dataset.shape[0]
    
    

    
    
    testF = Tester(DnsDetect, loss, optimizer, batch_size)
    testF.test(numte, test_dataloader, save_mode_path)


    print("it work!")
    g()



if __name__ =="__main__":
    main()
    print(datetime.now())
    
    
    
    
    
save.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 14:45:07 2023

@author: haikunzhang
"""
import os
import re
from boxx import *
from datetime import datetime
import shutil

curr_path = os.getcwd()
curr_files = os.listdir(curr_path)
p/curr_files

print("please inter the accuarcy:----------")
accuarcy = input()
# accuarcy= 0.86
file_name = str(datetime.now().strftime('%y.%m.%d %H.%M') ) + '_' + str(accuarcy) +'.py'

with open(file_name, 'w', encoding = 'utf-8') as f:
    f.write(file_name + '.txt')
    
    for i in curr_files:
        if '.py' in i:
            f.write(i + '\n' + '\n')
            with open(i, 'r', encoding = 'utf-8') as fpy:
                for line in fpy:
                    f.write(line)
                f.write('\n')

            
shutil.copy('./final_model.pth', './' + file_name +'_' + 'model.pth')
