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

# def get_k_fold_data(k, i, X, y):
#     assert k >1
#     fold_size = X.shape[0]//k
#     X_train, y_train = None, None
    
#     for j in range(k):
#         p/j
#         idx = slice(j * fold_size, (j + 1) * fold_size)
#         p/idx
        
        
#         X_part, y_part = X[idx,:],y[idx]
        
#         if j == i: 
#             X_valid, y_valid = X_part, y_part
#         elif X_train is None:
#             X_train, y_train = X_part, y_part
            
#         else:
#             X_train = torch.cat((X_train, X_part), dim=0) #dim=0增加行数，竖着连接
#             y_train = torch.cat((y_train, y_part), dim=0)
        
#     return X_train, y_train, X_valid, y_valid

# # x = torch.randn((50,5))
# # y = torch.randn((50,1))
# # x = pd.read_csv("dataset/dataset_train_tunnel.csv")
# # x = np.array(x, dtype = float32)
# # x = torch.tensor(x)
# y = pd.read_csv("dataset/dataset_test_tunnel.csv")

# y = np.array(y)
# # y = torch.tensor(y)
# # X_train, y_train, X_valid, y_valid = get_k_fold_data(10, 0, x, y)

# from torch import nn 
# import torch 

# class FocalLoss(nn.Module):
#     def __init__(self, gama=1.5, alpha=0.25, weight=None, reduction="mean") -> None:
#         super().__init__() 
#         self.loss_fcn = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)
#         self.gama = gama 
#         self.alpha = alpha 

#     def forward(self, pre, target):
#         logp = self.loss_fcn(pre, target)
#         p = torch.exp(-logp) 
#         loss = (1-p)**self.gama * self.alpha * logp
#         return loss.mean()




