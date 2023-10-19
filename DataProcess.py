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
        str2idx = np.zeros(size, dtype='int32')
        for i in range(max_length):
            if s[i] in self.dict:
                 str2idx[i] = self.dict[s[i]]
            else:
                str2idx[i] = 1
                
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
        one_hot = np.eye(self.classes, dtype='float32')
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
            donum = np.int32([donum])
            doent = np.int32([0 if not doent else int(doent)])
            do123 = np.int32([do123])
            dosym = np.int32([dosym])
            subnum =  np.int32([subnum])
            subdoent = np.int32([0 if not subdoent else int(subdoent)])
            
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
    
    def get_all_data_puredata(self):
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
        sizeeeeee = self.size//2
        
        for line in batch_txts:
            la, doa, donum, doent, do123, dosym, subdoa, subnum, subdoent = line
            # print(la, doa, donum, doent, do123, dosym, subdo, subnum,'\n')
            la = one_hot[int(la)]
            doa = self.str_to_indexes(doa, sizeeeeee)
            subdoa = self.str_to_indexes(subdoa, sizeeeeee)
            
            batch_indices.append(np.concatenate((doa, subdoa), axis = 0))
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
    
    
    def get_all_data_np(self):
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
        
        batchsize = 100
        
        print(self.typet + "data transfer starting...........")
        datasize = len(self.data)  
        start_index = 0
        end_index = datasize
        batch_txts = self.data[start_index:end_index]  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        batch_indices = []
        one_hot = np.eye(self.classes, dtype='int64')
        classes = []
        
        
        i_end = batchsize
        final_data_d = []
        final_data_l = []
        # what-self.data
        # what-batch_txts
        sizeeeeee = (self.size-6)//2
        for i in range(0, datasize+batchsize-1, batchsize):
            if  i_end > datasize:
                i_end = datasize
            for line in self.data[i:i_end]:
                la, doa, donum, doent, do123, dosym, subdoa, subnum, subdoent = line
                # print(la, doa, donum, doent, do123, dosym, subdo, subnum,'\n')

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
            i_end += batchsize
            if batch_indices:
                final_data_d.append(batch_indices)
                final_data_l.append(classes[i:i_end])
            batch_indices = []
            
        # tree/batch_indices
        print(self.typet + "data transfer finishing...........")   #12986

        # tree-final_data
        

        return final_data_d,final_data_l