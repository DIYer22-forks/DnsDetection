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




class DataProcess(object):
    def __init__(self, data_file, alpha,
                 input_size, num_of_classes) -> None:
        '''
        args:
            data_file: csv file
            input_size: features
            num of classes: label
        '''
        self.data_file = data_file
        self.alpha = alpha
        self.alpha_size = len(alpha)
        self.input_size = input_size
        self.classes = num_of_classes
        self.dict = {}
        self.dict['unk'] = 0
        # one-hot编码
        for idx, alp in enumerate(self.alpha):
            self.dict[alp] = idx 
        self.size = input_size
        print("data loading..............")
    
    def caculate_entropy(self, s):
        lens = len(s)
        num_counts = Counter(s)
        entropy = 0
        for count in num_counts.values():
            probability = count/lens
            entropy += - probability * math.log2(probability)
        return entropy



    def load_data(self):
        data = []
        with open(self.data_file, 'r', encoding='utf-8', ) as f:
            rr = csv.reader(f, delimiter=",", quotechar='"')
            next(rr)
            for line in rr:
                txt = ''
                for s in line:
                    label = int(line[1])
                    # 先去掉最后一个.
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

                data.append([label, dotxt, dosize, entropy, Anum, nnum, subdotxt, subdosize])
        self.data = np.array(data)
        # p/data
        # p/self.data
        print("data load finishing..............")
        g()

    def str_to_indexes(self, s):
        """
        reverse abc to 123

        """
        s = s.lower()
        max_length = min(len(s), self.size)
        str2idx = np.zeros(self.size, dtype='int64')
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
        '''
        print("data transfer starting...........")
        datasize = len(self.data)  
        start_index = 0
        end_index = datasize
        batch_txts = self.data[start_index:end_index]
        batch_indices = []
        one_hot = np.eye(self.classes, dtype='int64')
        classes = []
        # what-self.data
        # what-batch_txts

        for line in batch_txts:
            la, doa, donum, doent, do123, dosym, subdo, subnum = line
            # print(la, doa, donum, doent, do123, dosym, subdo, subnum,'\n')
            la = one_hot[int(la)]
            doa = self.str_to_indexes(doa)
            donum = self.str_to_indexes(donum)
            doent = self.str_to_indexes(doent)
            do123 = self.str_to_indexes(do123)   #有点问题，考虑如果是12会转成1，2，但是如果是1就是1，十位和百位的差异，网络敏感性
            dosym = self.str_to_indexes(dosym)
            subdo = self.str_to_indexes(subdo)
            subnum = self.str_to_indexes(subnum)
            # print(la, doa, donum, doent, do123, dosym, subdo, subnum,'\n')
            batch_indices.append([doa, donum, doent, do123, dosym, subdo, subnum])
            classes.append(la)
        # tree/batch_indices
        # tree/classes
        print("data transfer finishing...........")   #12986



        # p/batch_txts[13]
        # tmp = self.str_to_indexes(batch_txts[0][1])
        # p/tmp
        return np.asarray(batch_indices, dtype='int64'), np.asarray(classes)
