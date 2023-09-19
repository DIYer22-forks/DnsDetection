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