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

# accuarcy = input("please inter:")

accuarcy= 0.92
# accuarcy= 0.81

p/accuarcy
file_name = str(datetime.now().strftime('%y.%m.%d %H.%M') ) + '_' + str(accuarcy) 

with open(file_name, 'w', encoding = 'utf-8') as f:
    f.write(file_name + '.txt')
    
    for i in curr_files:
        if '.py' in i and 'chang' not in i:
            f.write(i + '\n' + '\n')
            with open(i, 'r', encoding = 'utf-8') as fpy:
                for line in fpy:
                    # p/line
                    # f.write(line)
                    pass
                f.write('\n')

            
shutil.copy('./final_model.pth', './' + file_name +'_' + 'model.pth')