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