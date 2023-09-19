import pandas as pd
from boxx import *
import boxx
import matplotlib.pyplot as plt
import random

path = boxx.relfile("../../b/c.csv")


file_train_1 = "dataset_train_tunnel_zip/archive/dataset_train_tunnel.csv"
file_test_1 = "dataset_test_tunnel_zip/archive/dataset_test_tunnel.csv"
file_collect = "train.csv"
c = os.path.abspath(__file__)
d = os.path.dirname(c)
# p/c
# p/d
df1 = pd.DataFrame(pd.read_csv(file_train_1,header=0),columns=['Domain','Label'])
df2 = pd.DataFrame(pd.read_csv(file_test_1,header=0),columns=['Domain','Label'])
dfc = pd.DataFrame(pd.read_csv(file_collect,header=0),columns=['Domain','Label'])

def show(df1):
    
    df1.shape
    df1.info()
    df1.columns

def change(file_name1):
    df1 = pd.DataFrame(pd.read_csv(file_name1,header=0))
    show(df1)

    for index, rows in df1.iterrows():
        if "hackbiji" in rows['Info']:
            df1.loc[index,'label'] = '1'
        else :
            df1.loc[index,'label'] = '0'
        # p/rows
    show(df1)
    df1 = df1.fillna(value=0)
    show(df1)
    df1.to_csv(file_name1, index=False)      

def count0(file_name1):
    df1 = pd.DataFrame(pd.read_csv(file_name1,header=0))
    count0 = sum(df1['label'])
    p/count0



def merge():
    df1 = pd.DataFrame(pd.read_csv(file_name1,header=0))
    df2 = pd.DataFrame(pd.read_csv(file_name2,header=0))

    df2 = df2.append(df1)
    show(df2)
    df2.to_csv(file_name2, index=False)    


# change(file_name1)
# count0(file_name1)
# merge()


def select_merge(file_load):
    df = pd.DataFrame(pd.read_csv(file_name2,header=0), columns=['Source', 'Destination', 'Name', 'Source Port', 'label', 'Info'])
    show(df)
    df.to_csv('dataset_simple', index=False)    
    
# show(dfc)
# dfc = dfc.drop_duplicates()
# dfc = dfc.drop(dfc[dfc['Domain'] == '0'].index)
# # dfc = dfc[~dfc['Domain'].str.contains('<Root>')]
# dfc.to_csv('train.csv')

blacks = dfc[dfc['Label'] == 1]
writes = dfc[dfc['Label'] == 0]
blacks = blacks.sample(348)
df1 = df1.append(blacks)
df1 = df1.append(writes)
# df = blacks.append(writes)
# df.to_csv('dataset_train.csv', index=False)
# show(df2)
# df2 = df2.drop(df2[df2['Domain'] == '0'].index)
# df2 = df2[~df2['Domain'].str.contains('<Root>')]
# show(df2)
# # df = pd.DataFrame(df2, columns=['Name', 'label'])
# df2.drop_duplicates() 
# df2.to_csv('temp.csv', index=False) 
# show(df)
# # select_merge(file_name2)


# df_simple = pd.DataFrame(pd.read_csv("dataset_simple", header=0))
# domain_counts = df_simple['Name'].value_counts()
# top_10_domain = domain_counts.head(20)
# top_10_domain.plot(kind = 'bar')

# plt.xlabel('域名')
# plt.ylabel('数量')
# plt.show()


# df_simple = pd.DataFrame(pd.read_csv("dataset_simple", header=0))
# domain_counts = df_simple['label'].value_counts()
# top_10_domain = domain_counts.head(20)
# top_10_domain.plot(kind = 'bar')

# blacks = df_simple[df_simple['label'] == 1]
# writes = df_simple[df_simple['label'] == 0]

# n = 302
# blackst = blacks.sample(n)
# writest = writes.sample(n)

# blackste = blackst.copy()
# blackstr = blacks.drop(blackst.index).sample(3788-n)
# writeste = writest.copy()
# writestr = writes.drop(writest.index).sample(3788-n)

# dataset_train = blackstr.append(writestr)
# dataset_test = blackste.append(writeste)

# dataset_train.to_csv('dataset_train.csv')
# dataset_test.to_csv('dataset_test.csv')

# show(dataset_train)
# show(dataset_test)
# plt.xlabel('域名')
# plt.ylabel('数量')
# plt.show()


# missing_values = df_simple.isnull().sum()
# print(missing_values)


# duplicated_rows = df_simple.duplicated()
# print(duplicated_rows)




# summary_stats = df_simple.describe()
# print(summary_stats)






















