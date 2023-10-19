"""
写一段用 Pytorch 跑深度学习的文本二分类代码, 数据为: `dataset/` 文件夹下的 dataset_test.csv dataset_train.csv dataset_validation.csv
这些数据的格式如下:
Domain,Label
upmysport.,0
megaupload.,0
qmvxa.,0
ytblxz.,1
0ymayzacmjda2bszg51aldbmo5fvprztaaaxxd5.,1
b91103d4aa00000000e97a8d6fed2cda6bde98b5da9d4cf49b9489da7521.f12a02c549c694a56f48dc192f5638dde914d1e1cbfbeb93b12d61c320ea.77741dafd2ec36e46b4d548227.,1
bbcmundo.,0
lzduda.,0
1ukarsemyqgm53me2nmbzpsy2qck0wwyedadbwx.,1
...以下省略

训练的时候可以把 Domain 最后的 `.` 去掉
"""

from boxx import *
import numpy as np
np.int = int

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

torch.random.manual_seed(0)

max_len = 64

# 加载数据集
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Domain'] = data['Domain'].apply(lambda x: x[:-1][-max_len:])
    return data

train_data = load_data('dataset/dataset_train.csv')
test_data = load_data('dataset/dataset_test.csv')
validation_data = load_data('dataset/dataset_validation.csv')

# 文本向量化
# vectorizer = TfidfVectorizer()
# X_train = vectorizer.fit_transform(train_data['Domain']).toarray()
# X_test = vectorizer.transform(test_data['Domain']).toarray()
# X_validation = vectorizer.transform(validation_data['Domain']).toarray()

ord_to_char = sorted(set(train_data.Domain.apply(list).sum()))
char_to_ord = dict(zip(ord_to_char, range(len(ord_to_char))))

# def domains_to_array(domains):
#     array = np.zeros((len(domains), max_len, len(char_to_ord)))
#     for data_idx, domain in enumerate(domains):
#         for char_idx, char in enumerate(domain):
#             array[data_idx, char_idx, char_to_ord[char]] = 1
#     return array

def domains_to_array(domains):
    array = np.zeros((len(domains), max_len, ), np.int32)
    for data_idx, domain in enumerate(domains):
        for char_idx, char in enumerate(domain):
            array[data_idx, char_idx, ] = char_to_ord[char]
    return array
X_train = domains_to_array(train_data['Domain'])
X_test = domains_to_array(test_data['Domain'])
X_validation = domains_to_array(validation_data['Domain'])

y_train = train_data['Label'].values
y_test = test_data['Label'].values
y_validation = validation_data['Label'].values

# 创建自定义 Dataset
class TextDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)

BATCH_SIZE = 64

train_dataset = TextDataset(X_train, y_train)
test_dataset = TextDataset(X_test, y_test)
validation_dataset = TextDataset(X_validation, y_validation)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=1)

# 创建简单的深度学习模型
# class TextClassifier(nn.Module):
#     def __init__(self, input_dim):
#         super(TextClassifier, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 32)
#         self.fc4 = nn.Linear(32, 1)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = torch.relu(self.fc3(x))
#         x = torch.sigmoid(self.fc4(x))
#         return x
    
class TextClassifier(nn.Module):
    def __init__(self, input_dim, embedding_dim=16):
        super(TextClassifier, self).__init__()
        self.embed = nn.Embedding(len(char_to_ord), embedding_dim)
        chage_dim = lambda c: c
        self.fc1 = nn.Linear(input_dim*embedding_dim, chage_dim(128))
        self.fc2 = nn.Linear(chage_dim(128), chage_dim(64))
        self.fc3 = nn.Linear(chage_dim(64), chage_dim(32))
        self.fc4 = nn.Linear(chage_dim(32), 1)

    def forward(self, x):
        embeds = self.embed(x.long())
        x = embeds.view(len(x), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x
    
    def is_evil(self, domains):
        if isinstance(domains, (str, bytes)):
            domains = [domains]
        x = torch.from_numpy(domains_to_array(domains))
        return self(x.long().to(device))

device = "cuda"
input_dim = X_train.shape[1]
model = TextClassifier(input_dim).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
NUM_EPOCHS = 100

# 模型评估
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            y_pred = model(X_batch.float().to(device)).squeeze()
            y_pred = (y_pred > 0.5).float()
            correct += (y_pred == y_batch.float().to(device)).sum().item()
            total += y_batch.size(0)
    return correct / total

for epoch in range(NUM_EPOCHS):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch.float().to(device)).squeeze()
        loss = criterion(y_pred, y_batch.float().to(device))
        loss.backward()
        optimizer.step()
    print("epoch: %02d, val_acc: %.2f%%" % (epoch, evaluate(model, validation_loader)* 100))

#%% 
train_accuracy = evaluate(model, train_loader)
test_accuracy = evaluate(model, test_loader)
validation_accuracy = evaluate(model, validation_loader)

print(f"Training accuracy: {train_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Validation accuracy: {validation_accuracy:.4f}")

for i, row in test_data.iterrows():
    prob = model.is_evil(row.Domain).item()
    print(f"Domain: % 20s, Prob:% 7.2f%%, Pred: {int(prob>.5)}, Lable: {row.Label}, Right: {int(prob>.5)==row.Label}" % (row.Domain[-20:], prob*100))
    if i>10: break

