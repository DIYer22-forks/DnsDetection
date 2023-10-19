# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 14:47:17 2023

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
# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, random_state=42),
    SVC(gamma=2, C=1, random_state=42),
    GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
    DecisionTreeClassifier(max_depth=5, random_state=42),
    RandomForestClassifier(
        max_depth=5, n_estimators=10, max_features=1, random_state=42
    ),
    MLPClassifier(alpha=1, max_iter=1000, random_state=42),
    AdaBoostClassifier(random_state=42),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]



# X, y = make_classification(
#     n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
# )
# rng = np.random.RandomState(2)
# X += 2 * rng.uniform(size=X.shape)
# linearly_separable = (X, y)   # ax + b = y

# datasets = [
#     make_moons(noise=0.3, random_state=0),
#     make_circles(noise=0.2, factor=0.5, random_state=1),
#     linearly_separable,
# ]


train_file = "dataset/dataset_train.csv"
test_file = "dataset/dataset_test.csv"

train_data = DataProcess('train', train_file, alpha, dim, 2)  # 128
train_data.load_data()
train_dataset, train_label = train_data.get_all_data_np() 

test_data = DataProcess('test', test_file, alpha, dim, 2)  # 128
test_data.load_data()
test_dataset, test_label = test_data.get_all_data_np() 

# X_train, y_train = train_data.get_all_data_np()
# X_test, y_test = test_data.get_all_data()



figure = plt.figure(figsize=(27, 9))
# i = 1
# iterate over datasets
# for ds_cnt, ds in enumerate(datasets):
  # x: domain    ,     y : label
for batch_num in range(5):
    X_train, y_train = train_dataset[batch_num], train_label[batch_num]
    X_train, y_train = np.array(X_train), np.array( y_train)
    
    X_test, y_test = test_dataset[batch_num], test_label[batch_num]
    X_test, y_test = np.array(X_test), np.array( y_test)
    
    
    pca = PCA(n_components=2)
    X_train = pca.fit_transform(X_train)
    X_test = pca.fit_transform(X_test)


    x_min, x_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    ax = plt.subplot(len(X_train ), len(classifiers) + 1, batch_num)
    if batch_num == 0:
        ax.set_title("Input data")
#     # # Plot the training points
    ax.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap=cm_bright, edgecolors="k")
    # Plot the testing points
    ax.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k"
    )
    ax.set_xticks(())
    ax.set_yticks(())
    batch_num += 1

# #     # iterate over classifiers
#     for name, clf in zip(names, classifiers):
#         ax = plt.subplot(len(datasets), len(classifiers) + 1, batch_num)

#         clf = make_pipeline(StandardScaler(), clf)
#         clf.fit(X_train, y_train)
#         score = clf.score(X_test, y_test)
#         DecisionBoundaryDisplay.from_estimator(
#             clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5
#         )

#         # Plot the training points
#         ax.scatter(
#             X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
#         )
#         # Plot the testing points
#         ax.scatter(
#             X_test[:, 0],
#             X_test[:, 1],
#             c=y_test,
#             cmap=cm_bright,
#             edgecolors="k",
#             alpha=0.6,
#         )

#         # ax.set_xlim(x_min, x_max)
#         # ax.set_ylim(y_min, y_max)
#         ax.set_xticks(())
#         ax.set_yticks(())
#         if ds_cnt == 0:
#             ax.set_title(name)
#         # ax.text(
#         #     # x_max - 0.3,
#         #     # y_min + 0.3,
#         #     ("%.2f" % score).lstrip("0"),
#         #     size=15,
#         #     horizontalalignment="right",
#         # )
#         batch_num += 1

plt.tight_layout()
plt.show()

g()