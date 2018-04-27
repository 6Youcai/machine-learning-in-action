#!/usr/bin/env python3

import numpy as np
from sklearn import linear_model

def load_data():
    data = []
    label = []
    fr = open("testSet.txt")
    for line in fr:
        line_arr = line.strip().split('\t')
        data.append([1.0, float(line_arr[0]), float(line_arr[1])])
        label.append(int(line_arr[2]))
    return data, label

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def grad_ascent(data, label):
    data_mat = np.mat(data)
    label_mat = np.mat(label).T
    m, n = data_mat.shape
    alpha = 0.001
    max_step = 500
    weights = np.ones((n, 1))
    for k in range(max_step):
        h = sigmoid(data_mat * weights)
        error = label_mat - h
        # 梯度上升？
        weights = weights + alpha * data_mat.T * error
    return weights

def stoc_grad_ascent(data_mat, label_mat):
        # 随机梯度上升
        pass

def sklearn():
    # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    model = linear_model.LogisticRegression()
    x_train, y_train = load_data()
    model.fit(x_train, y_train)
    y_hat = model.predict(x_train)
    print(y_hat)
