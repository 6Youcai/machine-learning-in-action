#!/usr/bin/env python3

import numpy as np

def load_data(file_name):
    num_feat = len(open(file_name).readline().split('\t')) - 1
    data_mat = []
    lable_mat = []
    fr = open(file_name)
    for line in fr:
        line_arr = []
        curr_line = line.strip().split('\t')
        for i in range(num_feat):
            line_arr.append(float(curr_line[i]))
        data_mat.append(line_arr)
        lable_mat.append(float(curr_line[-1]))
    return data_mat, lable_mat

def stand_regress(xarr, yarr):
    x_mat = np.mat(xarr)
    y_mat = np.mat(yarr).T
    xTx = x_mat.T * x_mat
    # 行列式
    if np.linalg.det(xTx) == 0.0:
        return
    ws = xTx.I * (x_mat.T * y_mat)
    return ws

def lwlr(test_point, xarr, yarr, k = 1.0):
    pass

def lwlr_test(test_arr, xarr, yarr, k = 1.0):
    pass

def rss_error(yarr, yhat_arr):
    # 误差平方和
    return ((yarr - yhat_arr)**2).sum()

# 岭回归

# lasso

# 
