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
    x_mat = np.mat(xarr)
    y_mat = np.mat(yarr).T
    # 局部加权线性回归
    m = x_mat.shape[0]
    weights = np.mat(np.eye((m)))
    for j in range(m):
        diff_mat = test_point - x_mat[j, :]
        weights[j, j] = np.exp(diff_mat * diff_mat.T / (-2.0 * k ** 2))
    xTx = x_mat.T * (weights * x_mat)
    if np.linalg.det(xTx) == 0:
        return
    ws = xTx.I * (x_mat.T * (weights * y_mat))
    return test_point * ws

def lwlr_test(test_arr, xarr, yarr, k = 1.0):
    m = np.shape(test_arr)[0]
    y_hat = np.zeros(m)
    for i in range(m):
        y_hat[i] = lwlr(test_arr[i], xarr, yarr, k)
    return y_hat

def rss_error(yarr, yhat_arr):
    # 误差平方和
    return ((yarr - yhat_arr)**2).sum()

def ridge_regress(x_mat, y_mat, lam = 0.2):
    # 岭回归
    xTx = x_mat.T * x_mat
    denom = xTx + np.eye(x_mat.shape[1]) * lam
    if np.linalg.det(denom) == 0.0:
        return
    ws = denom.I * (x_mat.T * y_mat)
    return ws

def ridge_test(xarr, yarr):
    x_mat = np.mat(xarr)
    y_mat = np.mat(yarr).T
    y_mean = np.mean(y_mat, 0)
    y_mat = y_mat - y_mean
    x_mean = np.mean(x_mat, 0)
    x_var = np.var(x_mat, 0)
    x_mat = (x_mat - x_mean) / x_var
    num_test = 30
    w_mat = np.zeros((num_test, x_mat.shape[1]))
    for i in range(num_test):
        ws = ridge_regress(x_mat, y_mat, np.exp(i - 10))
        w_mat[i: ] = ws.T
    return w_mat

# lasso
# 前向逐步回归
