#!/usr/bin/env python3

import numpy as np

def load_data(file_name):
    data_mat = []
    fr = open(file_name)
    for line in fr:
        cur_line = line.strip().split('\t')
        flt_line = list(map(float, cur_line))
        data_mat.append(flt_line)
    fr.close()
    return data_mat

def distance_OL(vect_a, vect_b):
    # 欧拉距离
    return np.sqrt(np.sum(np.power(vect_a - vect_b, 2)))

def rand_cent(data_set, k):
    # 随机中心点
    n = data_set.shape[1]
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        min_j = min(data_set[:, j])
        max_j = max(data_set[:, j])
        range_j = float(max_j - min_j)
        centroids[:, j] = min_j + range_j * np.random.rand(k, 1)
    return centroids

def kMeans(data_set, k, distance_fn = distance, cent_fn = rand_cent):
    pass
