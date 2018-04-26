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
    return np.mat(data_mat)

def euler_distance(vect_a, vect_b):
    # 欧拉距离
    return np.sqrt(np.sum(np.power(vect_a - vect_b, 2)))

def rand_cent(data_set, k):
    # 随机选取
    n = data_set.shape[1]
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        min_j = min(data_set[:, j])
        max_j = max(data_set[:, j])
        range_j = float(max_j - min_j)
        centroids[:, j] = min_j + range_j * np.random.rand(k, 1)
    return centroids

def kMeans(data_set, k, distance_fn = euler_distance, cent_fn = rand_cent):
    m = data_set.shape[0]
    cluster_assment = np.mat(np.zeros((m, 2))) ## index, distance
    # 初始中心点
    centroids = cent_fn(data_set, k)
    cluster_changed = True
    # 迭代至不再变化
    while cluster_changed:
        cluster_changed = False
        for i in range(m):
            min_dist = np.inf
            min_index = -1
            # 寻找最近的质心点
            for j in range(k):
                dist_JI = distance_fn(centroids[j, :], data_set[i, :])
                if dist_JI < min_dist:
                    min_dist = dist_JI
                    min_index = j
            if cluster_assment[i, 0] != min_index:
                cluster_changed = True
            cluster_assment[i, :] = min_index, min_dist ** 2
        # 更换质心
        for cent in range(k):
            pst_in_clust = data_set[np.nonzero(cluster_assment[:, 0].A == cent)[0]]
            centroids[cent, :] = np.mean(pst_in_clust, axis = 0) # 所有点平均值
    return centroids, cluster_assment

def biKmeans(dataSet, k, distMeas=euler_distance):
    pass
