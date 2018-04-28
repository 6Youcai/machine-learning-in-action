#!/usr/bin/env python3

import numpy as np
from sklearn import decomposition

def load_data(file_name):
    fr = open(file_name)
    content = [line.strip().split('\t') for line in fr]
    data = [list(map(float, line)) for line in content]
    return data

def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals #remove mean
    covMat = cov(meanRemoved, rowvar=0)
    eigVals,eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)            #sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat+1):-1]  #cut off unwanted dimensions
    redEigVects = eigVects[:,eigValInd]       #reorganize eig vects largest to smallest
    lowDDataMat = meanRemoved * redEigVects#transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat

def sklearn():
    # http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA
    model = decomposition.PCA(n_components = 0.85)
    data = load_data('testSet.txt')
    model.fit(data)
    print(model.explained_variance_)
    print(model.explained_variance_ratio_)
