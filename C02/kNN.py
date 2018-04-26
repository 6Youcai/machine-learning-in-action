#!/usr/bin/env python3

from numpy import *
import operator
import os
from sklearn import neighbors
from sklearn import datasets

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ["A", "A", "B", "B"]
    return group, labels

def classify0(inx, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inx, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 ##
    sortedClassCount = sorted(classCount.items(),
                                key = operator.itemgetter(1),
                                reverse = True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1])) #
        index += 1
    return returnMat, classLabelVector

def autoNorm(dataSet):
    # min-max标准化
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet / tile(ranges, (m,1))
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                    datingLabels[numTestVecs:m], 3)
        print("classifier: %d, real: %d" %(classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("the total error rate is: %f" %(errorCount / float(numTestVecs)))

def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabeels = []
    trainingFileList = listdir("trainingDigits")
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0] # 9_91.txt
        classNumber = int(fileStr.split('_')[0])
        hwLabeels.append(classNumber)
        trainingMat[i, :] = img2vector("trainingDigits/%s" % fileNameStr)

    testFileList = listdir("testDigits")
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumber = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector("testDigits/%s" %fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabeels, 3)
        print("file:%s, classifier: %d" %(fileNameStr, classNumber))
        if classifierResult != classNumber:
            errorCount += 1.0
    print("the total number of error is: %d" % errorCount)
    print("the total error rate is: %f" %(errorCount/float(mTest)))

##
def img2vector_2(dir_name, file_name):
    answer = int(file_name.split('_')[0])
    returnVect = zeros((1, 1024))
    fr = open(os.path.join(dir_name, file_name))
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    fr.close()
    return returnVect, answer

def creat_training(dir_name):
    training_files = os.listdir(dir_name)
    m = len(training_files)
    training_mat = zeros((m, 1024))
    training_label = []
    for i in range(m):
        file_name = training_files[i]
        mat, lable = img2vector_2(dir_name, file_name)
        training_mat[i, :]  = mat
        training_label.append(lable)
    return training_mat, training_label

def SK():
    knn = neighbors.KNeighborsClassifier()
    training_mat, training_labe = creat_training("trainingDigits")
    knn.fit(training_mat, training_labe)

    testFileList = listdir("testDigits")
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumber = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector("testDigits/%s" %fileNameStr)
        predicted_label = knn.predict(vectorUnderTest)[0]
        print("for %s, the predict answer is %d" %(fileNameStr, predicted_label))
