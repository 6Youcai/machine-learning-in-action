#!/usr/bin/env python3

import numpy as np
from sklearn import datasets
from sklearn import naive_bayes

def load_date_set():
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1] # 人工标注
    return posting_list, class_vec

def create_vocab_list(data_set):
    vocab_set = set([])
    for document in data_set:
        # 集合求并集
        vocab_set = vocab_set | set(document)
    return list(vocab_set)

def set_of_words_to_vec(vocab_list, input_set):
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            # 有还是没有
            return_vec[vocab_list.index(word)] = 1
        else:
            print("this word %s is not in my vocabulary" %word)
    return return_vec

def bag_of_word(vocab_list, input_set):
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        # 词袋模式
        return_vec[vocab_list.index(word)] += 1
    return return_vec

def train(train_matrix, train_category):
    num_train = len(train_matrix) # 训练集个数
    num_words = len(train_matrix[0]) # 词汇表大小
    p_abuse = np.sum(train_category) / float(num_train) #
    p0_num = np.ones(num_words); p1_num = np.ones(num_words) ##
    p0_denom = 2.0; p1_denom = 2.0 ##
    for i in range(num_train):
        # 侮辱类
        if train_category[i] == 1:
            p1_num += train_matrix[i]
            p1_denom += np.sum(train_matrix[i])
        # 非侮辱类
        else:
            p0_num += train_matrix[i]
            p0_denom += np.sum(train_matrix[i])
    p1_vect = np.log(p1_num / p1_denom) ##
    p0_vect = np.log(p0_num / p0_denom) ##
    return p0_vect, p1_vect, p_abuse

def classify(vect_to_classify, p0_vec, p1_vec, p_class1):
    p1 = np.sum(vect_to_classify * p1_vec) + np.log(p_class1)
    p0 = np.sum(vect_to_classify * p0_vec) + np.log(1.0 - p_class1)
    if p1 > p0:
        return 1
    else:
        return 0

def testing_NB():
    list_of_post, list_class = load_date_set()
    my_vocab_list = create_vocab_list(list_of_post)
    train_matrix = []
    for post_in_doc in list_of_post:
        train_matrix.append(set_of_words_to_vec(my_vocab_list, post_in_doc))
    p0v, p1v, pab = train(np.array(train_matrix), np.array(list_class))

    test_entry = ['love', 'my', 'dalmation']
    this_doc = np.array(set_of_words_to_vec(my_vocab_list, test_entry))
    print(test_entry, "classified as", classify(this_doc, p0v, p1v, pab))

def sk():
    # http://scikit-learn.org/stable/modules/naive_bayes.html
    iris = datasets.load_iris()
    model = naive_bayes.GaussianNB()
    model.fit(iris.data, iris.target)
    y_hat = model.predict(iris.data)
    print(y_hat)
