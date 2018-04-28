#!/usr/bin/env python3

import numpy as np
from math import log
import operator
from sklearn import tree
from sklearn.datasets import load_iris

def shanno_ent(data):
    # 计算香农熵
    number = len(data)
    label_count = {}
    for feat in data:
        current_label = feat[-1]
        label_count[current_label] = label_count.get(current_label, 0) + 1
    entropy = 0.0
    for label in label_count:
        prob = float(label_count[label]) / number
        entropy -= prob * log(prob, 2)
    return entropy

def create_data():
    # q1_ans, q2_ans, label; questions
    data = [[1, 1, 'yes'], [1, 1, 'yes'],
            [1, 0, 'no'], [0, 1, 'no'],
            [0, 1, 'no']]
    questions = ['no-surfacing', 'flippers']
    return data, questions

def split_data(data, index, value):
    sub_data = []
    for row in data:
        # 一旦发现符合要求的值，添加之
        if row[index] == value:
            reduced_row = row[: index]
            reduced_row.extend(row[index+1: ])
            sub_data.append(reduced_row)
    return sub_data

def best_feature_to_split(data):
    number_feature = len(data[0]) - 1
    base_entropy = shanno_ent(data)
    best_info_gain = 0.0
    best_feature = -1
    # 用哪一个特征
    for i in range(number_feature):
        feat_list = [example[i] for example in data]
        unique_vals = set(feat_list)
        new_entropy = 0.0
        # 特征的每一个值
        for value in unique_vals:
            sub_data = split_data(data, i, value)
            prob = len(sub_data) / float(len(data))
            new_entropy += prob * shanno_ent(sub_data) # 新的熵
        # 信息增益是熵的减小，或是数据无序度的减少
        info_gain = base_entropy - new_entropy
        # 按照获取最大信息增益的方法划分数据集
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature

def majority_cnt(class_list):
    # 如果数据集已经处理了所有属性，但是类型标签仍不唯一，
    # 此时序要定义该叶子节点，采用多数表决法
    class_count = {}
    for vote in class_list:
        class_count[vote] = class_count.get(vote, 0) + 1
    sorted_count = sorted(class_count.items(), key = operator.itemgetter(1),
                            reverse = True)
    return sorted_count[0][0]

def create_tree(data, questions):
    class_list = [example[-1] for example in data]
    # 类别完全相同
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 遍历完成
    if len(data[0]) == 1:
        return majority_cnt(class_list)
    best_feature = best_feature_to_split(data) # best_feature_index
    best_question = questions[best_feature]
    my_tree = {best_question: {}}
    del questions[best_feature]
    best_values = [example[best_feature] for example in data]
    unique_vals = set(best_values)
    for value in unique_vals:
        sub_questions = questions[:] #
        # 划分数据集
        sub_data = split_data(data, best_feature, value)
        # 递归调用
        my_tree[best_question][value] = create_tree(sub_data, sub_questions)
    return my_tree

def classify(input_tree, questions, answers):
    first_str = list(input_tree)[0]
    second_dict = input_tree[first_str]
    feat_index = questions.index(first_str)
    for key in second_dict.keys():
        if answers[feat_index] == key:
            if type(second_dict[key]).__name__ == "dict":
                # 递归调用
                class_lable = classify(second_dict[key], questions, answers)
            else:
                class_lable = second_dict[key]
    return class_lable

def lens_data():
    fr = open("lenses.txt")
    data = [line.strip().split('\t') for line in fr]
    questions = ['age', 'prescript', 'astigmatic', 'tear_rate']
    return data, questions

def sk():
    # http://scikit-learn.org/dev/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
    model = tree.DecisionTreeClassifier()
    iris = load_iris()
    data, taget = iris.data, iris.target
    model.fit(data, taget)

    model = tree.DecisionTreeClassifier()
    data, label = create_data()
    data_in = np.mat(data)[:, :-1]
    target = np.mat(data)[:, -1]
    print(data_in, "\n\n\n\n\n", target)
    model.fit(data_in, target)
    y_hat = model.predict([['1', '0'], ['1', '1']])
    print(y_hat)
