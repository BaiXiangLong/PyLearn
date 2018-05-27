#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
脚本名称：Classification.py
脚本用作用：分类
备注：
提高分类准确率的方案：
1、换模型
    模型是描述问题的核心，如果有决心，我建议换模型
    >>更换词权重计算方法
    >>使用卷积神经网络
    在分类问题上，我认为现在的模型已经能足够描述这个问题，不需要再换模型了
2、训练数据角度，尽量提高训练数据量和训练数据质量。
    >>训练数据量：尽可能多，直到分类准确率进入平滑期
    >>训练数据质量：标签与文章内容尽量与大众思路一致
    训练数据确实是一个很大的问题，需要尽快解决
3、算法角度：
    >>尝试不同的分类器：朴素贝叶斯，SVM，决策树
    >>尝试信息融合算法，证据理论或者boosting或者adaboost
    >>使用不同分类器的加权，按照常规，准确率能提高2~3个百分点
    
步骤：改善数据质量>>改善算法
吐槽：手动标注数据真的很恶心，很难受
多种分类算法加权融合得到的分类器算法时间复杂度很高，是只使用朴素贝叶斯算法时间复杂度的1000倍！！
正在发现时间复杂度小的算法加以使用。
基于
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import Analysis.Classify_Method

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(path)
from sklearn.datasets.base import Bunch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB  # 导入多项式贝叶斯算法
import Analysis.SQLconfig
import Analysis.FormatData

reload(sys)
sys.setdefaultencoding("utf-8")


def ds_envidence_theory(m1, m2):
    """
    基于D-S证据理论的多分类器结果融合计算
    :param m1: 第一个概率分配函数
    :param m2: 第二个概率分配函数
    :return: 融合后的概率分配函数
    """
    m1 = np.array(m1)
    m2 = np.array(m2)
    re = []  # 融合后的概率分配函数
    n1 = len(m1)
    n2 = len(m2)
    if n1 != n2:
        re = np.zeros(n1)
        re[n1 - 1] = 1
    else:
        for ind in range(n1):
            if ind == n1 - 1:
                re.append(m1[ind] * m2[ind])
            else:
                re.append(m1[ind] * m2[ind] + m1[ind] * m2[n1 - 1] + m1[n1 - 1] * m2[ind])
        re = np.array(re)
        re = re/np.sum(re)
    return re


def test_classify(data, RealLabel, dic, Classification):
    """
    分类器
    :param data: 数据 
    :param RealLabel: 真是类别
    :param Classification: 分类器
    :return: 分类正确率
    """
    test_bunch = Bunch(content=[], label=[])
    test_tfidfspace = Bunch(tdm=[], vocabulary={})
    stpwrdlst = Analysis.SQLconfig.sql0.select('stop_words', ['word'], None, None, None)
    test_tfidfspace.vocabulary = dic.vocabulary
    test_vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.5,
                                      vocabulary=dic.vocabulary)
    for ii1, ii2 in zip(data, RealLabel):
        ii1 = Analysis.FormatData.TextCut(ii1)
        test_bunch.content.append(ii1)
        test_bunch.label.append(ii2)
    test_tfidfspace.tdm = test_vectorizer.fit_transform(test_bunch.content)
    predicted = Classification.predict(test_tfidfspace.tdm)
    return predicted


def train_classification(data, label, method='NB'):
    """
    训练分类器
    :param data: 训练数据的属性
    :param label: 训练数据的标签
    :param method: 构建分类器的方法，‘NB’:朴素贝叶斯，'SVM'：支持向量机,‘RFC’:随机森林，‘DTC’：决策树，‘SVMCV’：加强版SVM
    :return: 
    """
    bunch = Bunch(contents=[], label=[])
    for ii, ii1 in zip(data, label):
        ii = Analysis.FormatData.TextCut(ii)
        bunch.contents.append(ii)
        bunch.label.append(ii1)
    stpwrdlst = Analysis.SQLconfig.sql0.select('stop_words', ['word'], None, None, None)
    tfidfspace = Bunch(label=bunch.label, tdm=[], vocabulary={})
    vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.5)
    tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
    tfidfspace.vocabulary = vectorizer.vocabulary_
    if method == 'NB':
        clf = Analysis.Classify_Method.naive_bayes_classifier(tfidfspace.tdm, tfidfspace.label)
    elif method == 'KNN':
        clf = Analysis.Classify_Method.knn_classifier(tfidfspace.tdm, tfidfspace.label)
    elif method == 'DTC':
        clf = Analysis.Classify_Method.decision_tree_classifier(tfidfspace.tdm, tfidfspace.label)
    elif method == 'SVM':
        clf = Analysis.Classify_Method.svm_classifier(tfidfspace.tdm, tfidfspace.label)
    elif method == 'SVMCV':
        clf = Analysis.Classify_Method.svm_cross_validation(tfidfspace.tdm, tfidfspace.label)
    elif method == 'LR':
        clf = Analysis.Classify_Method.logistic_regression_classifier(tfidfspace.tdm, tfidfspace.label)
    elif method == 'GB':
        clf = Analysis.Classify_Method.gradient_boosting_classifier(tfidfspace.tdm, tfidfspace.label)
    elif method == 'RF':
        clf = Analysis.Classify_Method.random_forest_classifier(tfidfspace.tdm, tfidfspace.label)
    else:
        clf = Analysis.Classify_Method.naive_bayes_classifier(tfidfspace.tdm, tfidfspace.label)
    return [clf, tfidfspace]


def vote():
    corr = []
    info = Analysis.SQLconfig.LocalSql.select('testdata', ['content', 'category_id', 'title', 'testdata.desc'])
    data = {'1': [], '2': [], '3': [], '7': [], '8': [], '10': [], '11': [], '12': []}
    sort_category = {'0': 1, '1': 2, '2': 3, '3': 7, '4': 8, '5': 10, '6': 11, '7': 12}
    for ii in info:
        data['%s' % (ii[1])] += [ii[2] * 3 + ii[3] + ii[0]]
        loop = 5
    # for train_number in range(10, 190, 10):
    for train_number in [190]:
        accuracy_rate = 0.
        for loop_time in range(loop):
            train_data = []
            train_label = []
            test_data = []
            test_label = []
            for ii in data.keys():
                data_list = data[ii]
                train_index = np.random.permutation(len(data_list))
                for iii in train_index[0: train_number]:
                    train_data.append(data_list[iii])
                    train_label.append(int(ii))
                for iii in train_index[train_number:]:
                    test_data.append(data_list[iii])
                    test_label.append(int(ii))
            classify_method = ['NB', 'KNN', 'DTC', 'SVM', 'SVMCV', 'LR', 'GB', 'RF']
            predict = []
            all_predict = []
            final_predict = []
            for method_ in classify_method:
                [classification, dic] = train_classification(train_data, train_label, method_)
                predict.append(test_classify(test_data, test_label, dic, classification))
            for ii in range(len(predict)):
                for iii in range(len(predict[ii])):
                    if ii == 0:
                        all_predict.append([])
                    all_predict[iii].append(predict[ii][iii])
            from collections import Counter
            for ii in all_predict:
                final_predict.append(Counter(ii).most_common(1)[0][0])
                final_predict_np = np.array(final_predict)
                test_label_np = np.array([test_label])
                corr_np = final_predict_np == test_label_np
            accuracy_rate += float(corr_np.sum()) / float(len(final_predict)) / loop
        print u'训练数据量为%d，准确率为%lf' % (train_number, accuracy_rate)
        corr.append(accuracy_rate)
    fig = plt.plot(corr)
    plt.show(fig)


def get_train_and_test_data(origin_data, train_number_):
    train_data_ = []
    train_label_ = []
    test_data_ = []
    test_label_ = []
    for ii_ in origin_data.keys():
        data_list_ = origin_data[ii_]
        train_index_ = np.random.permutation(len(data_list_))
        for iii_ in train_index_[0: train_number_]:
            train_data_.append(data_list_[iii_])
            train_label_.append(int(ii_))
        for iii_ in train_index_[train_number_:]:
            test_data_.append(data_list_[iii_])
            test_label_.append(int(ii_))
    return [train_data_, train_label_, test_data_, test_label_]


def get_classificaton_accuracy(_train_data, _train_label, _test_data, _test_label, _method_='NB', loop_time=3):
    """
    获得分类器的准确度
    :param _train_data: 训练数据的属性
    :param _train_label: 训练数据的标签
    :param _test_data: 测试数据的属性
    :param _test_label: 测试数据的标签
    :param _method_: 分类器方法
    :param loop_time: 循环次数
    :return: 分类器的准确率
    """
    accuracy_ = 0
    for loop_ in range(loop_time):
        [classification_, dic_] = train_classification(_train_data, _train_label, _method_)
        pre_ = test_classify(_test_data, _test_label, dic_, classification_)
        accuracy_ += get_accuracy(pre_, test_label) / loop_time
    return accuracy_


def get_accuracy(_predict, real):
    if len(_predict) != len(real):
        return 0.
    _pre_np = np.array(_predict)
    _real_np = np.array(real)
    _acc_np = _pre_np == _real_np
    return float(_acc_np.sum())/float(len(_pre_np))

if __name__ == '__main__':
    corr = []
    info = Analysis.SQLconfig.LocalSql.select('testdata', ['content', 'category_id', 'title', 'testdata.desc'])
    data = {'1': [], '2': [], '3': [], '7': [], '8': [], '10': [], '11': [], '12': []}
    sort_category = {'0': 1, '1': 2, '2': 3, '3': 7, '4': 8, '5': 10, '6': 11, '7': 12}
    category_sort = {'1': 0, '2': 1, '3': 2, '7': 3, '8': 4, '10': 5, '11': 6, '12': 7}
    for ii in info:
        data['%s' % (ii[1])] += [ii[2]*3 + ii[3] + ii[0]]
    loop = 1
    # classify_method = ['NB', 'KNN', 'DTC', 'SVM', 'SVMCV', 'LR', 'GB', 'RF']
    classify_method = ['NB', 'KNN', 'LR']
    accuracy_method = [0.8358613761928678, 0.7308890005022602, 0.8226017076845806]
    """
    for ii in classify_method:
        acc_ = 0
        for iii in range(loop):
            [train_data, train_label, test_data, test_label] = get_train_and_test_data(data, 190)
            [classification, dic] = train_classification(train_data, train_label, ii)
            pre = test_classify(test_data, test_label, dic, classification)
            acc_ += get_accuracy(pre, test_label)/loop
        accuracy_method.append(acc_)
    print accuracy_method
    """
    # for train_number in range(10, 190, 10):
    for train_number in [190]:
        accuracy_rate = 0.
        for loop_time in range(loop):
            [train_data, train_label, test_data, test_label] = get_train_and_test_data(data, train_number)
            predict = []
            all_predict = []
            final_predict = []
            for method_ in classify_method:
                [classification, dic] = train_classification(train_data, train_label, method_)
                predict.append(test_classify(test_data, test_label, dic, classification))
            for ii in range(len(predict)):
                for iii in range(len(predict[ii])):
                    if ii == 0:
                        all_predict.append([])
                    all_predict[iii].append(predict[ii][iii])
            for ii in all_predict:
                m = np.zeros(len(sort_category) + 1)
                m[len(sort_category)] = 1.
                for iii in range(len(ii)):
                    m1 = np.zeros(len(sort_category) + 1)
                    m1[category_sort.get('%d' % ii[iii])] = accuracy_method[iii]
                    m1[len(sort_category)] = 1.0 - m1.sum()
                    m = ds_envidence_theory(m, m1)
                index_max = np.where(np.max(m))[0][0]
                final_predict.append(sort_category.get('%d' % index_max))
            print final_predict
            print test_label
            final_predict_np = np.array(final_predict)
            test_label_np = np.array([test_label])
            corr_np = final_predict_np == test_label_np
            accuracy_rate += float(corr_np.sum())/float(len(final_predict))/loop
        print u'训练数据量为%d，准确率为%lf' % (train_number, accuracy_rate)
        corr.append(accuracy_rate)
    fig = plt.plot(corr)
    plt.show(fig)
