#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
脚本名称：Classification.py
脚本用作用：分类
"""
import sys
import os
path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(path)
import SQLconfig


def collect_train_data():
    """
    从服务器数据库拉去数据
    :return: 
    """
    info = SQLconfig.sql1.select(u'dcweb_article', [u'content', u'category_id', u'title', u'dcweb_article.desc'])
    for ii in info:
        dic = {u'content': ii[0], u'category_id': ii[1], u'title': ii[2], u'traindata.desc': ii[3]}
        SQLconfig.LocalSql.add(u'traindata', dic)


def sort_train_data():
    """
    数据排序
    :return: 无 
    """
    for ii in [1, 2, 3, 7, 8, 10, 11, 12]:
        info = SQLconfig.LocalSql.select(u'traindata', [u'content', u'title', u'traindata.desc'], 'category_id=%d' % ii
                                         , None, None)
        for iii in info:
            dic = {u'content': iii[0], u'category_id': ii, u'title': iii[1], u'testdata_temp.desc': iii[2]}
            SQLconfig.LocalSql.add(u'testdata_temp', dic)


def delete_train_data(num):
    """
    删除多余的数据
    :param num: 最大训练数据量
    :return: 无
    """
    dic = {'1': 0, '2': 0, '3': 0, '7': 0, '8': 0, '10': 0, '11': 0, '12': 0}
    info = SQLconfig.LocalSql.select(u'testdata_temp', [u'id', u'content', u'category_id', u'title',
                                                        u'testdata_temp.desc'])
    for ii in info:
        if dic['%s' % ii[2]] > num:
            SQLconfig.LocalSql.delete(u'testdata_temp', 'id = %d' % ii[0])
        else:
            dic['%s' % ii[2]] += 1


def delete_repeat():
    """
    数据去重
    :return: 无 
    """
    content = []
    index = 0
    while True:
        data = SQLconfig.LocalSql.select('testdata_temp', ['id', 'content'], None, 1, index)
        if len(data) == 0:
            break
        if data[1] in content:
            SQLconfig.LocalSql.delete('testdata_temp', 'id=%d' % data[0])
        else:
            content.append(data[1])
        index += 1


def trans_train_data():
    """
    从服务器数据库拉去数据
    :return: 
    """
    info = SQLconfig.LocalSql.select(u'testdata', [u'content', u'category_id', u'title', u'testdata.desc'])
    for ii in info:
        dic = {u'content': ii[0], u'category_id': ii[1], u'title': ii[2], u'testdata_temp': ii[3]}
        SQLconfig.LocalSql.add(u'testdata_temp', dic)

if __name__ == '__main__':
    sort_train_data()

