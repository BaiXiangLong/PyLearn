#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
# 脚本名称：gethotword.py
# 脚本作用：统计热词
'''

import sys
import os
path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(path)
print path
import Analysis.FormatData
import Analysis.SQLconfig
from sklearn.feature_extraction.text import TfidfVectorizer
reload(sys)
sys.setdefaultencoding("utf-8")


def rungethotword():
    continue_flag = True
    while continue_flag:
        info = Analysis.SQLconfig.sql1.select('dcweb_article', ['title', 'content', 'id'], 'tag_mark = 0', 1500, None)
        num_info = len(info)
        if num_info == 0:
            print u"已完成全部热词统计"
            return
        if num_info < 1500:
            info1 = Analysis.SQLconfig.sql0.select('train', ['title', 'content', 'id'], None, 1500 - num_info, None)
            info += info1
            continue_flag = False
        print u"已读取%d条数据" % num_info
        Data_content = []
        Data_ID = []
        for date in info:
            Data_content.append(Analysis.FormatData.TextCut(date[0] * 2 + date[1]))
            Data_ID.append(date[2])
            print u"正在统计id = %d文章的关键字" % (date[2])
        # 读热词
        dict_hot_word = {}  # 所有的关键词列表
        hotwordlist = []  # 关键词列表
        hotword = Analysis.SQLconfig.sql1.select('dcweb_tag', ['name', 'number'], None, None, None)
        for word in hotword:
            dict_hot_word.update({word[0]: word[1]})
            hotwordlist.append(word[0])

        update_dic = {}
        # 读取停用词表
        stpwrdlst = Analysis.SQLconfig.sql0.select('stop_words', ['word'], None, None, None)

        '''用新文章更新热词词典，并记录新文章的热词'''
        dict = {}  # 中间变量
        dict_1 = []
        # 文章Tfidf化
        vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.2)
        tfidf = vectorizer.fit_transform(Data_content)
        word = vectorizer.get_feature_names()
        weight = tfidf.toarray()
        for i in range(num_info):  # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
            print u"-------这里输出第", i, u"类文本的词语tf-idf权重------"
            dict.clear()
            dict_1 = []
            for j in range(len(word)):
                if weight[i][j] > 0.1:  # 如果一个词的Tfidf值大于0.1才进入统计列表
                    dict.update({word[j]: weight[i][j]})
            dict_1 = sorted(dict.iteritems(), key=lambda asd: asd[1], reverse=True)
            dict.clear()
            for ii in dict_1:
                if len(dict) < 5:
                    dict.update({ii[0]: ii[1]})
                    if update_dic.has_key(ii[0]):
                        update_dic.update({ii[0]: 1000 * ii[1] + dict_hot_word.get(ii[0])})
                    else:
                        update_dic.update({ii[0]: 1000 * ii[1]})
            Analysis.SQLconfig.sql1.update('dcweb_article', {'tag_mark': 1}, 'id = %d' % Data_ID[i])

        # #######################热词存回数据库###############################
        i = 0
        dict_hot_word_list = sorted(update_dic.iteritems(), key=lambda asd: asd[1], reverse=True)
        for ii in dict_hot_word_list:
            i += 1
            print u"第%d条 : %s : %d\n" % (i, ii[0], ii[1])
            if dict_hot_word.has_key(ii[0]):
                dic = {'number': ii[1] + dict_hot_word.get(ii[0])}
                Analysis.SQLconfig.sql1.update('dcweb_tag', dic, 'name="%s"' % ii[0])
            else:
                dic = {'name': ii[0], 'number': ii[1]}
                Analysis.SQLconfig.sql1.add('dcweb_tag', dic)
        # #######################热词存回数据库结束##################################

if __name__ == "__main__":
    rungethotword()
