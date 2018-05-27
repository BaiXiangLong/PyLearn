#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
# 脚本名称：matchWordAndContent.py
# 脚本作用：将文章与热词联系在一起
"""
import sys
import os
path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(path)
print path
import Analysis.SQLconfig
import Analysis.FormatData
from sklearn.feature_extraction.text import TfidfVectorizer
reload(sys)
sys.setdefaultencoding("utf-8")


def runmatchWordAndContent():
    continue_flag = True
    while continue_flag:
        info = Analysis.SQLconfig.sql1.select('dcweb_article', ['id', 'title', 'content'], 'tag_mark = 1', 1500, None)
        numcontent = len(info)
        if numcontent == 0:
            print u"已完成全部热词统计"
            return
        if numcontent < 1500:
            continue_flag = False
            info1 = Analysis.SQLconfig.sql0.select('train', ['id', 'title', 'content'], None, 1500 - numcontent, None)
            info += info1
        print u"已读取%d条数据" % numcontent
        Data_content = []
        Data_ID = []
        for ii in info:
            Data_ID.append(ii[0])
            Data_content.append(Analysis.FormatData.TextCut(ii[1] * 2 + ii[2]))
            print u"读取id = %d文章内容" % ii[0]

        dict_hot_word = {}  # 所有的关键词列表
        hotwords = Analysis.SQLconfig.sql1.select('dcweb_tag', ['id', 'name'], None, None, None)
        for iii in hotwords:
            dict_hot_word.update({iii[1]: iii[0]})

        # 读取停用词表
        stpwrdlst = Analysis.SQLconfig.sql0.select('stop_words', ['word'], None, None, None)

        '''得到文章的关键词'''
        FeaturesWordList = []  # 文章的关键词列表
        dict = {}  # 中间变量
        dict_1 = []  # 中间变量
        # 文章Tfidf化
        vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.2)
        tfidf = vectorizer.fit_transform(Data_content)
        word = vectorizer.get_feature_names()
        weight = tfidf.toarray()
        for i in range(numcontent):  # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
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
            FeaturesWordList.append(dict.items())
        '''以上程序已经所有文章关键词的提取'''

        # #######################文章与关键词匹配###############################
        for ii in range(numcontent):
            for jj in FeaturesWordList[ii]:
                if dict_hot_word.has_key(jj[0]):
                    id_tag = dict_hot_word.get(jj[0])
                    value = jj[1] * 1000
                else:
                    continue
                if Data_ID[ii] is not None and id_tag is not None and value is not None:
                    dic = {'article_id': Data_ID[ii], 'tag_id': id_tag, 'value': value}
                    Analysis.SQLconfig.sql1.add('dcweb_article_tag', dic)
                    print u'文章号 : %d , 标签号 : %d , 重要度 : %d' % (Data_ID[ii], id_tag, value)
            Analysis.SQLconfig.sql1.update('dcweb_article', {'tag_mark': 2}, 'id = %d' % Data_ID[ii])
        # #######################文章与关键词匹配##################################

