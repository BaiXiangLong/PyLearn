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
print path
from sklearn.datasets.base import Bunch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB  # 导入多项式贝叶斯算法
import Analysis.SQLconfig
import Analysis.FormatData

reload(sys)
sys.setdefaultencoding("utf-8")


def RunClassify():
    """分类算法入口
        :return: 无
    """
    # 提取数据+分词+建立Bunch数据
    bunch = Bunch(contents=[], label=[])
    info = Analysis.SQLconfig.sql0.select('train', ['content', 'category_id'], None, None, None)
    for ii in info:
        ii[0] = Analysis.FormatData.TextCut(ii[0])
        bunch.contents.append(ii[0])
        bunch.label.append(ii[1])
    stpwrdlst = Analysis.SQLconfig.sql0.select('stop_words', ['word'], None, None, None)
    tfidfspace = Bunch(label=bunch.label, tdm=[], vocabulary={})
    vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.5)
    tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
    tfidfspace.vocabulary = vectorizer.vocabulary_
    clf = MultinomialNB(alpha=0.001).fit(tfidfspace.tdm, tfidfspace.label)

    continue_flag = True
    while continue_flag:
        test_bunch = Bunch(contents=[], id=[])
        info = Analysis.SQLconfig.sql0.select('article', ['id', 'content'], 'category is null', 1500, None)
        # info = Analysis.SQLconfig.sql1.select('dcweb_article', ['id', 'content'], 'category_id is null', 1500, None)
        num_info = len(info)
        if num_info == 0:
            print u"已完成全部分类"
            return
        if num_info < 1500:
            info1 = Analysis.SQLconfig.sql0.select('train', ['id', 'content'], None, 1500-num_info, None)
            info += info1
            continue_flag = False
        for ii in info:
            test_bunch.id.append(ii[0])
            test_bunch.contents.append(ii[1])
        print u"已读%d条数据" % num_info

        test_tfidfspace = Bunch(id=test_bunch.id, tdm=[], vocabulary={})
        test_tfidfspace.vocabulary = tfidfspace.vocabulary
        test_vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.5,
                                          vocabulary=tfidfspace.vocabulary)
        test_tfidfspace.tdm = test_vectorizer.fit_transform(test_bunch.contents)
        predicted = clf.predict(test_tfidfspace.tdm)

        i = 0
        for article_id, expect_cate in zip(test_tfidfspace.id, predicted):
            i = i + 1
            if i > num_info:
                break
            dic = {'category': expect_cate}
            Analysis.SQLconfig.sql0.update('article', dic, 'id = %d' % article_id)
            # Analysis.SQLconfig.sql1.update('dcweb_article', dic, 'id = %d' % article_id)
            print u"存第%s条数据！" % i

if __name__ == '__main__':
    RunClassify()
