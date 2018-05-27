#! /usr/bin/env python
# -*- coding: utf-8 -*-
import jieba
from networkx import from_scipy_sparse_matrix, pagerank
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
import sys
import os
path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(path)
print path

import Analysis.SQLconfig

def cut_sentence(sentence):
    """ 
    分句 
    :param sentence: 
    :return: 
    """
    if not isinstance(sentence, unicode):
        sentence = sentence.decode('utf-8')
    delimiters = frozenset(u'。！？')
    buf = []
    for ch in sentence:
        buf.append(ch)
        if delimiters.__contains__(ch):
            yield ''.join(buf)
            buf = []
    if buf:
        yield ''.join(buf)

def load_stopwords():
    """ 
    加载停用词 
    :param path: 
    :return: 
    """
    import os
    path = os.path.dirname(os.path.abspath(__file__)) + '/stop_words.txt'
    with open(path) as f:
        stopwords = f.readlines()
    stopwrdlist = []
    for i in stopwords:
        i1 = i.decode('gbk')
        i1 = i1.strip("\r\n")
        stopwrdlist.append(i1)
    return frozenset(stopwrdlist)

def cut_words(sentence):
    """ 
    分词 
    :param sentence: 
    :return: 
    """
    stpwrdlst = Analysis.SQLconfig.sql0.select('stop_words', ['word'], None, None, None)
    return filter(lambda x: not stpwrdlst.__contains__(x), jieba.cut(sentence))


def get_abstract(content, size=3):
    """ 
    利用textrank提取摘要 
    :param content: 
    :param size: 
    :return: 
    """
    docs = list(cut_sentence(content))
    stpwrdlst = Analysis.SQLconfig.sql0.select('stop_words', ['word'], None, None, None)
    tfidf_model = TfidfVectorizer(tokenizer=jieba.cut, stop_words=stpwrdlst)
    tfidf_matrix = tfidf_model.fit_transform(docs)
    normalized_matrix = TfidfTransformer().fit_transform(tfidf_matrix)
    similarity = from_scipy_sparse_matrix(normalized_matrix * normalized_matrix.T)
    scores = pagerank(similarity)
    tops = sorted(scores.iteritems(), key=lambda x: x[1], reverse=True)
    size = min(size, len(docs))
    indices = map(lambda x: x[0], tops)[:size]
    return map(lambda idx: docs[idx], indices)


def RunAbstract():
    """
    摘要的主函数
    :return: 无
    """
    continue_flag = True
    # 取数据操作的改变，直接搜索语句就好了
    while continue_flag:
        info = Analysis.SQLconfig.sql0.select('article', ['id', 'content'], 'article.desc is null', 1000, None)
        if len(info) == 0:
            print u"已完成全部摘要"
            return
        if len(info) < 1000:
            continue_flag = False
        print u"已读取%d条数据" % len(info)
        for ii in info:
            if len(ii[1]) < 2:
                print u"id = %d 文章太短不必要摘要" % ii[0]
                continue
            s = get_abstract(ii[1])
            s[0] = s[0].strip('\r\n')
            s[0] = s[0].strip(' ')
            s[0] = s[0].replace(' ', '')
            s[0] = s[0].replace('\n', '')
            s[0] = s[0].replace('\r', '')
            s[0] = s[0].replace('\t', '')
            dic = {'article.desc': s[0]}
            Analysis.SQLconfig.sql0.update('article', dic, 'id=%d' % ii[0])
            print u'id = %d : %s' % (ii[0], s[0])

if __name__ == "__main__":
    RunAbstract()
