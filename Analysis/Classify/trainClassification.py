#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
脚本名称：trainClassification.py
脚本作用：训练分类器
"""
import sys
import os
path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(path)
print path
import Analysis.FormatData
from sklearn.datasets.base import Bunch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB  # 导入多项式贝叶斯算法
import Analysis.SQLconfig
reload(sys)
sys.setdefaultencoding("utf-8")


# 提取数据+分词+建立Bunch数据
bunch = Bunch(contents=[], label=[])
info = Analysis.SQLconfig.sql0.select('train', ['content', 'category_id'], None, None, None)
for ii in info:
    ii[0] = Analysis.FormatData.TextCut(ii[0])
    bunch.contents.append(ii[0])
    bunch.label.append(ii[1])
# 提取数据+分词+建立Bunch数据结束

# 读取停用词
stpwrdlst = Analysis.SQLconfig.sql0.select('stop_words', ['word'], None, None, None)

# 构建tf-idf词向量空间对象
tfidfspace = Bunch(label=bunch.label, tdm=[], vocabulary={})

# 构建使用TfidfVectorizer初始化向量空间模型
# 这里面有TF-IDF权重矩阵还有我们要的词向量空间坐标轴信息vocabulary_
vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.5)
'''
关于参数，你只需要了解这么几个就可以了： 
stop_words: 
传入停用词，以后我们获得vocabulary_的时候，就会根据文本信息去掉停用词得到 
vocabulary: 
之前说过，不再解释。 
sublinear_tf: 
计算tf值采用亚线性策略。比如，我们以前算tf是词频，现在用1+log(tf)来充当词频。 
smooth_idf: 
计算idf的时候log(分子/分母)分母有可能是0，smooth_idf会采用log(分子/(1+分母))的方式解决。默认已经开启，无需关心。 
norm: 
归一化，我们计算TF-IDF的时候，是用TF*IDF，TF可以是归一化的，也可以是没有归一化的，一般都是采用归一化的方法，默认开启. 
max_df: 
有些词，他们的文档频率太高了（一个词如果每篇文档都出现，那还有必要用它来区分文本类别吗？当然不用了呀），所以，我们可以 
设定一个阈值，比如float类型0.5（取值范围[0.0,1.0]）,表示这个词如果在整个数据集中超过50%的文本都出现了，那么我们也把它列 
为临时停用词。当然你也可以设定为int型，例如max_df=10,表示这个词如果在整个数据集中超过10的文本都出现了，那么我们也把它列 
为临时停用词。 
min_df: 
与max_df相反，虽然文档频率越低，似乎越能区分文本，可是如果太低，例如10000篇文本中只有1篇文本出现过这个词，仅仅因为这1篇 
文本，就增加了词向量空间的维度，太不划算。 
当然，max_df和min_df在给定vocabulary参数时，就失效了。 
'''

# 此时tdm里面存储的就是if-idf权值矩阵
tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
tfidfspace.vocabulary = vectorizer.vocabulary_

path = os.path.abspath(__file__)
path1 = os.path.split(path)
wordbag_path = path1[0] + '\\train_set.dat'
classification_path = path1[0] + '\\Classification_NB.dat'
Analysis.FormatData.writebunchobj(wordbag_path, tfidfspace)
# 训练分类器：输入词袋向量和分类标签，alpha:0.001 alpha越小，迭代次数越多，精度越高
clf = MultinomialNB(alpha=0.001).fit(tfidfspace.tdm, tfidfspace.label)
Analysis.FormatData.writebunchobj(classification_path, clf)
