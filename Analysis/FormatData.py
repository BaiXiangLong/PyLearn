# coding=utf-8
'''
脚本名称 ： FormatData.py
功 能 ： 一些函数
'''

import sys
import MySQLdb
import FormatData
import jieba
import cPickle as pickle
from sklearn.datasets.base import Bunch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB  # 导入多项式贝叶斯算法

# ###################基础常量###################################
wordbag_path = "./train_word_bag_my/train_set_my.dat"
stopword_path = "./train_word_bag_my/hlt_stop_words.txt"
classification_path = "./train_word_bag_my/classification_NB.dat"
###############################################################

# 函数名称：CutText
# 功能：分词
# 输入：字符串
# 输出：分词后的字符串
def TextCut(str1):
    # 第一步：分词
    content = str1
    content = content.replace("\r\n", "")  # 删除换行
    content = content.replace(" ", "")  # 删除空行、多余的空格
    content_seg = jieba.cut(content)  # 为文件内容分词
    return " ".join(content_seg)

# 函数名称：_readfile
# 函数功能：读取文件
# 输 入 ：文件地址
# 输出 ： 文件的字符串
def readfile(path):
    with open(path, "rb") as fp:
        content = fp.read()
    return content

# 函数名称：_readbunchobj
# 函数功能：读取bunch对象
# 输入数据：文件地址
# 输出数据：bounch对象
def readbunchobj(path):
    with open(path, "rb") as file_obj:
        bunch = pickle.load(file_obj)
    return bunch

# 函数名称：_writebunchobj
# 函数功能：存储bunch对象
# 输入数据：参数1：文件地址，参数2：bunch对象
# 输出数据：无
def writebunchobj(path, bunchobj):
    with open(path, "wb") as file_obj:
        pickle.dump(bunchobj, file_obj)
