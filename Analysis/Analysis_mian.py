#! /usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import SQLconfig
path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)
sys.path.append('/urs/local/bin/python')
print path

import logging
import AutoDesc.auto_abstract
import Classify.Classification
import TagCloud.gethotword
import TagCloud.matchWordAndContent
import TransPath.TransPath



def RunAnalysis():
    # 修改当前工作路径
    path_now = os.path.dirname(os.path.abspath(__file__))
    path_before = os.getcwd()
    print 'path_now', path_now
    print 'path_before', path_before
    os.chdir(path_now)

    print u"当前目录", os.getcwd()


    logging.basicConfig(filename='/Analysislog.log', level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logging.debug(u"文本处理开始...")

    print u"------文本处理开始：------\n"

    try:
        logging.debug(u"分类开始...")
        print u"------分类开始...------\n"
        Classify.Classification.RunClassify()
        print u"\n------分类成功执行!------\n"
        logging.debug(u"分类成功执行！")

    except Exception, e:
        log = u"分类发生错误!ERROR(%s):%s" % (e.args[0], e.message)
        print log
        logging.debug(log)

    try:
        logging.debug(u"摘要开始...")
        print u"------摘要开始...------\n"
        AutoDesc.auto_abstract.RunAbstract()
        print u"\n------摘要成功执行!------\n"
        logging.debug(u"摘要成功执行！")

    except Exception, e:
        log = u"摘要发生错误!ERROR(%s):%s" % (e.args[0], e.message)
        print log
        logging.debug(log)

    try:
        print u"------文章转移开始------\n"
        logging.debug(u'文章转移开始...')
        TransPath.TransPath.runTransPath()
        logging.debug(u"文章转移成功执行!")
        print u"\n------文章转移成功执行!------\n"

    except Exception, e:
        log = u"文章转移路径出错!ERROR(%s):%s" % (e.args[0], e.message)
        print log
        logging.debug(log)

    try:
        logging.debug(u'关键字统计开始...')
        print u"------关键词统计开始...------\n"
        TagCloud.gethotword.rungethotword()
        try:
            TagCloud.matchWordAndContent.runmatchWordAndContent()
            print u"\n------关键字统计结束!------\n"
            logging.debug(u"关键词统计成功执行!")
            print u"------文本处理成功执行!------\n"
            logging.debug(u"文本处理成功执行!")
        except Exception, e:
            log = u"热词与文章匹配发生错误!ERROR(%s):%s" % (e.args[0], e.message)
            print log
            logging.debug(log)
    except Exception, e:
        log = u"统计热词发生错误!ERROR(%s):%s" % (e.args[0], e.message)
        print log
        logging.debug(log)
        logging.debug(u"文本处理有错误!")
        
    os.chdir(path_before)

if __name__ == '__main__':
    # 修改当前工作路径
    path_now = os.path.dirname(os.path.abspath(__file__))
    path_before = os.getcwd()
    print 'path_now', path_now
    print 'path_before', path_before
    os.chdir(path_now)
    RunAnalysis()
    os.chdir(path_before)
