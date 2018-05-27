#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import sys
import os

path1 = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path1)
sys.path.append('/usr/local/bin/python')
print path1

import logging
import AutoRecommend.autocomm_CT




def RunRecommend():
    logging.basicConfig(filename='Recommend.log', level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print u"------用户个性化推荐开始...------\n"
    logging.debug(u"用户个性化推荐开始...")
    try:
        AutoRecommend.autocomm_CT.run_recommend()
        print u"\n------用户个性化推荐成功运行------\n"
        logging.debug(u"用户个性化推荐运行成功！")
    except Exception, e:
        log = u"\n用户个性化推荐错误！ERROR(%s):%s\n" % (e.args[0], e.message)
        print log
        logging.debug(log)

if __name__ == "__main__":
    RunRecommend()
