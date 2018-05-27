#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@脚本名称：TransPath.py
@脚本作用：搬运数据
@备注：只搬运了'title'、'content'、'data_publish'、'love_count'=0、'click_count'=0、'img'字段
"""
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import os
path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(path)
print path

import Analysis.SQLconfig

def runTransPath():
    continue_flag = True
    # 取数据操作的改变，直接搜索语句就好了
    while continue_flag:
        info = Analysis.SQLconfig.sql0.select('article', ['title', 'author', 'article.desc', 'content', 'image_path',
                                                          'posttime', 'category', 'id'], 'process_state = 0', 1000,
                                              None)
        if len(info) == 0:
            print u"已完成全部转移"
            return
        if len(info) < 1000:
            continue_flag = False
        print u"已读取%d条数据" % len(info)
        for ii in info:
            if ii[2] is None:
                ii[2] = " "
            elif len(ii[2]) > 299:
                ii[2] = ii[2][0:299]
            dicDate = {'title': ii[0], 'dcweb_article.desc': ii[2], 'content': ii[3], 'love_count': 0, 'click_count': 0,
                       'date_publish': ii[5], 'category_id': ii[6],
                       'publisher_id': Analysis.SQLconfig.dicPublisher.get(ii[1]),
                       'img': ii[4], 'tag_mark': 0}
            try:
                Analysis.SQLconfig.sql1.add('dcweb_article', dicDate)
                Analysis.SQLconfig.sql0.update('article', {'process_state': 1}, 'id = %d' % ii[7])
            except:
                print u"转移路径出错"
            print u"id = %d文章已转移" % ii[7]
