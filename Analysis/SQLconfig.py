#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@脚本名称：SQLconfig.py
@脚本作用：所有数据库配置配置
"""
from MySQLTool import MySQLTools
import os

# sql0是dcspider数据库
sql0 = MySQLTools(
    host='111.230.181.121',
    port=3306,
    user='root',
    passwd='ykl123',
    db='dcspider',
    charset='utf8'
)

# sql1是dcserver数据库
sql1 = MySQLTools(
    host='111.230.181.121',
    port=3306,
    user='root',
    passwd='ykl123',
    db='dcserver',
    charset='utf8'
)

# 本地数据库
LocalSql = MySQLTools(
    host='localhost',
    port=3306,
    user='root',
    passwd='',
    db='test',
    charset='utf8'
)

# 类别字典
dicClassToCategory = {}
i = -1
while True:
    i += 1
    info = sql1.select('dcweb_category', ['id', 'name'], None, 1, i)
    if len(info) == 0:
        break
    dicClassToCategory.update({'%s' % info[1]: '%d' % info[0]})

# 来源对应字典
dicPublisher = {}
i = -1
while True:
    i += 1
    info = sql1.select('dcweb_publisher', ['id', 'name'], None, 1, i)
    if len(info) == 0:
        break
    dicPublisher.update({'%s' % info[1]: '%d' % info[0]})
