#!/usr/local/bin/python
# -*- coding: UTF-8 -*-

import MySQLdb


class MySQLTools:
    """数据库基本操作库"""
    def __init__(self, *args, **kwargs):
        self._conn = MySQLdb.connect(*args, **kwargs)
        self._cursor = self._conn.cursor()

    def _getCount(self, tablename):
        """
        @summary: 得到数据表的数据条数
        :return: 当前数据表的数据条数
        """
        sql = 'select * from %s' % tablename
        a = self._cursor.execute(sql)
        return a

    def exeCute(self, sql = ''):
        """
        @summary: 针对读操作返回结果集
        :param sql: 执行的数据库语句 
        :return: 
        """
        try:
            self._cursor.execute(sql)
            records = self._cursor.fetchall()
            return records
        except MySQLdb.Error, e:
            error = u'MySQL execute failed! ERROR(%s): %s' %(e.args[0], e.args[1])
            print error

    def _QuMark(self, text):
        """
        @summary: 规范化双引号
        :param text: 文本文档
        :return: 规范化后的文本
        """
        i = 0
        while i < len(text):
            if text[i] == '"':
                text = text[0: i] + '\\' + text[i: len(text)]
                i += 1
            i += 1
        return text

    def exeCuteCommit(self, sql='', arg=None):
        """
        @summary: 针对更新，删除，事物等的异常处理
        :param sql: 执行的SQL语句
        :param arg: 参数
        :return: 无
        """
        try:
            if arg is None:
                self._cursor.execute(sql)
            else:
                self._cursor.execute(sql, arg)
            self._conn.commit()
        except MySQLdb.Error, e:
            self._conn.rollback()
            error = u'MySQL execute failed! ERROR(%s):%s' % (e.args[0], e.args[1])
            print error

    def select(self, tablename, param=None, con=None, num=None, offset=None):
        """
        @summary: 从数据库中得到多条数据
        :rtype: object
        :param tablename: 数据库的名称 
        :param param: 得到数据的列
        :param con: 得到数据库的条件
        :param num: 得到数据的条数
        :param offset: 偏移量
        :return: 数据列表
        @举例 q = m1._select('shool_news', ['title'], None, 2, 1)
        @从shool_news数据表中取出title,没有条件,取出两条数据,偏移量为1
        """
        sql = 'SELECT '
        if param is None:
            sql += '* '
        else:
            for strParam in param:
                sql += '%s,' % strParam
        sql = sql.strip(',')
        sql += ' FROM %s' % tablename
        if con is not None:
            sql += ' WHERE %s' % con
        if num is not None:
            sql += ' LIMIT %d' % num
        if offset is not None:
            sql += ' OFFSET %d' % offset
        data = self.exeCute(sql)
        if data is None:
            return []
        re = []
        if num is 1:
            for ii in data:
                for iii in ii:
                    # re.append(iii.decode('utf-8'))
                    re.append(iii)
        else:
            for ii in data:
                temp = []
                for iii in ii:
                    # temp.append(iii.decode('utf-8'))
                    temp.append(iii)
                if len(temp) < 2:
                    re.extend(temp)
                else:
                    re.append(temp)
        return re

    def add(self, tablename, dicDate):
        """
        @summary: 插入数据
        :param tablename: 数据表格的名称 
        :param dicDate: 参数字典
        :return: 无
        """
        sql = 'INSERT INTO %s (' % tablename
        for ii in dicDate.keys():
            sql += '%s,' % ii
        sql = sql.strip(',')
        sql += ') VALUES ('
        for ii in dicDate.keys():
            if type(dicDate.get(ii)) == str or type(dicDate.get(ii)) == unicode:
                sql += '"%s",' % self._QuMark(dicDate.get(ii))
            elif type(dicDate.get(ii)) == int:
                sql += '%d,' % dicDate.get(ii)
            else:
                sql += '"%s",' % dicDate.get(ii)
        sql = sql.strip(',')
        sql += ')'
        self.exeCuteCommit(sql)

    def update(self, tablename, dataDic, con=None):
        """
        @summary: 修改数据库的值
        :param tablename: string,数据表的名字
        :param datalist: dic更新值的字典
        :param con: boolean条件
        :return: 
        @举例：dic = {'Class': 12, 'abstract': 'bxl'}
        m1._update('shool_news', dic, 'title = "白向龙测试数据"')
        """
        sql = 'UPDATE %s SET ' % tablename
        for ii in dataDic.keys():
            if type(dataDic.get(ii)) == str or type(dataDic.get(ii)) == unicode:
                sql += '%s = ' % ii + '"%s",' % self._QuMark(dataDic.get(ii))
            elif type(dataDic.get(ii)) == int:
                sql += '%s = ' % ii + '%d,' % dataDic.get(ii)
            else:
                sql += '%s = ' % ii + '"%s",' % dataDic.get(ii)
        sql = sql.strip(',')
        if con is not None:
            sql += ' WHERE %s' % con
        self.exeCuteCommit(sql)


    def delete(self, tablename, con = None):
        """
        删除数据
        :param tablename: 数据库的名字
        :param con: 删除条件
        :return: 无
        """
        sql = 'DELETE FROM %s' % (tablename)
        if con is not None:
            sql += ' WHERE %s' % (con)
        self.exeCuteCommit(sql)

    def close(self):
        """
        关闭数据库
        :return: 无
        """
        self._conn.close()
