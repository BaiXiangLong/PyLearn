# encoding=utf-8

"""
智能推荐算法主函数
"""

import sys
import SQLconfig
reload(sys)
sys.setdefaultencoding("utf-8")

# 运行一次可以更新一个用户的推荐列表
# 导入目标用户的阅读记录
ReadHistory = SQLconfig.sql0.select('user_history', ['article_id', 'like_value', 'readtime'],
                                    'user_id = %d' % (SQLconfig.user_id))
# 用户阅读记录按阅读实现正序排序
ReadHistory = sorted(ReadHistory, key=lambda x: (x[1]), reverse=True)
"""
while len(ReadHistory) > 10:
    ReadHistory.remove(ReadHistory[-1])
"""

# 基于相似用户推荐，选择相似用户时尽量选择浏览历史相似的用户，在这里就是看过相同文章多的用户，这是第一次过滤，可以大大降低矩阵的维度
PossibleUser = [] # 和目标用户有相同阅读记录的用户
SimsilarUser = [] # 和目标用户有相同阅读记录超过阈值条数的用户
for i in ReadHistory:
    PossibleUser.extend(SQLconfig.sql0.select('user_history', ['user_id'], 'article_id = %d' % (i[0])))
for i in PossibleUser:
    if i not in SimsilarUser and PossibleUser.count(i) > 2 and i != SQLconfig.user_id:
        SimsilarUser.append(i)
SimsilarUser.append(SQLconfig.user_id)  # 在相似阅读记录中加入目标用户

# 构建用户阅读记录字典
UserHistoryDict = {}
# 先添加目标用户的阅读记录
for i in SimsilarUser:
    ReadHistoryTemp = SQLconfig.sql0.select('user_history', ['article_id', 'like_value'],
                                        'user_id = %d' % (i))
    TempDic = {}
    for ii in ReadHistoryTemp:
        TempDic.update({ii[0]: ii[1]})
    UserHistoryDict.update({i: TempDic})

