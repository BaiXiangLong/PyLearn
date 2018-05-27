关键词提取算法使用说明：
依赖的库
sklearn
numpy
jieba

文件介绍：
  1、gethotword.py ：统计热词
  2、matchWordAndContent.py ：将文章与热词联系在一起
  3、FormatData.py ：一些工具函数
  4、do_mysql.py ：一些数据库操作

数据库前提：
  1、为每一个数据表格建立文件名+_tag的表，字段见nwpu_news_tag一样，也可以将上一个目录下三个表直接放进去。_tag表记录了文章id与热词id的匹配关系，还有对应热词在这篇的权重和在全部文章中的权重。
  2、nwpu_news需要添加一个id字段
  3、新建或导入hot_words表

使用：
  先运行文件1，统计所有表的热词，并存在hot_words表中
  再运行文件2，统计每一篇文章的热词，并存在相应的_tag表中