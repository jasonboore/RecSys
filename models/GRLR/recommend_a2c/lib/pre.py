import pandas as pd
import numpy as np

ITEM_NUMBER = 1682          #项目的个数
USER_NUMBER = 943          #用户数
K = 10

sources_number = np.asarray(pd.read_csv('../data/contrast/all_item_source.csv'))
print(sources_number)

'''处理标签'''
ua_test = np.asarray(pd.read_csv('../data/net_data/ua_test.csv'))
test_data = np.zeros(1000, dtype=np.int32)

for i in range(1000):
    test_data[i] = ua_test[i][1]
print(test_data)
test_data = test_data.reshape((100, 10))
print(test_data)

'''找到top-k的索引'''
top_k = np.zeros((100, K), dtype=int)
print(top_k)

pre_k = np.zeros(100)


for i in range(len(sources_number)):
    # print(sources_number[i].argsort()[-10:][::-1]+USER_NUMBER)

    # for j in range(len(top_k[i])):

    top_k[i] = sources_number[i].argsort()[-K:][::-1]+USER_NUMBER
    pre_k[i] = len(np.intersect1d(top_k[i], test_data[i])) / K
print(pre_k)
print(pre_k.mean())






