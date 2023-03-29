import pandas as pd
import numpy as np

K = 1
get_list = []
data = np.asarray(pd.read_csv('../data/all_item_source.csv'))
print(data[0])
for i in range(len(data)):
    print(data[i].argsort()[-K:][::-1])
    get_list.append(data[i].argsort()[-K:][::-1])
get_list = np.asarray(get_list)
print(get_list)

test_data = np.zeros(100, dtype=np.int32)
ua_test = np.asarray(pd.read_csv('../data/ua_test.csv'))

for i in range(100):
    test_data[i] = ua_test[i][1]
print(test_data)

test_data = test_data.reshape((10, 10))
print(test_data)

pre_k = np.zeros(10)

for i in range(10):
    print(len(np.intersect1d(test_data[i], get_list[i])) / K)
    pre_k[i] = len(np.intersect1d(test_data[i], get_list[i])) / K

# print(np.intersect1d(test_data[0], get_list[0]))
print(pre_k.mean())


