import pandas as pd
import numpy as np

# 超参数
BASE_LEN = 90570
TEST_LEN = 9430
ITEM_LEN = 1682
USER_NUMBER = 943
ITEM_NUMBER = 1682
e = 1

# 读取ua_base数据
f = open('../data/ua.base')
ua_base = []
for i in range(BASE_LEN):
    h = f.readline().split('\t')
    h = list(map(int, h))
    ua_base.append(h[0:3])
f.close()
ua_base = np.asarray(ua_base)-1
for i in range(len(ua_base)):
    ua_base[i][2] += 1
# pd.DataFrame(ua_base).to_csv('../data/ua_base.csv', header=True, index=False)


# 读取ua_test数据
f = open('../data/ua.test')
ua_test = []
for i in range(TEST_LEN):
    h = f.readline().split('\t')
    h = list(map(int, h))
    ua_test.append(h[0:3])
f.close()
ua_test = np.asarray(ua_test)-1
for i in range(len(ua_test)):
    ua_test[i][2] += 1
print(ua_test)
# pd.DataFrame(ua_test).to_csv('../data/ua_test.csv', header=True, index=False)