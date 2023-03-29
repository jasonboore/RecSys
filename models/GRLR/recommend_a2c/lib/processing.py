import pandas as pd
import numpy as np

USER_NUMBER = 943
ITEM_NUMBER = 1682
BASE_LEN = 90570

# ua_base = np.asarray(pd.read_csv('../data/ua_base.csv'))
# ua_test = np.asarray(pd.read_csv('../data/ua_test.csv'))
# for i in range(len(ua_base)):
#     ua_base[i][1] += USER_NUMBER
# for i in range(len(ua_test)):
#     ua_test[i][1] += USER_NUMBER
#
# pd.DataFrame(ua_base).to_csv('../data/net_data/ua_base.csv', header=True, index=False)
# pd.DataFrame(ua_test).to_csv('../data/net_data/ua_test.csv', header=True, index=False)
# 读取ua_base数据
# f = open('../data/u.item', encoding='ISO-8859-1')
# u_item = []
# for i in range(ITEM_NUMBER):
#     h = f.readline()[0:-1].split('|')
#     # h = list(map(int, h))
#     u_item.append(h)
# f.close()
# print(u_item)
# u_item = np.asarray(u_item)
# pd.DataFrame(u_item).to_csv('../data/net_data/u_item.csv', header=True, index=False)

# u_item = np.asarray(pd.read_csv('../data/net_data/u_item.csv'))
# for i in range(len(u_item)):
#     u_item[i][0] -= 1
# pd.DataFrame(u_item).to_csv('../data/net_data/u_item.csv',header=True, index=False)

# # 读取ua_base数据
# f = open('../data/u.user', encoding='ISO-8859-1')
# u_item = []
# for i in range(USER_NUMBER):
#     h = f.readline()[0:-1].split('|')
#     # h = list(map(int, h))
#     u_item.append(h)
# f.close()
# print(u_item)
# u_item = np.asarray(u_item)
# pd.DataFrame(u_item).to_csv('../data/net_data/u_user.csv', header=True, index=False)
#
#
u_user = np.asarray(pd.read_csv('../data/net_data/u_user.csv'))
for i in range(len(u_user)):
    u_user[i][0] -= 1

pd.DataFrame(u_user).to_csv('../data/net_data/u_user.csv', header=True, index=False)

