import numpy as np
import pandas as pd

# 对age进行分段，映射成7组
def age_map(age):
    if age >= 1 and age <= 7: return 1
    if age >= 8 and age <= 16: return 2
    if age >= 17 and age <= 29: return 3
    if age >= 30 and age <= 39: return 4
    if age >= 40 and age <= 49: return 5
    if age >= 50 and age <= 59: return 6
    if age >= 60: return 7

def year_map(year):
    if year >= 0 and year <= 1899: return 1
    if year >= 1900 and year <= 1909: return 2
    if year >= 1910 and year <= 1919: return 3
    if year >= 1920 and year <= 1929: return 4
    if year >= 1930 and year <= 1939: return 5
    if year >= 1940 and year <= 1949: return 6
    if year >= 1950 and year <= 1959: return 7
    if year >= 1960 and year <= 1969: return 8
    if year >= 1970 and year <= 1979: return 9
    if year >= 1980 and year <= 1989: return 10
    if year >= 1990 and year <= 1999: return 11
    if year >= 2000: return 12

gender_map = {'M':1, 'F':0}
occupations_dict = {'technician': 1,
     'other': 0,
     'writer': 2,
     'executive': 3,
     'administrator': 4,
     'student': 5,
     'lawyer': 6,
     'educator': 7,
     'scientist': 8,
     'entertainment': 9,
     'programmer': 10,
     'librarian': 11,
     'homemaker': 12,
     'artist': 13,
     'engineer': 14,
     'marketing': 15,
     'none': 16,
     'healthcare': 17,
     'retired': 18,
     'salesman': 19,
     'doctor': 20}

u_item = np.asarray(pd.read_csv('../data/net_data/u_item.csv'))
u_user = np.asarray(pd.read_csv('../data/net_data/u_user.csv'))

# print(u_item)
# print(u_user)

for i in range(len(u_user)):
    # 为年龄编码
    u_user[i][1] = age_map(u_user[i][1])

    # 为性别编码
    u_user[i][2] = gender_map[u_user[i][2]]

    # 为职业编码
    u_user[i][3] = occupations_dict[u_user[i][3]]
    # u_user[i][4] = int(u_user[i][4][:3])

print(u_user)

u_item = np.nan_to_num(u_item)
u_item[266] = np.nan_to_num(u_item[266])
# time_list = []
print(len(u_item))
item = np.zeros((len(u_item), 20))
for i in range(len(item)):
    if i == 266:
        item[i][0] = year_map(0)
    else:
        item[i][0] = year_map(int(u_item[i][2][7:]))
print(item)

item[:, 1:] = u_item[:, 5:]
print(item[0])


data_set = np.zeros((2625, 27))

# 将user的信息赋值
data_set[0:943, 0:3] = u_user[0:943, 1:4]

print(data_set[0])
data_set[943: , 6:-1] = item[:, :]

# 将item的信息赋值
# data_set[]
print(data_set[943])

print(data_set)

pd.DataFrame(data_set).to_csv('../data/net_data/data_set.csv', header=True, index=False)
