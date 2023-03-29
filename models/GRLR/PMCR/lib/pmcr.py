import pandas as pd
import numpy as np

# 超参数
BASE_LEN = 90570
TEST_LEN = 9430
ITEM_LEN = 1682
USER_NUMBER = 943
ITEM_NUMBER = 1682
e = 1

# # 读取ua_base数据
# f = open('../data/ua.base')
# ua_base = []
# for i in range(BASE_LEN):
#     h = f.readline().split('\t')
#     h = list(map(int, h))
#     ua_base.append(h[0:3])
# f.close()
#
# # 读取ua_test数据
# f = open('../data/ua.test')
# ua_test = []
# for i in range(TEST_LEN):
#     h = f.readline().split('\t')
#     h = list(map(int, h))
#     ua_test.append(h[0:3])
# f.close()

ua_base = np.asarray(pd.read_csv('../data/ua_base.csv')).tolist()
ua_test = np.asarray(pd.read_csv('../data/ua_test.csv')).tolist()
# 生成每个用户的项目表，每个用户选择了哪些项目
# 创建用户数量大小的空列表，以便加入，顺便加入每个用户对所选项目的评分，
# 用元组表示，(所选项目， 打分)
I_U_list = []
for i in range(USER_NUMBER):
    I_U_list.append([])
for i in range(BASE_LEN):
    I_U_list[ua_base[i][0]].append((ua_base[i][1], ua_base[i][2]))
print(I_U_list[0])

select_list = []
for i in range(USER_NUMBER):
    select_list.append([])
for i in range(USER_NUMBER):
    for j in range(len(I_U_list[i])):
        select_list[i].append(I_U_list[i][j][0])
print(select_list[0])

# 生成每个用户的偏好集，偏好用元组表示，(u, i, j) 表示对于用户u，比起j,更喜欢i
I_U_P_list = []
for i in range(USER_NUMBER):
    I_U_P_list.append([])
i = 0
for I_U in I_U_list:
    for j in range(len(I_U)):
        k = j + 1
        while k < len(I_U):
            m = I_U[j][1]
            n = I_U[k][1]
            if m > n:
                I_U_P_list[i].append((I_U[j][0], I_U[k][0]))
            if m < n:
                I_U_P_list[i].append((I_U[k][0], I_U[j][0]))
            k += 1
    print(i)
    i += 1
print(I_U_P_list[0])

# 创建user列表，待用
user_list = range(USER_NUMBER)
# 对所有用户遍历，为他们推荐，tar_user中存的是目标用户的所选项目和评分
all_item_source = []
tar_user_index = 0
for tar_user in select_list:
    # if tar_user_index == 0:
    #     tar_user_index += 1
    #     continue

    # 每个用户分配的给已选项目的资源不同，需要重新分配
    item_source = []
    for i in range(ITEM_LEN):
        item_source.append(0.0)

    for index in tar_user:
        item_source[index] = e

    # 对所有的其他用户求解
    for other_user_index in user_list:
        if other_user_index != tar_user_index:
            # 计算重叠偏好集
            OPS = []
            # 准备其他用户的已选定的项目，目标用户已选定的项目，其他用户的偏好集
            tar_select = select_list[tar_user_index]
            other_select = select_list[other_user_index]
            other_prefer = I_U_P_list[other_user_index]
            # 取已选项目的交集
            common_select = list(set(tar_select).intersection(set(other_select)))
            # 将目标用户和其他用户选择的项目合起来
            all_select = list(set(tar_select).union(set(other_select)))

            for i in common_select:
                for j in all_select:
                    pre = (i, j)
                    if pre in other_prefer:
                        OPS.append(pre)

            # 计算互补偏好集
            # CPS = []
            # for prefer in other_prefer:
            #     if prefer not in OPS:
            #         CPS.append(prefer)
            CPS = list(set(other_prefer) - set(OPS))
            # 如果完全重叠，则进入下一个用户
            if len(CPS) == 0:
                continue

            # 创建第i个项目的互补偏好集的字典，i作键，偏好i项目的元组作值
            CPS_i = {}
            for i in range(len(CPS)):
                index = CPS[i][0]
                if index in CPS_i.keys():
                    CPS_i[index].append(CPS[i])
                else:
                    CPS_i[index] = []
                    CPS_i[index].append(CPS[i])


            # 用公式1计算该其他用户收到的资源
            ru_ = (len(OPS)*e)/(ITEM_LEN-1)

            # 计算其他用户再分配给每个项目i的资源量, 需要遍历第i个项目的互补偏好集
            for key, value in CPS_i.items():
                riu_ = ru_ * len(value) / len(CPS)
                item_source[key] += riu_
        print("结束目标用户{}，其他用户{}".format(tar_user_index, other_user_index))

    all_item_source.append(item_source)
    print("结束{}目标用户".format(tar_user_index))

    tar_user_index += 1
    if tar_user_index == 10:
        break

for i in range(len(all_item_source)):
    for j in range(len(select_list[i])):
        all_item_source[i][select_list[i][j]] = 0.0


print(all_item_source)
all_item_source = np.asarray(all_item_source)
pd.DataFrame(all_item_source).to_csv('../data/all_item_source.csv', header=True, index=False)
















