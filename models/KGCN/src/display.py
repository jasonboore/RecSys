import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties



def display_4D(x, y_1, y_2, y_3, title):
    font_set = FontProperties(fname=r"/System/Library/Fonts/STHeiti Medium.ttc")
    plt.title(title, fontproperties=font_set)
    new_x = [1, 2, 3, 4, 5, 6, 7]
    plt.plot(new_x, y_3, color='#CC00FF', linestyle=':', marker='s', label='KGCN')
    plt.plot(new_x, y_2, color='#0033FF', linestyle='--', marker='x', label='RippleNet')
    plt.plot(new_x, y_1, color='#009933', linestyle='-.', marker='2', label='MKR')
    # plt.plot(new_x, y_4, color='#FF6600', linestyle='-',marker='^', label='KGNN-LS')
    plt.xticks(new_x, x)
    plt.legend()
    plt.xlabel('推荐个数', fontproperties=font_set)
    plt.ylabel(title, fontproperties=font_set)
    # plt.savefig('res_20M/{}.pdf'.format(title))
    plt.show()


def insert_factor(y):
    mids = []
    for i in range(len(y) - 1):
        mid = (y[i] + y[i + 1]) / 2
        mids.append(mid)
    res = []
    for i in range(len(y) - 1):
        res.append(y[i])
        res.append(mids[i])
    res.append(y[len(y) - 1])
    return res


def cal_f1(recall, precision):
    f1 = [2 / (1 / precision[i] + 1 / recall[i]) for i in range(len(recall))]
    return f1


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2. - 0.2, 1.03 * height, '%s' % int(height))


if __name__ == '__main__':
    # start_1899, end_1899, step_1899 = 50, 1000, 50
    # x_1899 = np.arange(start_1899, end_1899, step_1899)
    # print(x_1899)
    # y = \
    #     [2014.1, 1730.1, 1695.8, 1689.7, 1702.9, 1700.8, 1685.7, 1660.8, 1665.5, 1675.2]
    # y = insert_factor(y)

    # m = []
    # for i in range(len(y)):
    #      if i%2==0:
    #          m.append(y[i])
    # print(y)
    # x = np.arange(50,1001,50)
    # y = [3308.9, 2014.1, 1872.1, 1730.1, 1712.95, 1695.8, 1692.75, 1689.7, 1696.3, 1702.9, 1701.85, 1700.8, 1693.25,
    #      1685.7, 1673.25, 1660.8, 1663.15, 1665.5, 1670.35, 1675.2]
    #
    # display(x, y)
    k_list = [1, 2, 5, 10, 20, 50, 100]
    MKR = \
        cal_f1(
            [0.1000, 0.1000, 0.1020, 0.0820, 0.0755, 0.0574, 0.0424]
            , [0.0064, 0.0154, 0.0501, 0.0908, 0.1471, 0.2649, 0.3777])
    RippleNet = \
        cal_f1([0.1300, 0.1050, 0.1010, 0.1070, 0.0825, 0.0602, 0.0464],
               [0.0070, 0.0340, 0.0500, 0.09550, 0.1520, 0.3090, 0.4280])
    KGCN_Bi = \
        cal_f1([0.1500, 0.1100, 0.1200, 0.1110, 0.0990, 0.0742, 0.0569],
               [0.0081, 0.0350, 0.0648, 0.1077, 0.1897, 0.3216, 0.4667])

    print(MKR)
    print(RippleNet)
    print(KGCN_Bi)

    # display_4D(k_list, MKR, RippleNet, KGCN_Bi, "F1")





