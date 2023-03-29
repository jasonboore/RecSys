import xlrd
import matplotlib.pyplot as plt

file = '../result/topk.xlsx'
xl = xlrd.open_workbook(file)
table = xl.sheets()[0]
x=[2,5,10,20,50,100]

# y1=table.row_values(16)
# y2=table.row_values(17)
# y3=table.row_values(18)
# y4=table.row_values(19)
# y5=table.row_values(20)
# plt.figure(1)
# plt.plot(x,y1,marker='*')
# plt.plot(x,y2,marker='o')
# plt.plot(x,y3,marker='s')
# plt.plot(x,y4,marker='+')
# plt.plot(x,y5,marker='D')
# plt.legend(['PER','CKE','MKR','RippleNet','MKR-Bine'])
# plt.title('TopK-Precision')
# plt.show()
#
y6=table.row_values(21)
y7=table.row_values(22)
y8=table.row_values(23)
y9=table.row_values(24)
y10=table.row_values(25)
plt.figure(2)
plt.plot(x,y6,marker='*')
plt.plot(x,y7,marker='h')
plt.plot(x,y8,marker='x')
plt.plot(x,y9,marker='o')
plt.plot(x,y10,marker='p')
plt.legend(['PER','CKE','MKR','RippleNet','MKR-Bine'])
plt.title('TopK-Recall')
plt.show()


# y1=table.row_values(0)
# y2=table.row_values(1)
# y3=table.row_values(2)
# y4=table.row_values(3)
# y5=table.row_values(4)
# plt.figure(1)
# plt.plot(x,y1,marker='*')
# plt.plot(x,y2,marker='o')
# plt.plot(x,y3,marker='s')
# plt.plot(x,y4,marker='+')
# plt.plot(x,y5,marker='D')
# plt.legend(['PER','CKE','MKR','RippleNet','MKR-Bine'])
# plt.ylim(0.02,0.12)
# plt.title('TopK-Precision')
# plt.show()
# #
y6=table.row_values(5)
y7=table.row_values(6)
y8=table.row_values(7)
y9=table.row_values(8)
y10=table.row_values(9)
plt.figure(2)
plt.plot(x,y6,marker='*')
plt.plot(x,y7,marker='h')
plt.plot(x,y8,marker='x')
plt.plot(x,y9,marker='H')
plt.plot(x,y10,marker='D')
plt.legend(['PER','CKE','MKR','RippleNet','MKR-Bine'])
plt.title('TopK-Recall')
plt.show()

##from matplotlib import pyplot as pl
##import numpy as np
##from scipy import interpolate
##
##x = np.linspace(-100,100,5)
##y = -1/3*x**3 + 9*x + 30
##pl.figure(figsize = (8, 4))
##pl.plot(x, y, color="blue", linewidth = 1.5)
##pl.show()
