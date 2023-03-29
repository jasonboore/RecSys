import xlrd
import matplotlib.pyplot as plt
import numpy as np

file = '../result/result.xlsx'
xl = xlrd.open_workbook(file)
table = xl.sheets()[0]
x=table.col_values(0)[:6]
# x=table.col_values(0)
y1=table.col_values(1)[:6]
y2=table.col_values(3)[:6]

bar_width = 0.2
index_mkr = np.arange(len(x))
index_bine = index_mkr + bar_width

plt.bar(index_mkr, height=y1, width=bar_width,  label='MKR')
plt.bar(index_bine, height=y2, width=bar_width, label='MKR-Bine')
plt.xticks(index_mkr + bar_width/2, x)
plt.ylim(0,0.9)
plt.legend(['MKR','MKR-Bine'])
plt.ylabel('Precision')
plt.title('EmbeddingSize-Precision')
plt.show()

y3=table.col_values(2)[:6]
y4=table.col_values(4)[:6]
bar_width = 0.2
index_mkr = np.arange(len(x))
index_bine = index_mkr + bar_width

plt.bar(index_mkr, height=y3, width=bar_width,  label='MKR')
plt.bar(index_bine, height=y4, width=bar_width, label='MKR-Bine')
plt.xticks(index_mkr + bar_width/2, x)
plt.ylim(0,0.9)
plt.legend(['MKR','MKR-Bine'])
plt.ylabel('Recall')
plt.title('EmbeddingSize-Recall')
plt.show()