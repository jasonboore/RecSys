import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

loss = np.asarray(pd.read_csv('../data/contrast/run-May01_19-29-30_LAPTOP-NE4I3J65-RecommendNetwork-tag-reward_100.csv'))
loss_x = loss[:, 1][:-110]
loss_y = loss[:, 2][:-110]
print(loss_x)
print(loss_y)

plt.figure()
plt.plot(loss_x, loss_y, 'r--', label='loss')
plt.show()

pre_x = ['SPRank', 'GRank', 'PMCR', 'DRL']
pre_y = [0.2958, 0.3522, 0.3652, 0.3722]

plt.figure()
plt.plot(pre_x, pre_y, 'ro-', label='pre@5')
plt.show()

