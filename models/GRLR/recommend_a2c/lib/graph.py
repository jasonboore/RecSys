import numpy as np
import pandas as pd

data_set = np.asarray(pd.read_csv('../data/net_data/ua_base.csv'))
ua_base_edges = data_set[:,0:2]
print(ua_base_edges)

pd.DataFrame(ua_base_edges).to_csv("../data/net_data/ua_base_edges.csv", header=True, index=False)
