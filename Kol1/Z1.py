#%%
import numpy as np
import pandas as pd
data = pd.read_excel('ex.xlsx') # Na kolosie bedzie read_csv jeden gwizd
col = data.columns
val = data.values
mean_col = np.mean(val, axis=0)
mean_std = np.std(val)
difference = val - mean_std
max_row_val = np.max(val, axis=1)
arr2 = val * 2
col_np = np.array(col)
col_max = col_np[np.max(val) == np.max(val, axis=0)]
arr9 = (val < mean_std).sum(axis=0)

# %%
