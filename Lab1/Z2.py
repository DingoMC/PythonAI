#%%
import numpy as np
import pandas as pd
data_excel = pd.read_excel("lab_1.xlsx")
cols = list(data_excel.columns)
vals = data_excel.values
even_rows = vals[::2,:]
odd_rows = vals[1::2,:]
diff = even_rows - odd_rows

#%% 2
avg = vals.mean()
std = vals.std()
arr3 = (vals - avg) / std

#%% 3
std2 = vals.std(axis=0)
avg2 = vals.mean(axis=0)
arr4 = (vals - avg2) / (std2 + np.spacing(std2))

#%% 4
arr5 =  std2 / (avg + np.spacing(std2))

#%% 5
arr6 = np.argmax(arr5)

#%% 6
arr7 = (vals > vals.mean(axis=0)).sum(axis=0)

#%% 7
maxv = vals.max()
max_cols = vals.max(axis=0)
cols = np.array(cols)
col_max = cols[maxv == max_cols]

#%% 8
array_mask = vals == 0
array_zeros = np.sum(array_mask, axis=0)
max_zeros = max(array_zeros)
cols = np.array(cols)
max_zeros_col = cols[max_zeros == array_zeros]

#%% 9
even_rows_sum = np.sum(even_rows, axis=0)
odd_rows_sum = np.sum(odd_rows, axis=0)
array_mask = even_rows_sum > odd_rows_sum
cols = np.array(cols)
arr9 = cols[array_mask]
# %%
