import numpy as np
import pandas as pd
data_excel = pd.read_excel("lab_1.xlsx")
cols = list(data_excel.columns)
vals = data_excel.values
arr1 = vals[::2,:]
arr2 = vals[1::2,:]
diff = arr1 - arr2
#2
avg = vals.mean()
std = vals.std()
arr3 = (vals - avg) / std
#3
std2 = vals.std(axis=0)
avg2 = vals.mean(axis=0)
arr4 = (vals - avg2) / (std2 + np.spacing(std2))
#4
arr5 =  std2 / (avg + np.spacing(std2))
#5
arr6 = np.argmax(arr5)
#6
arr7 = (vals > vals.mean(axis=0)).sum(axis=0)
#7
maxv = vals.max()
max_cols = vals.max(axis=0)
cols = np.array(cols)
col_max = cols[maxv == max_cols]
print(col_max)
# 8
# 9

