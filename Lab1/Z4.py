# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
data_excel = pd.read_excel("lab_1.xlsx")
cols = list(data_excel.columns)
vals = data_excel.values
corr_matrix = np.array(data_excel.corr())
# %%

fig, ax  = plt.subplots(7, 7, figsize = (10, 10))
for i in range (0, 7):
    for j in range (0, 7):
        x = vals[:,i]
        y = vals[:,j]
        ax[i,j].scatter(x, y)
        x1 = np.arange(0, 100, 1)
        y1 = x1 * corr_matrix[i][j]
        ax[i,j].plot(x1,y1)
        
# %%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for x in range (0, 7):
    for y in range (0, 7):
        z = corr_matrix[x][y]
        ax.scatter(x, y, z, c=z)
# %%
