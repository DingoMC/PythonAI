#%%
import pandas as pd
data = pd.read_excel('loan_data.xlsx')
columns = list(data.columns)
mask = data['Gender'].values == 'Female'
data['Gender'][mask] = 1
data['Gender'][~mask] = 0
# %%
