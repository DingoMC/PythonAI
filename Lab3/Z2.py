#%%
import numpy as np
import pandas as pd
data = pd.read_excel('loan_data.xlsx')
columns = list(data.columns)
def qualitative_to_0_1(data, column, value_to_be_1):
    mask = data[column].values == value_to_be_1
    data[column][mask] = 1
    data[column][~mask] = 0
    return data
qualitative_to_0_1(data, 'Gender', 'Male')
qualitative_to_0_1(data, 'Married', 'Yes')

cat_feature = pd.Categorical(data.Property_Area)
one_hot = pd.get_dummies(cat_feature)
data = pd.concat([data, one_hot], axis=1)
data = data.drop(columns=['Property_Area'])

tp1 = 7
fp1 = 26
tn1 = 17
fn1 = 73
sensivity = tp1 / (tp1 + fn1)
precision = tp1 / (tp1 + fp1)
specifity = tn1 / (tn1 + fp1)
accuracy = (tp1 + tn1) / (tp1 + fp1 + tn1 + fn1)
F1 = (2 * sensivity * precision) / (sensivity + precision)
# %%
