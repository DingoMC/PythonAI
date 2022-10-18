# %%
import matplotlib.pyplot as plt
import numpy as np
x = np.arange(-5, 5, 0.01)

# %% Wykres 1
y = np.tanh(x)
plt.plot(x, y)

# %% Wykres 2
y = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
plt.plot(x, y)

# %% Wykres 3
y = 1 / (1 + np.exp(-x))
plt.plot(x, y)

# %% Wykres 4
y = np.where(x <= 0, 0, x)
plt.plot(x, y)

# %% Wykres 5
y = np.where(x <= 0, np.exp(x) - 1, x)
plt.plot(x, y)

# %%
