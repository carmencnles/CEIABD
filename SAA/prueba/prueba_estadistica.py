import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

datos = [5, 7, 8, 8, 9, 10, 10, 14, 14, 15, 20]
print("Media:", np.mean(datos))
print("Mediana:", np.median(datos))
print("Varianza:", np.var(datos))
print("Cuartiles:", np.percentile(datos,[25, 50, 75]))
plt.boxplot(datos)
plt.show()