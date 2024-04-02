import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

lines = pd.read_csv('dso_2.csv', sep=',')
print(lines)
fig, ax = plt.subplots()
x = lines.iloc[:,0]
y = [item - 1.65 for item in lines.iloc[:,1]] 

period = 1 / (125.4*10**6) *2 
phase = 0.5 * period
z1 = 2 * ((x-(0*10**(-9))) % period >= phase) - 1
z2 = 2 * ((x+(2.5*10**(-9))) % period >= phase) - 1
ax.plot(x,y,x,z2)
plt.show()