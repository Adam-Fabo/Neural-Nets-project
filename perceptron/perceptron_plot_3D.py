# Created by: Adam Fabo
# Date: 22.5.2022
# Created at HMU Crete
# Class: Neural Networks
# File contains script to plot 3D scatter of data with plane of perceptron (Chapter 4 in documentation)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv('../../data_banknote_auth_trimmed.txt', sep=",", header=None)

# data description
# 1. variance of Wavelet Transformed image (continuous)
# 2. skewness of Wavelet Transformed image (continuous)
# 3. curtosis of Wavelet Transformed image (continuous)
# 4. entropy of image (continuous)
# 5. class (integer)

data.columns = ["Variance", "Skewness", "Curtosis", "Entropy", "Class"]

data_0 =  data.where(data["Class"]==0)
data_1 =  data.where(data["Class"]==1)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

print(data_0["Class"].to_numpy())

ax.scatter(data_0["Variance"].to_numpy(), data_0["Skewness"].to_numpy(), data_0["Curtosis"].to_numpy(), c = data_0["Class"].to_numpy())
ax.scatter(data_1["Variance"].to_numpy(), data_1["Skewness"].to_numpy(), data_1["Curtosis"].to_numpy(), c = 'gold')

ax.set_xlabel('Variance')
ax.set_ylabel('Skewness')
ax.set_zlabel('Curtosis')
ax.legend(["Class 1","Class 2"])


# https://stackoverflow.com/questions/3461869/plot-a-plane-based-on-a-normal-vector-and-a-point-in-matlab-or-matplotlib

# weights from trained perceptron

#normal = [-1.42173728, -1.57387856, -1.63842546]
#d = 2.05

normal = [-1.26 , -1.43 , -1.49]
d = 1.9

# a plane is a*x+b*y+c*z+d=0
# [a,b,c] is the normal.

# create x,y
xx, yy = np.meshgrid(range(-15,15), range(-15,15))

# calculate corresponding z
z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]

# plot the surface
#plt3d = plt.figure().gca(projection='3d')
ax.plot_surface(xx, yy, z,alpha = 0.5)
plt.show()







