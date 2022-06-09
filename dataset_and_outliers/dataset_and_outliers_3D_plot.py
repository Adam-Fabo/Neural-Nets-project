# Created by: Adam Fabo
# Date: 22.5.2022
# Created at HMU Crete
# Class: Neural Networks
# File contains script to plot 3D graph of input data (Chapter 2 in documentation)

# plots 3d scatter of data
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv('data_banknote_auth_trimmed.txt', sep=",", header=None)

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
plt.show()


