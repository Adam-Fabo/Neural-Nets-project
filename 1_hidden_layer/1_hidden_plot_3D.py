# Created by: Adam Fabo
# Date: 22.5.2022
# Created at HMU Crete
# Class: Neural Networks
# File contains script to plot 3D error of neural network with 1 hidden layer (Chapter 4 in documentation)



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm

data = pd.DataFrame()



num_of_neurons= 10

num_of_epochs = 3000
epoch_step = 100

for i in range(1,num_of_neurons+1):
    for j in range(epoch_step,num_of_epochs+epoch_step,epoch_step):
        d =  pd.read_csv("partial_train/" + str(i) + "_neurons/" + str(j) + "_epochs.csv")
        data = pd.concat([data,d])


total_error = []
for neuron_num in range(1,num_of_neurons+1):


    row_error = []
    for epoch_num in range(epoch_step ,num_of_epochs + epoch_step, epoch_step):

        cumulative_error = 0

        for j in range(10):

            epochs = data[(data["Epochs"] == epoch_num) & (data["Neurons"] == neuron_num)].dropna(thresh=1)
            #print(epoch_num, " ", j)
            err = epochs.iloc[j]["Error"]

            # replace right and left bracket
            err = err.replace('[','').replace(']','')

            # load array from string
            err = np.fromstring(err,sep = ",",dtype=float)

            # get last known error
            cumulative_error += err[-1]

            #print(err.size)
        row_error.append(cumulative_error/10)

    total_error.append(row_error)


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
X = np.arange(1, 11, 1)
Y = np.arange(100, 3100, 100)

X, Y = np.meshgrid(Y, X)

# Plot the surface.
surf = ax.plot_surface(Y, X, np.array(total_error), cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)


fig.colorbar(surf)

ax.set_xlabel('Num of neurons')
ax.set_ylabel('Num of epochs')
ax.set_zlabel('Error')
#ax.set_zscale('log')
ax.set_title("Error at end of traning")

plt.show()



