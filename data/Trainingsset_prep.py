import numpy as np

dataset = np.loadtxt("data/Trainingsset.csv", delimiter=",")
selected_column = dataset[:, :2]
dataset[:, [0, 1]] = dataset[:, [1, 0]]

np.savetxt("data/Trainingsset_prep.csv", selected_column, delimiter=",")
