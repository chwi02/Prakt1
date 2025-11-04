import numpy as np

dataset = np.loadtxt("data/Trainingsset.csv", delimiter=",")
selected_column = dataset[:, :2]
np.savetxt("Trainingsset_prep.csv", selected_column, delimiter=",")
