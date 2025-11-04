from project.Aufgabe1_prep import rerun
import pickle
import numpy as np

rerun_appl=False

testset = np.loadtxt("data/Testset.csv", delimiter=",")
XTest = testset[:, 1:]
yTest = testset[: ,0]

if rerun_appl:
    rerun()


with open('models/CART1.pkl', 'rb') as f:
    model = pickle.load(f)


y = model.predict(XTest)
Fehler=np.sum(y!=yTest)
print(Fehler)

### PDF mit Plots in output/scatterplots.pdf

