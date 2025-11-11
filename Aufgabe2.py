import numpy as np
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader, PdfWriter
import pickle

from project.CARTDecisionTree import bDecisionTree

rerun_appl=True # falls Modell erstellt werden muss

dataset = np.loadtxt("data/Trainingsset.csv", delimiter=",")
testset = np.loadtxt("data/Testset.csv", delimiter=",")
allData = np.loadtxt("data/AllData.csv", delimiter=",")

XTrain = dataset[:, 1:]
yTrain = dataset[: ,0]
XTest = testset[:, 1:]
yTest = testset[: ,0]

XTrain_f=XTrain[:,[0,6]]
XTest_f=XTest[:,[0,6]]

if rerun_appl:
    myTree = bDecisionTree(minLeafNodeSize=4)
    myTree.fit(XTrain_f,yTrain)
    with open('models/CART2.pkl', 'wb') as f:
        model = pickle.dump(myTree,f)
else:
    with open('models/CART2.pkl', 'rb') as f:
        myTree = pickle.load(f)

y = myTree.predict(XTest_f)

XX, YY = np.mgrid[      # Wertebereich auf den in der Aufgabe gegebenen angepasst
    11:15:0.02,
    0:6:0.02      
]
X = np.array([XX.ravel(), YY.ravel()]).T
Z = myTree.predict(X).reshape(XX.shape)



# Plot
fig, (ax_plot, ax_text) = plt.subplots(
2, 1,
figsize=(8.27, 11.69),
gridspec_kw={'height_ratios': [5, 1]}  # Verhältnis Plot zu Text
)

fig.suptitle("Aufgabe 2", fontsize=16)


ax_plot.pcolormesh(XX, YY, Z, cmap=plt.cm.Set1, shading='auto')
ax_plot.scatter(XTest[:,[0]], XTest[:,[6]] , c=yTest, s=60, alpha=0.6)
ax_plot.set_xlabel("Feature 1 (Alcohol)")
ax_plot.set_ylabel("Feature 7 (Flavanoids)")
ax_plot.set_title("pcolormesh-Graph")

# Text hinzufügen
ax_text.axis("off")
ax_text.text(0.5, 0.5, "Mit dem bloßen Auge sind 4 Falsch klassifizierte Werte erkennbar.\n Die in Aufgabe 1 festgestellten Fehler lassen sich in diesem Plot bestätigen.",
            ha="center", va="center", fontsize=12)


plt.savefig('outputs/aufgabe2.pdf')


