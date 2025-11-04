import numpy as np
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader, PdfWriter

from project.binaryTree import tree 
from project.CARTDecisionTree import XTrain, XTest, yTrain, yTest, bDecisionTree





XTrain_f=XTrain[:,[0,6]]
XTest_f=XTest[:,[0,6]]
"""
for i in range(1,50):
    myTree = bDecisionTree(minLeafNodeSize=i)
    myTree.fit(XTrain_f,yTrain)
    y = myTree.predict(XTest_f)
    Fehler=np.sum(y!=yTest)
    print('leafs '+str(i)+' :Fehler %e' % Fehler)
"""
myTree = bDecisionTree(minLeafNodeSize=3)
myTree.fit(XTrain_f,yTrain)
y = myTree.predict(XTest_f)

"""
XX, YY = np.mgrid[
    XTrain_f[:,0].min():XTrain_f[:,0].max():0.005,
    XTrain_f[:,1].min():XTrain_f[:,1].max():0.005
]    
X = np.array([XX.ravel(), YY.ravel()]).T
Z = myTree.predict(X).reshape(XX.shape)
"""

XX, YY = np.mgrid[
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

ax_plot.pcolormesh(XX, YY, Z, cmap=plt.cm.Set1, shading='auto')
ax_plot.scatter(XTest[:,[0]], XTest[:,[6]] , c=yTest, s=60, alpha=0.6)
ax_plot.set_xlabel("Feature 1 (Alcohol)")
ax_plot.set_ylabel("Feature 7 (Flavanoids)")
ax_plot.set_title("pcolormesh-Graph")

# Text hinzufügen
ax_text.axis("off")
ax_text.text(0.5, 0.5, "Hier steht der Text unter dem Graphen.",
            ha="center", va="center", fontsize=12)


plt.savefig('outputs/pcolormesh.pdf')
#plt.show()

output = PdfWriter()
pdfOne = PdfReader(open("outputs/scatterplots.pdf", "rb"))
pdfTwo = PdfReader(open("outputs/pcolormesh.pdf", "rb"))

output.add_page(pdfOne.pages[0])
output.add_page(pdfTwo.pages[0])

with open("outputs/plots.pdf", "wb") as outputStream:
    output.write(outputStream)