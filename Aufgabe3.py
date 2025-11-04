from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader, PdfWriter
 


train_dataset = np.loadtxt("data/Trainingsset.csv", delimiter=",")
test_dataset = np.loadtxt("data/Testset.csv", delimiter=",")

X=train_dataset[:,1:]
Y=train_dataset[:,0]
X_test=test_dataset[:,1:]
Y_test=test_dataset[:,0]

err=list()
for i in range(1,51):
    rf = RandomForestClassifier(n_estimators=i, random_state=42)
    rf.fit(X, Y) 
    y_test=rf.predict(X_test)
    #print(y_test)
    err.append(np.sum(y_test!=Y_test))

err=np.array((np.arange(1,51),err))
#print(err)

plt.plot(err[0], err[1])
plt.xlabel('Baeume')           
plt.ylabel('Fehler')           
plt.title('Fehlerverlauf')      
pdf_path = "outputs/aufgabe3.png"
#plt.savefig(pdf_path, format='png', bbox_inches= "tight")

plt.savefig('outputs/aufgabe3.pdf')
#plt.show()

with open("outputs/scatterplots.pdf", "rb") as f1, open("outputs/pcolormesh.pdf", "rb") as f2, open("outputs/aufgabe3.pdf", "rb") as f3:
    pdfOne = PdfReader(f1)
    pdfTwo = PdfReader(f2)
    pdfThree = PdfReader(f3)
    output = PdfWriter()
    output.add_page(pdfOne.pages[0])
    output.add_page(pdfTwo.pages[0])
    output.add_page(pdfThree.pages[0])
    with open("outputs/plots.pdf", "wb") as out:
        output.write(out)