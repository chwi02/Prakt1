from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt


train_dataset = np.loadtxt("data/Trainingsset.csv", delimiter=",")
test_dataset = np.loadtxt("data/Testset.csv", delimiter=",")

X=train_dataset[:,1:]
Y=train_dataset[:,0]
X_test=test_dataset[:,1:]
Y_test=test_dataset[:,0]

err=list()
for i in range(1,51):
    rf = RandomForestClassifier(noOfTrees=i, random_state=42)
    rf.fit(X, Y) 
    y_test=rf.predict(X_test)
    err.append(np.sum(y_test!=Y_test))

err=np.array((np.arange(1,51),err))

plt.plot(err[0], err[1])
plt.xlabel('Baeume')           
plt.ylabel('Fehler')           
plt.title('Fehlerverlauf')      

plt.savefig('outputs/aufgabe3.pdf')

