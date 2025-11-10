import numpy as np 
from project.CARTDecisionTreeRF import bDecisionTree
import numpy as np
import matplotlib.pyplot as plt

class randomForestKlassifikation:
    def __init__(self,noOfTrees=10,threshold = 10**-8, xDecimals = 8, minLeafNodeSize=3, perc=1.0, random_state=None):
        self.perc = perc
        self.threshold = threshold
        self.xDecimals = xDecimals
        self.minLeafNodeSize = minLeafNodeSize
        self.bTree = []
        self.noOfTrees = noOfTrees

        if random_state is not None:
            np.random.seed(random_state)

        for i in range(noOfTrees):
            tempTree = bDecisionTree(threshold = self.threshold, xDecimals = self.xDecimals , minLeafNodeSize=self.minLeafNodeSize)
            self.bTree.append(tempTree)
            
    def fit(self,X,y):
        self.samples = []
        for i in range(self.noOfTrees):
            bootstrapSample = np.random.randint(X.shape[0],size=int(self.perc*X.shape[0]))
            self.samples.append(bootstrapSample)     #*\label{code:realRF:0}
            bootstrapX = X[bootstrapSample,:]
            bootstrapY = y[bootstrapSample]
            self.bTree[i].fit(bootstrapX,bootstrapY)
    
    def predict(self,X):
        ypredictTemp = np.zeros( (X.shape[0],self.noOfTrees) )
        ypredict = np.zeros( X.shape[0] )
        for i in range(self.noOfTrees):
            ypredictTemp [:,i] = self.bTree[i].predict(X)
        #ypredict = np.zeros(X.shape[0])
        for j in range(X.shape[0]):
            unique, counts = np.unique(ypredictTemp[j,:], return_counts=True)
            i = np.argmax(counts)
            ypredict[j] = unique[i]
        return(ypredict)
        
if __name__ == '__main__':   
    train_dataset = np.loadtxt("data/Trainingsset.csv", delimiter=",")
    test_dataset = np.loadtxt("data/Testset.csv", delimiter=",")

    X=train_dataset[:,1:]
    Y=train_dataset[:,0]
    X_test=test_dataset[:,1:]
    Y_test=test_dataset[:,0]

    err=list()
    for i in range(1,51):
        rf = randomForestKlassifikation(noOfTrees=i, random_state=42)
        rf.fit(X, Y) 
        y_test=rf.predict(X_test)
        err.append(np.sum(y_test!=Y_test))

    err=np.array((np.arange(1,51),err))

    plt.plot(err[0], err[1])
    plt.xlabel('Baeume')           
    plt.ylabel('Fehler')           
    plt.title('Fehlerverlauf')      

    plt.savefig('outputs/aufgabe3.pdf')

    #plt.figure()
    #plt.plot(list(n_trees_range), errors, marker='o')   # GEÄNDERT: Plot-Befehle verbessert
    #plt.xlabel('Anzahl Bäume')
    #plt.ylabel('Fehler (Anzahl falscher Klassifikationen)')
    #plt.title('Fehlerverlauf Random Forest')
    #plt.grid(True)
    #plt.savefig('outputs/aufgabe3.pdf')
    #plt.show() 


    myForest = randomForestKlassifikation(noOfTrees=24,minLeafNodeSize=5,threshold=2) #randomForestRegression
    myForest.fit(X,Y)
    yPredict = np.round(myForest.predict(X_test))
    yDiff = yPredict - Y_test
    print('Mittlere Abweichung: %e ' % (np.mean(np.abs(yDiff))))
                