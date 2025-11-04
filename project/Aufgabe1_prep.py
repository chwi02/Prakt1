import numpy as np
from project.CARTDecisionTree import bDecisionTree
from project.plot import gen_plot
import pickle

def rerun():
    

    dataset = np.loadtxt("data/Trainingsset.csv", delimiter=",")
    testset = np.loadtxt("data/Testset.csv", delimiter=",")
    allData = np.loadtxt("data/AllData.csv", delimiter=",")
    np.random.seed(42)
    MainSet = np.arange(0,dataset.shape[0])

    XTrain = dataset[:, 1:]
    yTrain = dataset[: ,0]     
    XTest = testset[:, 1:]
    yTest = testset[: ,0]


    with open('./models/CART1.pkl', 'wb') as f:
            myTree = bDecisionTree(minLeafNodeSize=5)
            myTree.fit(XTrain,yTrain)
            pickle.dump(myTree, f)

    for i in range(1,50,2):
        myTree = bDecisionTree(minLeafNodeSize=i)
        myTree.fit(XTrain,yTrain)
        y = myTree.predict(XTest)
        Fehler=np.sum(y!=yTest)
        print('leafs '+str(i)+' :Fehler %e' % Fehler)

    gen_plot()

