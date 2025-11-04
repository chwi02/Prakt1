import numpy as np
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader, PdfWriter

from binaryTree import tree 



class bDecisionTree:
    def _calGiniImpurity(self,y):
        unique, counts = np.unique(y, return_counts=True)
        N = counts/len(y)
        G = 1 - np.sum(N**2)
        return G

    def _bestSplit(self,X,y,feature):
        G = 1
        bestSplit = np.inf
        XSort = np.unique(X[:,feature].round(self.xDecimals))
        XDiff = (XSort[1:] + XSort[:-1]) / 2
        for i in range(XDiff.shape[0]):
            index = X[:,feature] < XDiff[i]
            G1 = self._calGiniImpurity(y[index])
            G2 = self._calGiniImpurity(y[~index])
            GSplit = np.mean(index)*G1 + np.mean(~index)*G2
            if G > GSplit:
                G = GSplit
                bestSplit = XDiff[i]
        return bestSplit, G

    def _chooseFeature(self,X,y):
        G         = np.zeros(X.shape[1])
        bestSplit = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            ( bestSplit[i] , G[i] ) = self._bestSplit(X,y,i)
        smallest = np.argmin(G)
        return G[smallest], bestSplit[smallest], smallest

    def _ComputeValue(self,y):
        unique, counts = np.unique(y, return_counts=True)
        i = np.argmax(counts)
        return(unique[i])

    def __init__(self,threshold = 0.1, xDecimals = 5, minLeafNodeSize=3):     ###### TODO: optimise minLeafNodeSize
        self.bTree = None
        self.threshold = threshold
        self.xDecimals = xDecimals
        self.minLeafNodeSize = minLeafNodeSize

    def _GenTree(self,X,y,parentNode,branch):
        commonValue = self._ComputeValue(y)
        initG = self._calGiniImpurity(y)
        if  initG < self.threshold or X.shape[0] <= self.minLeafNodeSize:
            self.bTree.addNode(parentNode,branch,commonValue)
            return
            
        (G, bestSplit ,chooseA) = self._chooseFeature(X,y)
        if G > 0.98*initG:
            self.bTree.addNode(parentNode,branch,commonValue)
            return
        
        if parentNode == None: 
            self.bTree = tree(chooseA, bestSplit, '<')
            myNo = 0
        else: 
            myNo = self.bTree.addNode(parentNode,branch,bestSplit,operator='<',varNo=chooseA)

        index = np.less(X[:,chooseA],bestSplit)
        XTrue  = X[index,:] 
        yTrue  = y[index]
        XFalse = X[~index,:]
        yFalse = y[~index]
                
        if XTrue.shape[0] > self.minLeafNodeSize:
            self._GenTree(XTrue,yTrue,myNo,True)
        else:
            commonValue = self._ComputeValue(yTrue)
            self.bTree.addNode(myNo,True,commonValue)
        if XFalse.shape[0] > self.minLeafNodeSize:
            self._GenTree(XFalse,yFalse,myNo,False)
        else:
            commonValue = self._ComputeValue(yFalse)
            self.bTree.addNode(myNo,False,commonValue)
        return()

    def fit(self, X,y):
        self._GenTree(X,y,None,None)
    
    def predict(self, X):
        return(self.bTree.eval(X))
    
    def decision_path(self, X):
        return(self.bTree.trace(X))
        
    def weightedPathLength(self,X):
        return(self.bTree.weightedPathLength(X)) 
        
    def numberOfLeafs(self):
        return(self.bTree.numberOfLeafs())
        
if __name__ == '__main__':        
    dataset = np.loadtxt("data/Trainingsset.csv", delimiter=",")
    testset = np.loadtxt("data/Testset.csv", delimiter=",")
    allData = np.loadtxt("data/AllData.csv", delimiter=",")

    np.random.seed(42)
    MainSet = np.arange(0,dataset.shape[0])
    #Trainingsset = np.random.choice(dataset.shape[0], 120, replace=False)
    #Testset = np.delete(MainSet,Trainingsset)
    XTrain = dataset[:, 1:]
    yTrain = dataset[: ,0]
    XTest = testset[:, 1:]
    yTest = testset[: ,0]
    """
    for i in range(1,50):
        myTree = bDecisionTree(minLeafNodeSize=i)
        myTree.fit(XTrain,yTrain)
        y = myTree.predict(XTest)
        Fehler=np.sum(y!=yTest)
        print('leafs '+str(i)+' :Fehler %e' % Fehler)
    """
    

#######1.2

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