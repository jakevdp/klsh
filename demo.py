import numpy as np
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from klsh import KernelLSH

data = load_iris()
X = data.data
y = data.target

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)


klsh = KernelLSH(Xtrain, nbits=8, kernel='rbf')
print klsh.hash_table_


for i in range(len(Xtest)):
    nbrs = klsh.query(Xtest[i])
    print ytest[i], ytrain[nbrs]
