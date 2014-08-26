import numpy as np
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from klsh import KernelLSH

data = load_iris()
X = data.data
y = data.target

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1,
                                                random_state=42)


klsh = KernelLSH(Xtrain, nbits=8, kernel='rbf', random_state=42)


print("Labels on results for 8-bit hash bins of iris data:")
for i, nbrs in enumerate(klsh.query_top_k(Xtest, 10)):
    print(ytest[i], ytrain[nbrs])
