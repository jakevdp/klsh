from sklearn.metrics import pairwise_kernels


def inner_product(X, Y):
    return np.squeeze(np.dot(X, np.expand_dims(Y, -1)), -1)

def rbf(X, Y):
    return pairwise_kernels.rbf(X, Y)

    
