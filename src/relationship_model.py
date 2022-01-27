import sklearn
from sklearn import neighbors

def model(X, y):
    """
    Create relationship model, fitted to the testing set.
    """
    mdl = neighbors.KNeighborsClassifier().fit(X, y)
    return mdl
