import pandas as pd
import sklearn
from sklearn import neighbors

def model(X_path, y_path):
    """
    Builds knn relationship model.
    """
    
    X = pd.read_csv(X_path, names=['D', 'R'], sep=' ', keep_default_na=False)
    y = pd.read_csv(y_path, keep_default_na=False)['party']
    
    mdl = neighbors.KNeighborsClassifier().fit(X, y)
    
    return mdl
