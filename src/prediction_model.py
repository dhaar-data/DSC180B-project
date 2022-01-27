import sklearn
from sklearn import feature_extraction
from sklearn import svm

def vectorizer(data):
    """
    TF-IDF vectorizer. 
    """
    tfidf_vectorizer = feature_extraction.text.TfidfVectorizer(ngram_range=(1, 3), max_features=200000)
    tfidf_vectorizer.fit(data)
    return tfidf_vectorizer
    
def model(X, y):
    """
    Fit SVC model to training data.
    """
    mdl = svm.SVC(kernel='linear', C=1.5, probability=True)
    mdl.fit(X, y)
    return mdl

def predict(X, y, mdl):
    """
    Predict probabilities for each class per data point.
    """
    X_predicted = mdl.predict_proba(X)
    
    # write csv for predicted?
    # pd.concat([X, y], axis=1).to_csv(output)
    return X_predicted
