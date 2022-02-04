import pandas as pd
import numpy as np
import sklearn
from sklearn import feature_extraction
from sklearn import svm

def build_model(X_train, y_train, **kwargs):
    """
    Vectorize train data (TF-IDF vectorizer with 1-3 grams and 200000 features) and build model (SVC, kernel=linear, C=1.5)
    """
    X = pd.read_csv(X_train, keep_default_na=False)['tweet_text']
    y = pd.read_csv(y_train, keep_default_na=False)['party']
    
    tfidf_vectorizer = feature_extraction.text.TfidfVectorizer(ngram_range=(1, 3), max_features=200000)
    X_vec = tfidf_vectorizer.fit_transform(X)
    
    mdl = svm.SVC(kernel='linear', C=1.5, probability=True)
    mdl.fit(X_vec, y)
    
    return tfidf_vectorizer, mdl
    
def transform_predict(vect, mdl, X_input, X_output, **kwargs):
    """
    Vectorize and predict probabilities for each class per data point. Writes predicted values to csv.
    """
    assert len(X_input) == len(X_output)
    
    
    for i in range(len(X_input)):
        X = pd.read_csv(X_input[i], keep_default_na=False)['tweet_text']
        
        vect_data = vect.transform(X)
        pred_data = mdl.predict_proba(vect_data)

        np.savetxt(X_output[i], pred_data)
            
    return
