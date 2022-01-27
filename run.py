import sys
import json
import os

sys.path.insert(0, 'src')
import etl
import prediction_model
import relationship_model

def main(targets):
    
    # TO-DO:
    ## Write EDA notebook for scraped data
    ## Inference model, bootstrap correction
    ## Code: Decide where/when we should write data to csv, what parameters should be, etc.
    
    if 'etl' in targets:
        
        data_config = json.load(open('config/data-params.json')) 
        tweets = etl.scrape_data(**data_config)
        cleaned_tweets = etl.clean_data(tweets)
        X_train, X_test, X_validation, y_train, y_test, y_validation = etl.split(cleaned_tweets, **data_config) # writes to a csv
        
    if 'predict' in targets:
        
        tf_idf = prediction_model.vectorizer(X_train)
        
        train_vec = tf_idf.transform(X_train)
        test_vec = tf_idf.transform(X_test)
        val_vec = tf_idf.transform(X_validation)
        
        pred_mdl = prediction_model.model(train_vec, y_train)
        # do we want these to be written to a csv?
        pred_test = prediction_model.predict(test_vec, y_train, pred_mdl)
        pred_val = prediction_model.predict(val_vec, y_validation, pred_mdl)
        
    if 'rel' in targets:
        
        rel_mdl = relationship_model.model(pred_test, y_test)


if __name__ == '__main__':
    
    targets = sys.argv[1:]
    main(targets)

    