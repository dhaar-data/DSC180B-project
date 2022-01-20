import sys
import json
import os

sys.path.insert(0, 'src')
import etl

def main(targets):
    
    if 'etl' in targets:
        
        data_config = json.load(open('config/data-params.json')) # need to switch to split-params.json later
        tweets = etl.split_data(**data_config)
        tweets = etl.clean_data(tweets)
        X_train, X_test, X_validation, y_train, y_test, y_validation = etl.split(data)

if __name__ == '__main__':
    
    targets = sys.argv[1:]
    main(targets)

    