import sys
import json
import os

sys.path.insert(0, 'src')
import etl
import eda
import prediction
import relationship
import inference

def main(targets):
    
    if 'data' in targets:
        
        data_config = json.load(open('config/data-params.json'))
        etl.transform_data(**data_config)
        
    if 'eda' in targets:
        
        eda_config = json.load(open('config/eda-params.json'))
        eda.conduct_eda(**eda_config)
    
    if 'predict' in targets:
        
        pred_config = json.load(open('config/predict-params.json'))
        
        tf_idf, pred_mdl = prediction.build_model(**pred_config)
        prediction.transform_predict(tf_idf, pred_mdl, **pred_config)
        
    if 'rel' in targets:
        
        rel_config = json.load(open('config/relationship-params.json'))
        rel_mdl = relationship.model(**rel_config)
        
    if 'inference' in targets:
        
        inf_config = json.load(open('config/inference-params.json'))
        estimators_df, ses_df, t_stat_df = inference.conduct_inference(rel_mdl, tf_idf, pred_mdl, **inf_config)
        
    # if 'scrape' in targets:
        
        # offer scrape validation data function
        # tweets = etl.scrape_data(**data_config)
        
    if 'all' in targets:
        
        data_config = json.load(open('config/data-params.json'))
        eda_config = json.load(open('config/eda-params.json'))
        pred_config = json.load(open('config/predict-params.json'))
        rel_config = json.load(open('config/relationship-params.json'))
        inf_config = json.load(open('config/inference-params.json'))
        
        etl.transform_data(**data_config)
        
        eda.conduct_eda(**eda_config)
        
        tf_idf, pred_mdl = prediction.build_model(**pred_config)
        prediction.transform_predict(tf_idf, pred_mdl, **pred_config)
        
        rel_mdl = relationship.model(**rel_config)
        estimators_df, ses_df, t_stat_df = inference.conduct_inference(rel_mdl, tf_idf, pred_mdl, **inf_config)
        
    if 'test' in targets:
        
        test_config = json.load(open('config/test-params.json'))
        pred_config = json.load(open('config/predict-params.json'))
        rel_config = json.load(open('config/relationship-params.json'))
        inf_config = json.load(open('config/inference-params.json'))
        
        etl.transform_data(**test_config)
        
        tf_idf, pred_mdl = prediction.build_model(**pred_config)
        prediction.transform_predict(tf_idf, pred_mdl, **pred_config)
        
        rel_mdl = relationship.model(**rel_config)
        estimators_df, ses_df, t_stat_df = inference.conduct_inference(rel_mdl, tf_idf, pred_mdl, **inf_config)

if __name__ == '__main__':
    
    targets = sys.argv[1:]
    main(targets)

    