# DSC180B-project
This is an exploration of the bootstrap post-prediction inference approach introduced by Wang et al. in *Methods for correcting inference based on outcomes predicted by machine learning* through a study of US political tweets. 

## Build Instructions
* Data Collection: For cleaning and splitting the data into train-test-validation sets, run `python run.py data`.
* EDA: TO-DO
* Prediction: For creating the prediction model, run `python run.py predict`.
* Relationship: For creating the relationship model, run `python run.py relationship`.
* Inference/Bootstrapping: TO-DO
* Scrape: TO-DO
    * Note that a Twitter developer account is necessary for this step. Once you have made an account, go to `config/data-params.json` and input your access keys and tokens provided to you by Twitter. 
* Run all:
    * To run the entire process on the dataset specified in `config/data-params.json`: run `python run.py all`
    * To run on test data: run `python run.py test`
