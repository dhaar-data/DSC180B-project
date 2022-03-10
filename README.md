# DSC180B-project
This is an exploration of the bootstrap post-prediction inference approach introduced by Wang et al. in [Methods for correcting inference based on outcomes predicted by machine learning](https://www.pnas.org/content/117/48/30266/tab-article-info) through a study of tweets and their corresponding political alignment in the US. When we look at a tweet, what are the kinds of words or phrases that most strongly indicate the tweet's alignment to Democrats or Republicans? In today's political climate, what topics or figures are most heavily scrutinized by one party or another? We seek to find these key figures, phrases, and topics through a statistical analyses of political tweets.

As said above, this statistical analyses also functions as an investigation of the bootstrap post-prediction inference approach. Post-prediction--or postpi, as Wang et al. calls it--is the use of predicted outcomes in lieu of observed outcomes during inferential analysis. If postpi is conducted without accounting for the use of predicted outcomes as is often the case, this leads to issues with bias and standard , among other things. While the aforementioned bootstrap post-prediction inference approach corrects these issues for a wide range of datasets, we seek to study its applicability specifically towards text data, particularly in political science. 

## Build Instructions
This project has already provided pre-scraped Twitter data in `data/out/raw`.
* Data Collection: To clean and split the data into train-test-validation sets, run `python run.py data`. The datasets will be stored in `data/out/clean` in separate csvs for covariates and outcomes. 
* EDA: To conduct EDA on the data, run `python run.py data eda`. The output will be three figures in `results/figures` as .png files.
* Prediction: To build the prediction model and predict the outcomes of the datasets, run `python run.py data predict`. The predicted values will be stored in `data/out/predicted` in txt files.
* Relationship: To build the relationship model based on the predicted and observed outcomes from the test dataset, run `python run.py data predict rel`.
* Inference/Bootstrapping: To conduct the bootstrap post-prediction inference on selected features, run `python run.py data predict rel inference`, equivalent to running all targets (see last bullet point). Features you want to conduct inference for can be adjusted in `config/inference-params.json`. This will output estimators, standard errors, and t-statistics of the features in a csv file located at `results/inference`.
* Run all:
    * To run the entire process on the dataset specified in `config/data-params.json`, run `python run.py all`
    * To run the entire process on test data in `test/testdata`, run `python run.py test`
    
## Running the Project
1. Clone this repo
    ```
    git clone https://github.com/dhaar-data/DSC180B-project.git
    ```
2. Build and run the docker image
    ```
    docker build -t ##
    docker run --rm -it ## /bin/bash.
    ```
3. Run the project according to build instructions above. As an example:
    ```
    python run.py all
    ```
    
## Project Website
```
https://dhaar-data.github.io/DSC180B-project/
```
