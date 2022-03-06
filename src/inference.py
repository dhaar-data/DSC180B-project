import pandas as pd
import numpy as np
import scipy
import statistics
import random
import sklearn

####
# needs to output estimators, ses, stats for param, nonparam, nocorr, and true. needs to output rmse. POSSIBLY needs to output graph.
####

def conduct_inference(rel_mdl, tf_idf, pred_mdl, bs, features, input_paths, output_paths):
    feature_all = tf_idf.vocabulary_
    features_index = [feature_all[x] for x in features]

    val_x = tf_idf.transform(pd.read_csv(input_paths[0], keep_default_na=False)['tweet_text'])
    val_y = pd.read_csv(input_paths[1])
    val_y_pred = pd.read_csv(input_paths[2], names=['D', 'R'], delim_whitespace = True).to_numpy()
    val_y_pred_nonprob = pred_mdl.predict(val_x)
    
    sample_size = len(val_y)
        
    param_estimators = []
    param_ses = []
    param_t_stats = []

    nonparam_estimators = []
    nonparam_ses = []
    nonparam_t_stats = []

    nocorr_estimators = []
    nocorr_ses = []
    nocorr_t_stats = []

    true_estimators = []
    true_ses = []
    true_t_stats = []

    i = 0 # counter for number of features
    for feature in features_index:
        # remove these later
        
        i += 1
                
        bootstrap_results = bootstrap(bs, sample_size, rel_mdl, val_x.tocsr()[:,feature].todense(), val_y, val_y_pred)
        nocorr_results = no_bootstrap(val_x.tocsr()[:,feature].todense(), val_y_pred_nonprob)
        true_results = no_bootstrap(val_x.tocsr()[:,feature].todense(), val_y['party'])

        param_estimators.append(bootstrap_results['parametric'][0])
        param_ses.append(bootstrap_results['parametric'][1])
        param_t_stats.append(bootstrap_results['parametric'][2])

        nonparam_estimators.append(bootstrap_results['non-parametric'][0])
        nonparam_ses.append(bootstrap_results['non-parametric'][1])
        nonparam_t_stats.append(bootstrap_results['non-parametric'][2])

        nocorr_estimators.append(nocorr_results[0])
        nocorr_ses.append(nocorr_results[1])
        nocorr_t_stats.append(nocorr_results[2])

        true_estimators.append(true_results[0])
        true_ses.append(true_results[1])
        true_t_stats.append(true_results[2])
    
    estimators_df = to_df(features, true_estimators, nocorr_estimators, nonparam_estimators, param_estimators, len(true_estimators))
    ses_df = to_df(features, true_ses, nocorr_ses, nonparam_ses, param_ses, len(true_ses))
    t_stat_df = to_df(features, true_t_stats, nocorr_t_stats, nonparam_t_stats, param_t_stats, len(true_t_stats))
    
    estimators_df.to_csv(output_paths[0], index=False)
    ses_df.to_csv(output_paths[1], index=False)
    t_stat_df.to_csv(output_paths[2], index=False)
    
    # calculating rmses
    rmses = {'estimators': [], 'ses': [], 't_stats': []}
#     for cat in pd.unique(estimators_df['category']):
#         error = rmse(estimators_df[estimators_df['category'] == cat]['true'].reset_index(drop=True),
#                      estimators_df[estimators_df['category'] == cat]['predicted'].reset_index(drop=True))
#         rmses['estimators'].append((cat, error))
    
#     for cat in pd.unique(ses_df['category']):
#         error = rmse(ses_df[ses_df['category'] == cat]['true'].reset_index(drop=True),
#                      ses_df[ses_df['category'] == cat]['predicted'].reset_index(drop=True))
#         rmses['ses'].append((cat, error))

#     for cat in pd.unique(t_stat_df['category']):
#         error = rmse(t_stat_df[t_stat_df['category'] == cat]['true'].reset_index(drop=True),
#                      t_stat_df[t_stat_df['category'] == cat]['predicted'].reset_index(drop=True))
#         rmses['t_stats'].append((cat, error))

#     print(rmses)
    return estimators_df, ses_df, t_stat_df, rmses

def rmse(y, y_pred):
    """
    Calculates rmse.
    
    Input: y observed outcomes, y_pred predicted outcomes
    Output: rmse of y_pred
    """
    return sum([(y[i] - y_pred[i])**2/len(y) for i in range(len(y))])**0.5

def standard_error(X, y_pred):
    """
    Calculates coefficient's standard error for categorical outcomes.
    
    Input: x covariates, y_pred predicted outcomes
    Output: standard error of coefficient
    """

    new_x = np.hstack([np.ones((X.shape[0], 1)), X])

    V = np.diagflat(np.product(y_pred, axis=1))
    cov = np.linalg.inv(np.dot(np.dot(new_x.T, V), new_x))
    return np.sqrt(np.diag(cov))

def no_bootstrap(val_x, val_y):
    """
    Inference without bootstrap. Depending on val_y input, this function can be used for the no correction approach or to find the true values.
    """
    inf_mdl = sklearn.linear_model.LogisticRegression().fit(val_x, val_y)
    se = standard_error(val_x, inf_mdl.predict_proba(val_x))[1]
    
    return [inf_mdl.coef_[0][0], se, inf_mdl.coef_[0][0]/ se]

def bootstrap(bs, sample, rel_mdl, val_x, val_y, val_y_pred):
    """
    Corrects post-prediction inference using the categorical bootstrap approach. 
    
    Input: bs number of bootstraps, sample number of sampling with replacement, val_x validation covariates, val_y validation outcomes,
           val_y_pred validation predicted outcomes
    Output: dictionary containing the parametric and non-parametric corrected estimator, standard error, and t-statistic
    """
    # class_dict = {0: 'D', 1: 'R'}
    estimators = []
    ses = []
    
    for i in range(bs):

        # sample
        sample_ix = random.choices(range(val_x.shape[0]), k=sample)
        sample_x = val_x[sample_ix,:]
        sample_y = val_y_pred[sample_ix,:]

        # using relationship model to predict probabilities and find the predicted y
        probabilities = rel_mdl.predict_proba(sample_y)
        sim_probabilities = np.random.binomial(1, probabilities)
        new_predicted_y = ['D' if i[1] == 0 else 'R' for i in sim_probabilities]

        # new_predicted_y = []
        # for row in sample_y:
        #     probability = np.random.multinomial(1, row)
        #     new_predicted_y.append(class_dict[np.where(probability==1)[0][0]])
        
        # fitting inference model
        new_mdl = sklearn.linear_model.LogisticRegression().fit(sample_x, new_predicted_y) 
        new_mdl_y = new_mdl.predict_proba(sample_x)

        estimators.append(new_mdl.coef_[0][0])
        ses.append(standard_error(sample_x, new_mdl_y)[1])
        
    return {
        'parametric': [statistics.median(estimators), statistics.median(ses), statistics.median(estimators) / statistics.median(ses)],
        'non-parametric': [statistics.median(estimators), statistics.stdev(estimators), statistics.median(estimators) / statistics.stdev(estimators)]
    }

def to_df(feature_names, true, no_corr, nonparam, param, n):
    df = pd.DataFrame(data={'true': true,
                            'predicted': no_corr,
                            'category': ['no_correction']*n,
                            'feature': feature_names})

    df = pd.concat([df, pd.DataFrame(data={'true': true,
                                           'predicted': nonparam,
                                           'category': ['non_parametric_bootstrap']*n,
                                           'feature': feature_names})], ignore_index=True)

    df = pd.concat([df, pd.DataFrame(data={'true': true,
                                           'predicted': param,
                                           'category': ['parametric_bootstrap']*n,
                                           'feature': feature_names})], ignore_index=True)
    
    return df
