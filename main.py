
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier # this is an sklearn wrapper for XGBoost. This allows us to use sklearn’s Grid Search with parallel processing in the same way we did for GBM
from sklearn import cross_validation, metrics
from sklearn.cross_validation import train_test_split

from matplotlib import pyplot
import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

from feature_engineering import process_features

# Aux functions
# ------------------------------------------------------------------------------
# ip calculation
def ip(y_target, y_pred):
    return 100*(2*(metrics.roc_auc_score(y_target, y_pred))-1)

# if __name__ == '__main__':

# Loading data
# ------------------------------------------------------------------------------
# define input data source
data_file = 'g1_raw.csv'

# principal variables
col_id              = 'id'
col_target          = 'TARGET'
col_muestra         = 'MUESTRA'
col_pred_old_mean   = 'meanpd_hatold'
col_pred_old_max    = 'maxpd_hatold'

# import data
print ('Reading the data...')
df    = pd.read_csv('data\\{}'.format(data_file), delimiter=None, sep=';')

# predictors
predictors = [x for x in df.columns if x not in [col_id, col_muestra, col_pred_old_mean, col_pred_old_max]]

# Splitting the data
# ------------------------------------------------------------------------------
train = df.loc[df['MUESTRA']=='Train']
test  = df.loc[df['MUESTRA']=='Test']
# otime = df.loc[df['MUESTRA']=='OTime']

# select predictors to process
X_train = train[predictors]
X_test  = test[predictors]
# X_otime = otime[predictors]

# Preprocessing features
# ------------------------------------------------------------------------------
X_train, X_test, features = process_features(X_train, X_test, scale=False, remove_missing=False, fill_na=True, per_missing=0.5, target=col_target)

# retrieve id
train = pd.concat([train[col_id], X_train], axis=1)
test  = pd.concat([test[col_id], X_test], axis=1)

# ------------------------------------------------------------------------------
# XGBOOST
# ------------------------------------------------------------------------------
start_time = time.time()

# declare classifier wrapper
clf = XGBClassifier(
    booster = 'gbtree',
    learning_rate =0.01,
    n_estimators=3000, #3000
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.7,
    colsample_bytree=0.7,
    objective= 'binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=99)

# retrieve params
xgb_param = clf.get_xgb_params()

'''
X_train, X_valid = train_test_split(train, test_size=test_size, random_state=0) # randomly split into 90% test and 10% CV -> still has the outcome at this point
# xgb sparse matrix
xgtrain = xgb.DMatrix(X_train[features], label= X_train[col_target])
xgvalid = xgb.DMatrix(X_valid[features], label= X_valid[col_target])
xgtest  = xgb.DMatrix(test[features])
'''

# build DMatrix internal data structure for xgboost
# IMPORTANT: use .as_matrix() to convert pandas dataframe (non-sparse representation) to numpy ndarray
xgtrain = xgb.DMatrix(train[features].as_matrix(), label=train[col_target].as_matrix())
xgtest  = xgb.DMatrix(test[features].as_matrix())

cv_folds = 5
early_stopping_rounds = 50

# cross-validation
# ------------------------------------------------------------------------------
print ('\nInitializing cross-validation...')
cvresult = xgb.cv(
    xgb_param,
    xgtrain,
    num_boost_round=clf.get_params()['n_estimators'],
    nfold=cv_folds,
    metrics='auc',
    early_stopping_rounds=early_stopping_rounds,
    verbose_eval=1)

# retrieve parameters
print ('\nXGBClassifier parameters')
clf.set_params(n_estimators=cvresult.shape[0])

# fit the algorithm on the training data
print('\nFit algorithm on train data...')
clf.fit(train[features].as_matrix(), train[col_target].as_matrix(), eval_metric='auc')

# Predict training set
# ------------------------------------------------------------------------------
print('\nPredicting on training set...')
dtrain_predictions = clf.predict(train[features].as_matrix())
dtrain_predprob = clf.predict_proba(train[features].as_matrix())[:,1]

# print model report:
print('Model Report')
print('Accuracy : %.4g' % metrics.accuracy_score(train[col_target].values, dtrain_predictions))
print('AUC Score (Train): %f' % metrics.roc_auc_score(train[col_target], dtrain_predprob))
print('IP Score  (Train): %f' % ip(train[col_target], dtrain_predprob))

print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))

# Predict test set
# ------------------------------------------------------------------------------
print('\nPredicting on test set...')
test['predprob'] = clf.predict_proba(test[features].as_matrix())[:,1]
results = test

# print model report:
print('Model Report')
print('AUC Score (Test): %f' % metrics.roc_auc_score(results[col_target], results['predprob']))
print('IP Score  (Test): %f' % ip(results[col_target], results['predprob']))

## predict out of time set
# ------------------------------------------------------------------------------
# print('\nPredicting on out of time set...')
# otime['predprob'] = clf.predict_proba(otime[predictors])[:,1]
# results = otime

## print model report:
# print('Model Report')
# print('AUC Score (OTime): %f' % metrics.roc_auc_score(results[col_target], results['predprob']))
# print('IP Score  (OTime): %f' % ip(results[col_target], results['predprob']))

# Plot importances
# ------------------------------------------------------------------------------
print('\nImportances...')
features_df = pd.DataFrame({'feature': pd.Series(features), 'importance': clf.feature_importances_})
features_df = features_df.sort_values('importance', ascending=False)
features_df = features_df.head(80) # limit the number of features to be shown in the bar plot

pyplot.bar(range(len(features_df)), features_df['importance'].values)
ind = np.arange(len(features_df['feature'].values))    # the x locations for the groups
pyplot.xticks(ind, features_df['feature'].values, rotation='vertical')
pyplot.show()


# Lime
# ------------------------------------------------------------------------------
print('\nUsing Lime to explain instances...')
import lime
import lime.lime_tabular
import re

# create the lime explainer
explainer = lime.lime_tabular.LimeTabularExplainer(X_train[features].as_matrix(), feature_names=features) # X_train.values, , class_names=(0,1)

def lime_explain_instance(id):

    test_instance_tot = test.loc[test[col_id]==id].head(1)
    test_instance = test_instance_tot[features]
    test_instance = test_instance.clip(-10000000.0, 10000000.0) # convert int to float instead?
    test_instance = test_instance.values[0]

    # prediction function: for classifiers, this should be function that takes a numpy array and outputs probability predictions
    predict_fn_xgb = lambda x: clf.predict_proba(x).astype(float)

    exp = explainer.explain_instance(test_instance, predict_fn_xgb, num_features=200) # test_instance.values
    print('Document id     : %d' % (id))
    print('Probability (=1):', clf.predict_proba([test_instance])[0,1])
    print('True class      : %s' % test_instance_tot[col_target].values[0])

    ll = []
    for i in range(1, len(exp.as_list()), 1):
        id_var = exp.as_map()[1][i][0]
        var = features[id_var]
        value = test_instance[id_var]
        crit = exp.as_list()[i][0]
        w = exp.as_list()[i][1]
        dd = {
            "variable": var,
            "value": value,
            "explanation": w,
            "criteria": crit
        }
        ll.append(dd)

    explainer_df = pd.DataFrame(ll)
    explainer_df = explainer_df.sort_values('explanation', ascending=False)
    explainer_df.head(10)
    explainer_df.tail(10)

    pyplot.bar(range(len(explainer_df)), explainer_df['explanation'].values)
    ind = np.arange(len(explainer_df['variable'].values))    # the x locations for the groups
    pyplot.xticks(ind, explainer_df['variable'].values, rotation='vertical')
    # pyplot.savefig('3_gbm_raw_feature_importance.png', bbox_inches='tight')
    pyplot.show()

    return explainer_df

# check top 15 of largest estimated probabilities
test[['id', 'TARGET', 'predprob']].sort_values('predprob', ascending=False).head(15)

"""
2016030520890380
2014120519399710
2015120012335320
2015060519288510
2015090014583910
2014120013445730
"""
explainer_df = lime_explain_instance(2016030520890380)

explainer_df.head(10)
explainer_df.tail(10)






explainer_df = lime_explain_instance(2014120519399710)
explainer_df.head(10)
explainer_df.tail(10)






explainer_df = lime_explain_instance(2015120012335320)
explainer_df.head(10)
explainer_df.tail(10)
