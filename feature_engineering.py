
from sklearn import preprocessing
import pandas as pd
import numpy as np

def get_features(train, test):
    """Get common features between subsets"""
    train_val   = list(train.columns.values)
    test_val    = list(test.columns.values)

    total_val = list(set(train_val) & set(test_val))
    return total_val


def process_features(train, test, scale=False, remove_missing=False, fill_na=True, per_missing=0.5, target='TARGET'):
    """Process features"""
    train_target = train[target]
    test_target  = test[target]
    tables       = pd.concat([train.drop(target, axis=1), test.drop(target, axis=1)], axis=0)
    train_idx    = range(0, train.shape[0])
    test_idx     = range(train.shape[0], tables.shape[0])

    print ('\nProcess features')
    print ('\nTrain Min: {}, Train Max: {}'.format(min(train_idx),max(train_idx)))
    print ('Test Min: {}, Test Max: {}'.format(min(test_idx),max(test_idx)))
    print ('Initial shape of the tables...')
    print (' -Train data set: {}'.format(train.shape))
    print (' -Test data set: {}'.format(test.shape))
    print (' -Total data set to handle: {}'.format(tables.shape))

    #Getting numerical and categorical variables
    numerical_features = tables.select_dtypes(include=["float","int","bool"]).columns.values
    # categorical_features=tables.select_dtypes(include=["object"]).columns.values

    #Scaling the values in case it is needed
    #---------------------------------------------------------------------------------------
    if scale:
        print ('\nScaling values values...')
        for table in tables:
            for feature in numerical_features:
                if feature in table.columns.values:
                    table[feature] = (table[feature] - table[feature].mean()) / (table[feature].max() - table[feature].min()) #preprocessing.scale(table[feature])
                    assert(table[feature].mean()<0.0001)
                    assert(table[feature].mean()<0.0001)
        print ('Scaled.')
        print (' -Train data set: {}'.format(train.shape))
        print (' -Test data set: {}'.format(test.shape))

    #Deteleting features with more than 'per_missing' % of missing values
    #---------------------------------------------------------------------------------------
    if remove_missing:
        print ('\nHandling missing values...')
        total_missing = train.isnull().sum()
        to_delete = total_missing[total_missing > len(train)*per_missing] # select features with more than 1/3 missing values
        for table in tables:
            table.drop(to_delete.index.tolist(), axis=1, inplace=True)
        print ('Removed features with {} of missing data!'.format(per_missing))
        print (' -Train data set: {}'.format(train.shape))
        print (' -Test data set: {}'.format(test.shape))

    #Filling NAs with median and most common
    #---------------------------------------------------------------------------------------
    if fill_na:
        print ('\nFilling NaN...')
        total_missing = tables.isnull().sum()
        print ('Total missing values: {}'.format(total_missing.sum()))
        numerical_features      = tables.select_dtypes(include=["float","int","bool"]).columns.values
        # categorical_features    = tables.select_dtypes(include=["object"]).columns.values
        i = 1
        t = 1

        for table in [tables]:
            print ('\nTable {}:'.format(t))
            for feature in numerical_features:
                pre_fill = table[feature].mean()
                # table[feature].fillna(0, inplace = True) # replace by 0 value
                table[feature].fillna(train[feature].median(), inplace = True) # replace by median value
                post_fill = table[feature].mean()
                if pre_fill != post_fill:
                    print ('{}. Feature: {}, Median before filling: {} vs. after filling {}'.format(i,feature,pre_fill,post_fill))

            # for feature in categorical_features:
            #     table[feature].fillna('None', inplace = True) # replace by most frequent value
            #     if feature in train.columns.values:
            #         table[feature + '_mean'] = train[target].groupby(table[feature]).transform('mean') #train.groupby(feature)[target].mean()
            #         table[feature + '_mean'] = np.log(table[feature + '_mean'])

            t += 1
        total_missing = train.isnull().sum()
        # print (table.head(5))
        print ('\nTotal missing values: {}'.format(total_missing.sum()))
        print ('NaNs filled with median value or major categorical value.')
        print (' -Common data set: {}'.format(table.shape))

    #---------------------------------------------------------------------------------------
    print ('\nGetting common features between...')
    train    = tables.ix[train_idx]
    test     = tables.ix[test_idx]
    features = get_features(train, test)
    train    = pd.concat([train, train_target], axis=1)
    test     = pd.concat([test, test_target], axis=1)

    print ('Final shape of the training and testing shapes...')
    print (' -Train data set: {}'.format(train[features].shape))
    print (' -Test data set: {}'.format(test[features].shape))

    return train, test, features
