# -*- coding: utf-8 -*-
"""
Entry for Kaggle Competition: San Fransisco Crime Classification
Author: Tobias Dawson
July 2020
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.model_selection import cross_validate, train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.dummy import DummyClassifier
import re
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
def truncate(string):
    string = string.lower()
    index = string.find('block')
    if index!=-1:
        return string[index+8:]
    else:
        return string
def choose_first(string):
    string=string.lower()
    index = string.find('/')
    if index!=-1:
        return string[:index]
    else:
        return string
def prepare_data():
    df_train = pd.read_csv('/kaggle/input/sf-crime/train.csv.zip')#.iloc[:100000]
    df_test = pd.read_csv('/kaggle/input/sf-crime/test.csv.zip')#.iloc[:100000]
    #get full feature_list
    features = pd.unique(df_train['Category'])
    #get rid of classes which occur less than 3 times: they ruin cross validation
    offence_counts = df_train['Category'].value_counts()
    rare_offences = offence_counts[offence_counts<3].index
    for offence in rare_offences:
        df_train = df_train[df_train['Category']!=offence]
    y = df_train['Category']
    #create empty training dataframe
    X = pd.DataFrame([])
    X_test = pd.DataFrame([])
    
    #standardise addresses
    df_train['Address'] = df_train['Address'].apply(truncate).apply(choose_first)
    df_test['Address'] = df_test['Address'].apply(truncate).apply(choose_first)

    #find common/predictive streets within address
    streets = []
    value_counts = df_train['Address'].value_counts()
    value_counts = value_counts[value_counts>10].index.tolist() #potential improvement: use WoE to select values better
    streets += value_counts[:300] #choose streets where most crimes are committed
    temp = df_train.set_index('Address').loc[value_counts]['Category']
    for i in pd.unique(temp.index):
        valc = temp.loc[i].value_counts() #get ordered list of the types of crime committed in each street
        if valc.iloc[0]/valc.sum() > 0.4: #also choose streets with higher than avg predictive power
            streets += [i]
    #loop to feature-engineer test data in same manner as train data
    df_list = [(X, df_train), (X_test, df_test)]
    count = 0
    for pair in df_list:
        new_df, old_df = pair[0], pair[1]
        #add Month, Hour, Minute
        new_df['Month'] = pd.to_datetime(old_df['Dates']).apply(lambda x: np.sin(np.pi*x.month/12)) #use sin to put December next to January
        new_df['Hour'] = pd.to_datetime(old_df['Dates']).apply(lambda x: np.sin(np.pi*x.hour/24)) #same for hour
        new_df['Minute'] = pd.to_datetime(old_df['Dates']).apply(lambda x: np.sin(np.pi*x.minute/60)) #same for minutes
        #rescale Month, Hour and Minute
        #scaler = MinMaxScaler()
        #new_df[['Month','Hour','Minute']] = scaler.fit_transform(new_df[['Month', 'Hour', 'Minute']])
        
        #rescale X & Y
        xyscaler = StandardScaler(copy=True).fit(old_df[['X','Y']])
        #create empty columns (necessary in this xase)
        new_df['X'] = pd.Series([], dtype=float)
        new_df['Y'] = pd.Series([], dtype=float)
        new_df[['X', 'Y']] = xyscaler.fit_transform(old_df[['X','Y']])
        new_df['r'] = new_df.apply(lambda x: np.sqrt(x['X']**2 + x['Y']**2), axis=1) #transform to polar coords
        new_df['theta'] = new_df.apply(lambda x: np.arctan(x['Y']/x['X'])/(np.pi),axis=1)
        new_df['r'] = new_df.apply(lambda x: x/new_df['r'].max(), axis=1) #normalise r
        #standardise addresses
        old_df['Address'] = old_df['Address'].apply(truncate).apply(choose_first)
        
        #delete all streets except the ones we want to one-hot encode
        old_df['Address'] = old_df['Address'].apply(lambda street: street if street in streets else float('NaN'))
        #one-hot encode streets
        new_df = pd.concat([new_df, 
                       pd.get_dummies(old_df['Address'], prefix='street',columns = streets)], 
                       axis=1)
        
        #One-hot encode Day of week
        new_df = pd.concat([new_df, pd.get_dummies(old_df['DayOfWeek'], prefix='day')], axis=1)
        #use PdDistrict as categorical feature, since there are only ten categories
        #and they are the same in train and test
        new_df = pd.concat([new_df, pd.get_dummies(old_df['PdDistrict'], prefix='district')], axis=1)
        if count==0:
            X = new_df
        elif count==1:
            X_test = new_df
        else:
            print('Houston, we have a problem!')
        count+=1
    return X, y, X_test, features


def run_classifiers(X, y):
    #dummy to compare
    #dummy_clf = DummyClassifier(strategy = 'stratified')
    #print('dummy scores:')
    #print(cross_val_score(dummy_clf, X, y, cv=3))
    
    #X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
    #clf = tree.DecisionTreeClassifier()
    #param_grid = {'max_depth': [18,21,24,27,31,33], 'min_samples_split': [200,250,300,350]}
    #GridSearch = GridSearchCV(clf, param_grid, scoring = 'neg_log_loss',cv=3)
    #GridSearch.fit(X, y)
    #print('Decision Tree:')
    #print(GridSearch.cv_results_['mean_test_score'])
    
    clf = RandomForestClassifier(max_depth=18,min_samples_split=400)
    print('starting fit')
    clf.fit(X,y)
    print('fit done')
    #param_grid = {'max_depth': [18], 'min_samples_split': [200,400,600]}
    #GridSearch = GridSearchCV(clf, param_grid, scoring = 'neg_log_loss',cv=3)
    #GridSearch.fit(X, y)
    #print('Random Forest:')
    #print(GridSearch.cv_results_['mean_test_score'])
    """
    min_samples_splits = [200, 250, 300]
    max_depths = [30,32]
    for min_samples_split in min_samples_splits:
        for max_depth in max_depths:
            clf = tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
            score = cross_validate(clf, X, y, cv=3, scoring = 'neg_log_loss')
            #clf.fit(X_train, y_train)
            #y_train_pred = clf.predict(X_train)
            #train_score = accuracy_score(y_train, y_train_pred)
            #y_pred = clf.predict(X_val)
            #test_score = accuracy_score(y_val, y_pred)
            print('min samples split=', min_samples_split)
            print('max depth:', max_depth)
            print('score =', score)
            #print('train score:')
            #print(train_score)
            #print('test score:')
            #print(test_score)
            print('\n')
    """
    return(clf)
def create_csv(X_test, clf, features):
    y_proba = clf.predict_proba(X_test)
    df = pd.DataFrame(y_proba, columns = clf.classes_)
    df.to_csv('result6.csv', index=True, header=True, index_label = 'Id')
    
X, y, X_test, features = prepare_data()
X_features = set(X.columns)
X_test_features = set(X_test.columns)
common_features = X_features.intersection(X_test_features)
X, X_test = X[common_features], X_test[common_features]
clf = run_classifiers(X=X, y=y)

create_csv(X_test,clf, features)
starting fit
fit done
clf2.fit(X, y)
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-2-89a61e12f444> in <module>
----> 1 clf2.fit(X, y)

NameError: name 'clf2' is not defined
X_test