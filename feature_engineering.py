# imports
import numpy as np
import pandas as pd
from pydataset import data
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import RFE

def select_kbest(X, y, k):
    '''
    This function takes in the predictors (X), the target (y), and the number 
    of features to select (k) and returns the names of the top k selected 
    features based on the SelectKBest class. It requires the X, y, and k be 
    predefined.
    '''
    # setting up my select k best
    f_selector = SelectKBest(score_func=f_regression, k=k)
    # fitting to X and y
    f_selector.fit(X, y)
    # putting a mask on to see which columns are selected 
    mask = f_selector.get_support()
    
    return X.columns[mask]

def rfe(X, y, k):
    '''
    This function takes in the predictors, the target, and the number 
    of features to select. It should return the top k features based 
    on the RFE class. It requires the X, y, and k be predefined.
    '''
    # build an LR model and fit to X and y
    model = LinearRegression().fit(X, y)
    # use the RFE function to select k best features
    lm = LinearRegression()
    rec = RFE(estimator=lm, n_features_to_select=k)
    # fit this bad boy to X and y 
    rec.fit(X, y)
    
    return X.columns[rec.support_]