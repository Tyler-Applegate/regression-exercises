import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def sse_comp(df, variables, target):
    '''
    This function will take in a pandas DataFrame, and predefiend list of variables to
    apply the LinearRegression Model to, as well as a predefined target variable.
    This function creates and fits a LinearRegression model, creates a 'yhat' or prediction column,
    calculates model residuals. It also creates a baseline model, and calculates baseline
    residuals. The Sum of Squared Errors are calculated for the baseline, and the model.
    Both are printed out, as well as the improvment in SSE over baseline
    '''
    # create and fit our LinearRegression model
    model = LinearRegression().fit(df[variables], df[target])
    # create a yhat/prediction column
    df['yhat'] = model.predict(df[variables])
    # create the residuals column by subtracting predicted tip from actual tip
    df['residuals'] = df[target] - df['yhat']
    # sum of squared errors for model
    sse = (df['residuals'] ** 2).sum()
    # create a baseline model
    df['yhat_baseline'] = df[target].mean()
    # baseline residuals
    df['baseline_residuals'] = df[target] - df['yhat_baseline']
    # baseline sse
    sse_baseline = (df['baseline_residuals'] ** 2).sum()
    print(f'Baseline Model SSE:          {sse_baseline:.2f}')
    print(f'Linear Regression Model SSE: {sse:.2f}')
    print(f'Improvement over baseline:   {sse_baseline-sse:.2f}')

def plot_residuals(df, variables, target):
    '''
        This function will take in a pandas DataFrame, and predefiend list of variables to
    apply the LinearRegression Model to, as well as a predefined target variable.
    This function creates and fits a LinearRegression model, creates a 'yhat' or prediction column,
    calculates model residuals. It also creates a baseline model, and calculates baseline
    residuals. These are then plotted in a histogram.
    '''
     # create and fit our LinearRegression model
    model = LinearRegression().fit(df[variables], df[target])
    # create a yhat/prediction column
    df['yhat'] = model.predict(df[variables])
    # create the residuals column by subtracting predicted tip from actual tip
    df['residuals'] = df[target] - df['yhat']
    # create a baseline model
    df['yhat_baseline'] = df[target].mean()
    # baseline residuals
    df['baseline_residuals'] = df[target] - df['yhat_baseline']
    fig, ax = plt.subplots(figsize=(13,7))
    ax.hist(df['baseline_residuals'], label='baseline residuals', alpha=.6)
    ax.hist(df['residuals'], label='model residuals', alpha=.6)
    ax.legend()
    return plt.show()

def regression_errors(df, variables, target):
    '''
    This function will take in a pandas DataFrame, and predefiend list of variables to
    apply the LinearRegression Model to, as well as a predefined target variable.
    This function creates and fits a LinearRegression model, creates a 'yhat' or prediction column,
    calculates model residuals. It will then calculate and print out the SSE,
    ESS, TSS, MSE, and RMSE of the model.
    '''
     # create and fit our LinearRegression model
    model = LinearRegression().fit(df[variables], df[target])
    # create a yhat/prediction column
    df['yhat'] = model.predict(df[variables])
    # create the residuals column by subtracting predicted tip from actual tip
    df['residuals'] = df[target] - df['yhat']
    # set an n value for number of observations
    n = df.shape[0]
    # sum of squared errors
    sse = (df['residuals'] ** 2).sum()
    # explained sum of squares
    ess = ((df['yhat'] - df['tip'].mean())**2).sum()
    # total sum of squares
    tss = ((df['tip'] - df['tip'].mean())**2).sum()
    # mean squared error
    mse = sse / n
    # root mean squared error
    rmse = mse ** .5
    
    print(f'    Sum of Square Errors (SSE): {sse:.2f}')
    print(f'Explained Sum of Squares (ESS): {ess:.2f}')
    print(f'    Total Sum of Squares (TSS): {tss:.2f}')
    print(f'      Mean Squared Error (MSE): {mse:.2f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')

def baseline_mean_errors(df, target):
    '''
    Takes in a predefined target variable, creates yhat_baseline,
    and baseline_residuals. Then calculates and prints out SSE, MSE, and RMSE
    for the baseline.
    '''
    # create a baseline model
    df['yhat_baseline'] = df[target].mean()
    # baseline residuals
    df['baseline_residuals'] = df[target] - df['yhat_baseline']
    # set an n value for number of observations
    n = df.shape[0]
    # baseline sum of squared errors
    sse_baseline = (df['baseline_residuals'] ** 2).sum()
    # baseline mean squared error
    mse_baseline = sse_baseline / n
    # baseline root mean squared error     
    rmse_baseline = mse_baseline ** .5
    
    print(f'   Baseline Sum of Squared Errors (SSE): {sse_baseline:.2f}')
    print(f'      Baseline Mean Squared Error (MSE): {mse_baseline:.2f}')
    print(f'Baseline Root Mean Squared Error (RMSE): {rmse_baseline:.2f}')