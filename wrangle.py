import pandas as pd
import numpy as np
import os
from env import host, user, password

# sets up a secure connection to the Codeup db using my login info

def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my env file to create a connection url to access
    the Codeup database.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


# This function will connect with the Codeup database select key features in the telco db
# to return a pandas DataFrame
def new_telco_regression():
    '''
    This function connects to the telco_df in the Codeup database, and returns the 'customer_id',
    'monthly_charges', 'tenure', and 'total_charges' from the telco_churn database for all customers
    with a 2 year contract
    '''
    sql_query = '''SELECT customer_id, monthly_charges, tenure, total_charges
                    FROM customers
                    WHERE contract_type_id LIKE 3;
                '''
    return pd.read_sql(sql_query, get_connection('telco_churn'))

# This function plays on top of new_telco_regression by 1st looking to see if there is a .csv of the telco Dataframe, and
# creating one if there is not. This optomizes performance/runtime, by only needing to connect to the server 1 time
# and then using a local .csv thereafter
def get_telco_regression():
    '''
    This function reads in Telco data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('telco_regression.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('telco_regression.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_telco_regression()
        
        # Cache data
        df.to_csv('telco_regression.csv')

    return df

def wrangle_telco():
    '''
    This function uses my login credentials to connect to the telco_churn dataset on 
    Codeup database. It uses a SQL query to return the 'customer_id', 'monthly_charges', 
    'tenure', and 'total_charges' of all customers with a 2 year contract. It also checks
    to see if there is an existing csv file, and if not, writes the database to a csv, to
    optimize speed.
    '''
    df = get_telco_regression()
    df = df.replace(r'^\s*$', np.nan, regex=True)
    df = df.dropna()
    df['total_charges'] = df['total_charges'].astype(float)
    
    return df

# This function will connect with the Codeup database select key features in the zillow db
# to return a pandas DataFrame
def new_zillow_data():
    '''
    This function connects to the Zillow df in the Codeup database, and returns the 'bedroomcnt',
    'bathroomcnt', 'calculatedfinishedsquarefeet', 'taxvaluedollarcnt', 'yearbuilt', 'taxamount',
    and 'fips' from the zillow database for all 2017 customers with a single family residential
    '''
    sql_query = '''SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
                    FROM properties_2017
                    WHERE propertylandusetypeid IN (261);
                '''
    return pd.read_sql(sql_query, get_connection('zillow'))

# This function plays on top of new_zillow_data by 1st looking to see if there is a .csv of the telco Dataframe, and
# creating one if there is not. This optomizes performance/runtime, by only needing to connect to the server 1 time
# and then using a local .csv thereafter
def get_zillow_data():
    '''
    This function reads in zillow data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow_2017.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('zillow_2017.csv')
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_zillow_data()
        
        # Cache data
        df.to_csv('zillow_2017.csv')

    return df    