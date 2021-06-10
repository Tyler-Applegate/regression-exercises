import pandas as pd
import numpy as np
import os
from env import host, user, password
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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

def split_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames; stratify on survived.
    return train, validate, test DataFrames.
    '''
    train_validate, test = train_test_split(df, test_size=.2, random_state=1221)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=1221)
    return train, validate, test

def wrangle_grades():
    '''
    Read student_grades csv file into a pandas DataFrame,
    drop student_id column, replace whitespaces with NaN values,
    drop any rows with Null values, convert all columns to int64,
    return cleaned student grades DataFrame.
    '''
    # Acquire data from csv file.
    grades = pd.read_csv('student_grades.csv')
    
    # Replace white space values with NaN values.
    grades = grades.replace(r'^\s*$', np.nan, regex=True)
    
    # Drop all rows with NaN values.
    df = grades.dropna()
    
    # Convert all columns to int64 data types.
    df = df.astype('int')
    
    return df


# Generic splitting function for continuous target.

def split_continuous(df):
    '''
    Takes in a df
    Returns train, validate, and test DataFrames
    '''
    # Create train_validate and test datasets
    train_validate, test = train_test_split(df, 
                                        test_size=.2, 
                                        random_state=1221)
    # Create train and validate datsets
    train, validate = train_test_split(train_validate, 
                                   test_size=.3, 
                                   random_state=1221)

    # Take a look at your split datasets

    print(f'train -> {train.shape}')
    print(f'validate -> {validate.shape}')
    print(f'test -> {test.shape}')
    return train, validate, test


################### Curriculum Functions ######################################

# Function for acquiring and prepping my student_grades df.


def wrangle_grades():
    """
    Read student_grades csv file into a pandas DataFrame,
    drop student_id column, replace whitespaces with NaN values,
    drop any rows with Null values, convert all columns to int64,
    return cleaned student grades DataFrame.
    """
    # Acquire data from csv file.
    grades = pd.read_csv("student_grades.csv")

    # Replace white space values with NaN values.
    grades = grades.replace(r"^\s*$", np.nan, regex=True)

    # Drop all rows with NaN values.
    df = grades.dropna()

    # Convert all columns to int64 data types.
    df = df.astype("int")

    return df


# Generic helper function to provide connection url for Codeup database server.


def get_db_url(db_name):
    """
    This function uses my env file to get the url to access the Codeup database.
    It takes in a string identifying the database I want to connect to.
    """
    return f"mysql+pymysql://{user}:{password}@{host}/{db_name}"


# Generic function that takes in a database name and a query.


def get_data_from_sql(str_db_name, query):
    """
    This function takes in a string for the name of the database I want to connect to
    and a query to obtain my data from the Codeup server and return a DataFrame.
    """
    df = pd.read_sql(query, get_db_url(str_db_name))
    return df


# Mother function to acquire and prepare Telco data.


def wrangle_telco():
    """
    Queries the telco_churn database
    Returns a clean df with four columns:
    customer_id(object), monthly_charges(float), tenure(int), total_charges(float)
    """
    query = """
            SELECT
                customer_id,
                monthly_charges,
                tenure,
                total_charges
            FROM customers
            JOIN contract_types USING(contract_type_id)
            WHERE contract_type = 'Two year';
            """
    df = get_data_from_sql("telco_churn", query)

    # Replace any tenures of 0 with 1
    df.tenure = df.tenure.replace(0, 1)

    # Replace the blank total_charges with the monthly_charge for tenure == 1
    df.total_charges = np.where(
        df.total_charges == " ", df.monthly_charges, df.total_charges
    )

    # Convert total_charges to a float.
    df.total_charges = df.total_charges.astype(float)

    return df


# Generic splitting function for continuous target.


def split_continuous(df):
    """
    Takes in a df
    Returns train, validate, and test DataFrames
    """
    # Create train_validate and test datasets
    train_validate, test = train_test_split(df, test_size=0.2, random_state=123)
    # Create train and validate datsets
    train, validate = train_test_split(train_validate, test_size=0.3, random_state=123)

    # Take a look at your split datasets

    print(f"train -> {train.shape}")
    print(f"validate -> {validate.shape}")
    print(f"test -> {test.shape}")
    return train, validate, test


def train_validate_test(df, target):
    """
    this function takes in a dataframe and splits it into 3 samples,
    a test, which is 20% of the entire dataframe,
    a validate, which is 24% of the entire dataframe,
    and a train, which is 56% of the entire dataframe.
    It then splits each of the 3 samples into a dataframe with independent variables
    and a series with the dependent, or target variable.
    The function returns 3 dataframes and 3 series:
    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test.
    """
    # split df into test (20%) and train_validate (80%)
    train_validate, test = train_test_split(df, test_size=0.2, random_state=123)

    # split train_validate off into train (70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=0.3, random_state=123)

    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=[target])
    y_train = train[target]

    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]

    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=[target])
    y_test = test[target]

    return X_train, y_train, X_validate, y_validate, X_test, y_test


def get_numeric_X_cols(X_train, object_cols):
    """
    takes in a dataframe and list of object column names
    and returns a list of all other columns names, the non-objects.
    """
    numeric_cols = [col for col in X_train.columns.values if col not in object_cols]

    return numeric_cols


def min_max_scale(X_train, X_validate, X_test, numeric_cols):
    """
    this function takes in 3 dataframes with the same columns,
    a list of numeric column names (because the scaler can only work with numeric columns),
    and fits a min-max scaler to the first dataframe and transforms all
    3 dataframes using that scaler.
    it returns 3 dataframes with the same column names and scaled values.
    """
    # create the scaler object and fit it to X_train (i.e. identify min and max)
    # if copy = false, inplace row normalization happens and avoids a copy (if the input is already a numpy array).

    scaler = MinMaxScaler(copy=True).fit(X_train[numeric_cols])

    # scale X_train, X_validate, X_test using the mins and maxes stored in the scaler derived from X_train.
    #
    X_train_scaled_array = scaler.transform(X_train[numeric_cols])
    X_validate_scaled_array = scaler.transform(X_validate[numeric_cols])
    X_test_scaled_array = scaler.transform(X_test[numeric_cols])

    # convert arrays to dataframes
    X_train_scaled = pd.DataFrame(X_train_scaled_array, columns=numeric_cols).set_index(
        [X_train.index.values]
    )

    X_validate_scaled = pd.DataFrame(
        X_validate_scaled_array, columns=numeric_cols
    ).set_index([X_validate.index.values])

    X_test_scaled = pd.DataFrame(X_test_scaled_array, columns=numeric_cols).set_index(
        [X_test.index.values]
    )

    return X_train_scaled, X_validate_scaled, X_test_scaled


def get_object_cols(df):
    """
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names.
    """
    # create a mask of columns whether they are object type or not
    mask = np.array(df.dtypes == "object")

    # get a list of the column names that are objects (from the mask)
    object_cols = df.iloc[:, mask].columns.tolist()

    return object_cols


def create_dummies(df, object_cols):
    """
    This function takes in a dataframe and list of object column names,
    and creates dummy variables of each of those columns.
    It then appends the dummy variables to the original dataframe.
    It returns the original df with the appended dummy variables.
    """

    # run pd.get_dummies() to create dummy vars for the object columns.
    # we will drop the column representing the first unique value of each variable
    # we will opt to not create na columns for each variable with missing values
    # (all missing values have been removed.)
    dummy_df = pd.get_dummies(object_cols, dummy_na=False, drop_first=True)

    # concatenate the dataframe with dummies to our original dataframe
    # via column (axis=1)
    df = pd.concat([df, dummy_df], axis=1)

    return df


### student_mat.csv for feature engineering lesson
def wrangle_student_math(path):
    df = pd.read_csv(path, sep=";")

    # drop any nulls
    df = df[~df.isnull()]

    # get object column names
    object_cols = get_object_cols(df)

    # create dummy vars
    df = create_dummies(df, object_cols)

    # split data
    X_train, y_train, X_validate, y_validate, X_test, y_test = train_validate_test(
        df, "G3"
    )

    # get numeric column names
    numeric_cols = get_numeric_X_cols(X_train, object_cols)

    # scale data
    X_train_scaled, X_validate_scaled, X_test_scaled = min_max_scale(
        X_train, X_validate, X_test, numeric_cols
    )

    return (
        df,
        X_train,
        X_train_scaled,
        y_train,
        X_validate_scaled,
        y_validate,
        X_test_scaled,
        y_test,
    )