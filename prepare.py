
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


###################### Prep Iris Data ######################

def prep_iris(df):
    '''
    This function takes in the iris df acquired by get_iris_data
    Returns the iris df with dummy variables encoding species.
    '''
    # drop and rename columns
    df = df.drop(columns='species_id').rename(columns={'species_name': 'species'})
    
    # create dummy columns for species
    species_dummies = pd.get_dummies(df.species, drop_first=True)
    
    # add dummy columns to df
    df = pd.concat([df, species_dummies], axis=1)
    
    return df

###################### Prep Titanic Data ######################

def titanic_split(df):
    '''
    This function take in the titanic data acquired by get_titanic_data,
    performs a split and stratifies survived column.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123, 
                                        stratify=df.survived)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123, 
                                   stratify=train_validate.survived)
    return train, validate, test



def impute_mean_age(train, validate, test):
    '''
    This function imputes the mean of the age column for
    observations with missing values.
    Returns transformed train, validate, and test df.
    '''
    # create the imputer object with mean strategy
    imputer = SimpleImputer(strategy = 'mean')
    
    # fit on and transform age column in train
    train['age'] = imputer.fit_transform(train[['age']])
    
    # transform age column in validate
    validate['age'] = imputer.transform(validate[['age']])
    
    # transform age column in test
    test['age'] = imputer.transform(test[['age']])
    
    return train, validate, test


def prep_titanic(df):
    '''
    This function take in the titanic data acquired by get_titanic_data,
    Returns prepped train, validate, and test dfs with embarked dummy vars,
    deck dropped, and the mean of age imputed for Null values.
    '''
    
    # drop rows where embarked/embark town are null values
    df = df[~df.embarked.isnull()]
    
    # encode embarked using dummy columns
    titanic_dummies = pd.get_dummies(df.embarked, drop_first=True)
    
    # join dummy columns back to df
    df = pd.concat([df, titanic_dummies], axis=1)
    
    # drop the deck column
    df = df.drop(columns='deck')
    
    # split data into train, validate, test dfs
    train, validate, test = titanic_split(df)
    
    # impute mean of age into null values in age column
    train, validate, test = impute_mean_age(train, validate, test)
    
    return train, validate, test

    ##########################################################
# ---------------- Prepare Titanic Data ---------------------- #

def impute_mode(df):
    '''
    impute mode for embark_town. Replaces missing values with most frequently occurring value.
    '''
    imputer = SimpleImputer(strategy='most_frequent', missing_values=None)
    df[['embark_town']] = imputer.fit_transform(df[['embark_town']])
    return df

def prep_titanic_data(df):
    '''
    takes in a dataframe of the titanic dataset as it is acquired and returns a cleaned dataframe
    argument: df: a pandas df with expected feature names and columns
    return: train, test, split: three dataframes with the cleanining operations performed on them
    '''
    #drop duplicates
    df = df.drop_duplicates()
    # drop cols we dont need in our model
    df = df.drop(columns= ['deck', 'embarked', 'class', 'age', 'passenger_id'])
    # replace missing values using imputer
    df = impute_mode(df)
    # divide df into a training and test dataset
    train, test = train_test_split(df, test_size= 0.2, random_state= 1349, stratify = df.survived)
    # divide train dataset into train and validate
    train, validate = train_test_split(train, train_size= 0.7, random_state=1349, stratify = train.survived)
    # change object dtype cols to integers
    dummy_train = pd.get_dummies(train[['sex','embark_town']], drop_first=[True, True])
    dummy_validate = pd.get_dummies(validate[['sex','embark_town']], drop_first=[True, True])
    dummy_test = pd.get_dummies(test[['sex','embark_town']], drop_first=[True, True])
    # merge each dummy df with appropriate dataset
    train = pd.concat([train, dummy_train], axis = 1)
    validate = pd.concat([validate, dummy_validate], axis = 1)
    test = pd.concat([test, dummy_test], axis = 1)
    # drop columns that have been converted to integers via dummy dfs
    train = train.drop(columns = ['sex','embark_town'])
    validate = validate.drop(columns = ['sex','embark_town'])
    test = test.drop(columns = ['sex','embark_town'])
    # return the three datasets
    return train, validate, test

