import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from meta_data import numeric_columns


def getColumnsWithMissingData(data=None, return_counts=False):
    # columns with missing values
    missing_values = data.isna().sum()

    # where missing values are greater than 0
    columns_with_missing_values = missing_values[missing_values > 0]

    if return_counts:
        return columns_with_missing_values

    return columns_with_missing_values.index


def removeOutliers(data=None, columns:list=[]):
    for column in columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    return data


def handleMissingValues(data=None, log=True):
    # create a deep copy
    data = data.copy(deep=True)

    # find missing data
    columns_with_missing_data = getColumnsWithMissingData(data=data, return_counts=True)

    # dropping because the missingness is way to large 
    data.drop(columns=['Loan_from', 
                       'organic_pesticide_expenditure', 
                       'food_banana_wilt_diseases',
                       'household_fertilizer',
                       ], 
            inplace=True)

    data.loc[1119, 'business_number'] = np.nan

    business_number_mode = data['business_number'].mode()
    AgricultureLand_mean = round(number=data['AgricultureLand'].mean(), ndigits=1)

    data.fillna({
        'business_number': 0,
        'AgricultureLand': AgricultureLand_mean,
    }, inplace=True)

    if log:
        print('Columns with missing data before handling\n\n'
              f'{columns_with_missing_data}')
        
    return data


def encodeCategoricals(data=None, 
                       return_columns_and_classes=False):
    data = data.copy(deep=True)
    
    label_encoder = LabelEncoder()

    # List of categorical columns to encode
    categorical_columns = data.select_dtypes(include=['object']).columns

    # Apply Label Encoding
    class_mappings = {}
    for column in categorical_columns:
        if column == 'HouseHoldID':
            continue
        data[column] = label_encoder.fit_transform(data[column])
        class_mappings[column] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    if return_columns_and_classes:
        return {'data': data,
                'columns': categorical_columns,
                'class_mappings': class_mappings
                }

    return data


# convert radios_owned to categorical
def convertRadiosOwned(data=None):
    data = data.copy(deep=True)

    data['radios_owned'] = data['radios_owned'].replace({
        2: 1, 3: 1, 21: 1, 4: 1
    })

    return data

# convert radios_owned to categorical
def convertPhonesOwned(data=None):
    data = data.copy(deep=True)

    data['phones_owned'] = data['phones_owned'].replace({
        1: 'one', 2: 'more_than_one', 3: 'more_than_one', 4: 'more_than_one', 5: 'more_than_one', 6: 'more_than_one', -99: 'none', 0: 'none'
    })

    return data

# convert business number to categorical
def convertBusinessNumber(data=None):
    data = data.copy(deep=True)

    data['business_number'] = data['business_number'].replace({
        '0': 0, '1': 1, '2': 1, '`': np.nan
    })

    return data


# Define a function to categorize the values
def categorizeStatus(value):
    if value >= 2.15:
        return "On Track"
    else:
        return "Struggling"
    
def resampleStatus(data=None):
    data = data.copy(deep=True)
    # Assuming 'data4_' is your original DataFrame
    X_imbalanced = data.drop('ProgressStatus', axis=1)
    y_imbalanced = data['ProgressStatus']

    # Apply Random Under-Sampling
    undersample = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = undersample.fit_resample(X_imbalanced, y_imbalanced)

    # Create a new DataFrame with resampled data
    resampled_data_ = pd.DataFrame(X_resampled, columns=X_imbalanced.columns)
    resampled_data_['ProgressStatus'] = y_resampled

    return resampled_data_


if __name__ == '__main__':
    # load data
    data_ = pd.read_csv('DataScientist_01_Assessment.csv')

    # Apply the function to create the new variable
    data_['ProgressStatus'] = data_['HHIncome+Consumption+Residues/Day'].apply(categorizeStatus)

    data1_ = convertRadiosOwned(data=data_)
    data2_ = convertPhonesOwned(data=data1_)
    data3_ = convertBusinessNumber(data=data2_)
    data4_ = encodeCategoricals(data=data3_, return_columns_and_classes=True)
    data5_ = handleMissingValues(data=data4_['data'])
    # data6_ = removeOutliers(data=data5_, columns=numeric_columns)
    data6_ = resampleStatus(data=data5_)
    data4_['class_mappings']


    data6_.to_csv('cleaned_data2.csv', index=False)
    data_.columns
    for index, row in data_.iterrows():
        print(row.Village)