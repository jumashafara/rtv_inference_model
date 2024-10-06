import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from meta_data import best_categoric_columns
from meta_data import best_numeric_columns
from meta_data import columns_to_work

def splitData(data=None, 
               target_column:str=None, 
               test_size:float=0.2, 
               random_state:int=None):

    # Features (X) and Target (y)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return {
        'X_train': X_train, 
        'X_test': X_test, 
        'y_train': y_train, 
        'y_test': y_test
    }

def trainSingleModel(model=None, split_data:dict=None):
    model = model.fit(X=split_data['X_train'], y=split_data['y_train'])
    return model 

def buildPipeline():
    # Preprocessing steps
    numeric_transformer = StandardScaler() # to reduce outlier impact
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Combine preprocessing steps
    column_transformer = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, best_numeric_columns),
            ('categorical', categorical_transformer, best_categoric_columns)
        ])
    
    # Pipeline
    pipe = Pipeline([
        ('column_transformer', column_transformer),
        ('model', RandomForestClassifier())
    ])

    return pipe

def buildGridSearchCV():
    pipe = buildPipeline()

    # Define the models and their hyperparameters
    param_grid = [
        {
            'model': [RandomForestClassifier()],
            'model__n_estimators': [50, 100, 200, 300, 400, 500],
            'model__max_depth': [None, 10, 20, 30, 40, 50],
            'model__min_samples_split': [2, 5, 7, 11, 13, 17],
            'model__min_samples_leaf': [1, 2, 3, 4, 5, 6]
        }
    ]

    # Initialize GridSearchCV
    grid_search = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1, verbose=1)

    return grid_search


if __name__ == '__main__':
    data_ = pd.read_csv('cleaned_data2.csv')

    split_data_ = splitData(data=data_, target_column='ProgressStatus')

    model = trainSingleModel(model=buildGridSearchCV(), split_data=split_data_)
    model.best_params_

    y_pred = model.predict(split_data_['X_test'])
    classification_report_ = pd.DataFrame(classification_report(y_true=split_data_['y_test'], 
                                                y_pred=y_pred, 
                                                output_dict=True))
    classification_report_

    # save model
    joblib.dump(value=model, filename='output_model.joblib')

    # load model
    loaded_model = joblib.load(filename='output_model.joblib')
    loaded_model.predict(data_.drop('ProgressStatus', axis=1))

    print(best_numeric_columns)
    print(best_categoric_columns)