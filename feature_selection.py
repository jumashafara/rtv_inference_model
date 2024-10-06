import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from meta_data import categorical_columns
from meta_data import numeric_columns
from preprocessing import removeOutliers
from meta_data import columns_to_work

def getBestCategoricalFeatures(data=None,
                               k:int=None,
                               return_scores=True,
                               return_indicies=True) -> dict: 
    data = data[categorical_columns].copy(deep=True)
    test = SelectKBest(score_func=chi2, k=k)
    fit = test.fit(data.drop('ProgressStatus', axis=1), data['ProgressStatus'])
    scores = fit.scores_.astype(int)
    # features = fit.transform(data.drop('ProgressStatus', axis=1))
    selected_indices = fit.get_support(indices=True)

    data = data.iloc[:, selected_indices]

    results = {'data': data}

    if return_scores:
        results['scores'] = scores
    if return_indicies: 
        results['indicies'] = selected_indices

    return results


def getBestNumericFeatures(data=None, 
                           k:int=None,
                           return_scores=True,
                           return_indicies=True) -> dict:
    # create a test object from SelectKBest
    test = SelectKBest(score_func=f_classif, k=k)

    # fit the test object to the data
    fit = test.fit(data.drop('ProgressStatus', axis=1), data['ProgressStatus'])

    # get the scores and features
    scores = fit.scores_.astype(int)

    # get the selected indices
    features = fit.transform(data.drop('ProgressStatus', axis=1))
    selected_indices = test.get_support(indices=True)

    data = data.iloc[:, selected_indices]

    results = {'data': data}

    if return_scores:
        results['scores'] = scores
    if return_indicies: 
        results['indicies'] = selected_indices

    return results


if __name__ == '__main__':
    data_ = pd.read_csv('cleaned_data.csv')
    data_[columns_to_work]

    categoric_results = getBestCategoricalFeatures(data=data_, k=7)
    numeric_result = getBestNumericFeatures(data=data_, k=13)


    numeric_result['data'].columns

    best_feature_set = pd.concat([numeric_result['data'], categoric_results['data'], data_['ProgressStatus']], axis=1)


    best_feature_set.to_csv('../best_dataset.csv', index=False)

    print(categoric_results.columns)