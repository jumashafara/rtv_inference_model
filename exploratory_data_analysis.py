import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from preprocessing import removeOutliers
from scipy.stats import chi2_contingency
from meta_data import best_numeric_columns
from meta_data import best_categoric_columns
import statsmodels.api as sm
from statsmodels.formula.api import ols

# load dataset
data_ = pd.read_csv('best_dataset.csv')
data_.columns

def createStackedBars(data=None, 
                      columns:list=[], 
                      target:str=''):
    for column in columns:
        # Create a crosstab for each categorical column against ProgressStatus
        crosstab = pd.crosstab(data[column], data[target])
        
        # Plot a stacked bar plot
        crosstab.plot(kind='bar', stacked=True)
        plt.title(f'{column} vs ProgressStatus')
        plt.show()


def performChi2Test(data=None,
                    columns:list=[],
                    target:str=''):
    # Perform Chi-Square test of independence for each categorical column
    for column in columns:
        # Create a crosstab for the categorical column and ProgressStatus
        crosstab = pd.crosstab(data[column], data[target])
        
        # Perform Chi-Square test
        chi2, p, dof, expected = chi2_contingency(crosstab)
        
        print(f"Chi-Square Test for {column} vs ProgressStatus:")
        print(f"Chi2 Statistic: {chi2}")
        print(f"P-Value: {p}\n")


def createBoxPlots(data=None, columns:list=[], target:str=''):
    for column in columns:
        data_swept = removeOutliers(data=data, column=column) 
        sns.boxplot(x=target, y=column, data=data_swept)
        plt.title(f'{column} vs ProgressStatus')
        plt.show()


def performANOVA(data=None, columns:list=[], target:str=''):

    # Perform ANOVA for each numeric variable against ProgressStatus
    for column in columns:
        model = ols(f'{column} ~ C({target})', data=data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        print(f"ANOVA result for {column} vs ProgressStatus:")
        print(anova_table)
        print("\n")



if __name__ == '__main__':
    
    createStackedBars(data=data_, columns=['composts'], target='ProgressStatus')
    data_[data_['composts'] == 1 ]['ProgressStatus'].value_counts()
    
    performChi2Test(data=data_, columns=[], target='ProgressStatus')

    performANOVA(data=data_, columns=['FormalEmployment'], target='ProgressStatus')

    createBoxPlots(data=data_, columns=['TimeToOPD'], target='ProgressStatus')
