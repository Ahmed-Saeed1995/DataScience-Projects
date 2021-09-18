import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression

def preprocessing(file):
    """Cleaning data and spoit features-HI Ranks"""
    df = pd.read_csv(file, encoding='latin-1')
    #Deleting unncessarly columns
    df.drop(["Coverage", "Country"], inplace= True, axis = 1)
    #fill missing values with mean
    for i in df.columns:
        df[i].fillna(df[i].mean() , inplace = True)
    #split features-HDI Ranks
    X = df.iloc[:,1:]
    y = df.iloc[:,:1]
    #return DataFrame to view and X,y splited
    return df, X, y

def read_file(df):
    #To review current CSV file
    print(df.head())

def cross_validation(model, X, y):
    #using cross validation instead train_test_split
    cross_valid = cross_validate(reg, X, y, cv= 10, return_train_score =True)
    return cross_valid


if __name__ == '__main__':
    #call funftions
    dataframe, X, y = preprocessing("HDI.csv")
    #Instance of model
    reg = LinearRegression()
    #Call cross validation fill with model reg and features ->X, HDI Rank ->y
    cross_model = cross_validation(reg, X, y)
    #Scores of train-test mean
    print("Cross Validation Score")
    print("Test score",cross_model["test_score"].mean())
    print("Train score",cross_model["train_score"].mean())
