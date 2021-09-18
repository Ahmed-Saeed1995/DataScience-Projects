####Import libraies####
import pandas as pd
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix

def load_csv(file):
    """Load csv file"""
    df =  pd.read_csv(file)
    return df

def get_Xy(file):
    """Split the DataFrame into explantory variable and response variable"""
    X = file.iloc[:,:-1]
    y = file.iloc[:,-1]
    return X, y

def split_data(X, y):
    """Split the DataFrame into 1% Test and 99% Training"""
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1, random_state=30)
    return X_train, X_test, y_train, y_test

def model_rf(X_train, y_train):
    """Fit the model training X ->> the explnatory variables"""
    model = RandomForestClassifier()
    fitter = model.fit(X_train, y_train)
    return fitter

def model_performace(y_true, y_predict):
    """Check the model prediction performace how much the model predict correctly"""
    print("confusion Matrix :\n", confusion_matrix(y_true, y_predict))
    print("f1_score :", f1_score(y_true, y_predict))

def save_model(model, name):
    """Save to deploy the model fitted"""
    joblib.dump(model, name + ".h5")

def load_model(file):
    """Load the model"""
    return joblib.load(file)

#Evaluate the programme
if __name__ == "__main__":
    dataframe = load_csv("heart_statlog_cleveland_hungary_final.csv")
    X, y = get_Xy(dataframe)
    X_train, X_test, y_train, y_test = split_data(X,y)
    model = model_rf(X_train, y_train)
    y_prediction = model.predict(X_test)

    #Model score
    print(model.score(X_test, y_test))
    print(model.score(X_train, y_train))

    #Call model performace function
    model_performace(y_test, y_prediction)
    # Correlation matrix
    # sns.heatmap(dataframe.corr(), annot=True)
    # plt.show()

    #Save the model and load model
    # save_model(model, "Random_forest")
    print(load_model("Random_forest.h5"))
