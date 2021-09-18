import pandas as pd
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score

def preprocessing(file):
    """
    show contents of the data file
    """
    df = pd.read_csv(file)
    # print(df.info())
    # print(df.describe())
    df.drop(["User ID"], inplace = True, axis =1)
    return df

def visualization(file):
    """
    some visualization on data
    """
    df = pd.read_csv(file)
    sns.countplot(df["Purchased"], hue = df["Gender"])
    sns.pairplot(df, kind = 'reg')

def to_numeric_data(file):
    """
    convert >>>> data to numeric values for ml
    """
    gender = pd.get_dummies(df["Gender"])
    newframe = pd.concat([df, gender], axis = 1)
    newframe.drop("Gender", inplace = True, axis = 1)
    return newframe

def get_Xy(newframe):
    """
    get all fetures X and predicted columns y
    """
    X = newframe.drop("Purchased", axis = 1)
    y = newframe["Purchased"]
    return X, y

def splits(X, y):
    """
    split the whole data to train the model
    """
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.1)
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    return x_train, x_test, y_train, y_test

def model_classifier(x_train, y_train):
    """
    user RandomForest algorithm to fit and predict the data
    """
    model = LogisticRegression(solver= 'liblinear',max_iter =100 )
    LR = model.fit(x_train, y_train)
    return LR

def model_score(model, xt, xs, yt, ys):
    """
    shows some model accuracy
    """
    print("Xy train model scores: ", model.score(xt, yt))
    print("Xy test model scores: ", model.score(xs, ys))

def metrics(y_test, y_pred):
    print("Confusion matrix \n", confusion_matrix(y_test, y_pred))
    print("Accuracy score", accuracy_score(y_test, y_pred))
    print("Precision score", precision_score(y_test, y_pred))
    print("F1_score", f1_score(y_test, y_pred))
    print("Recall", recall_score(y_test, y_pred))

def save_model(model , FileName):
    """
    save the model as name.h5
    """
    joblib.dump(model, FileName)
    print(f"Scucessfully model saved as {FileName}\n")

def load_model(FileName):
    """
    load deploied model
    """
    return joblib.load(FileName)

if __name__ == "__main__":
    file = "Social_Network_Ads.csv"
    df = preprocessing(file)
    newframe = to_numeric_data(df)
    X, y = get_Xy(newframe)
    x_train, x_test, y_train, y_test = splits(X, y)
    LR = model_classifier(x_train, y_train)
    y_pred = LR.predict(x_test)
    model_score(LR, x_train, x_test, y_train, y_test)
    metrics(y_test , y_pred)
    sns.heatmap(confusion_matrix(y_test, y_pred,), annot= True)
    plt.show()
    save_model(LR,"LogReg.h5")
    print(load_model("LogReg.h5"))
