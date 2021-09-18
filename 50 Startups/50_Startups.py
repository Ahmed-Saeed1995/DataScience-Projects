#import libararies
import pandas as pd
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error

class Startups:

    def __init__(self, file):

        self.df = pd.read_csv(file)

    def preprocessing(self):
        """
        show simple summary to see what we should do before train data
        """
        print(self.df.info())
        print(self.df.describe())

    def fill_missing_values(self):
        """
        manually fill the missing values
        """
        self.df.iloc[19] = self.df["Marketing Spend"].mean()
        self.df.iloc[47] = self.df["Marketing Spend"].mean()
        self.df.iloc[48] = self.df["Marketing Spend"].mean()
        self.df.iloc[47] = self.df["R&D Spend"].mean()
        self.df.iloc[49] = self.df["R&D Spend"].mean()

    def delete_cols(self):
        """
        Delete unnessary columns
        """
        self.df.drop(["State"], inplace = True, axis = 1 )


    def std_scaler(self):
        """
        Scaling the whole data between (-1, 1)
        """
        std_scale = StandardScaler()
        scaler = std_scale.fit(self.df)
        newframe = std_scale.fit_transform(self.df)
        return scaler, newframe

    def get_Xy(self, dataframe):
        """
        Make the new numpy.arry as new DataFrame then split columns into X, y
        """
        newdf = pd.DataFrame(dataframe, columns =["R&D Spend","Administration","Marketing Spend","Profit"])
        X = newdf.iloc[:,:-1]
        y = newdf.iloc[:,-1]
        return X, y

    def train_test(self, X, y):
        """
        split the data using train_test_split
        with test_size = 70% of the whole data
        """
        x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = 20, test_size = 0.2)
        return x_train, x_test, y_train, y_test

    def model_algorithm(self, algorithm, x_train, y_train):
        """
        initialize model to predict new data
        """
        rf = algorithm
        model = rf.fit(x_train, y_train)
        return model

    def model_score(self, x_t, x_ts, y_t, y_ts):
        """
        Show model performance, the best value is 1.0
        """
        print("_"*60)
        print("X-y train model scores are :", model.score(x_t,y_t))
        print("X-y test model scores are :",model.score(x_ts,y_ts))

    def metrics(self, y_true, y_pred):
        """
        Show how much errors occure, the besr value is 0.0
        """
        print("_"*60)
        print("mean absolute error is :", mean_absolute_error(y_true, y_pred))
        print("mean squared error is :", mean_squared_error(y_true, y_pred))
        print("median absolute error is :", median_absolute_error(y_true, y_pred))

    def save_model(self, model, name):
        """
        Save model
        """
        joblib.dump(model, name)
        print("File saved successfully!")

    def load_model(self, name):
        """
        Load fitted model with appropriate weighted
        """
        return joblib.load(name)

#"""TEST CODE"""
if __name__ == "__main__":
    g = Startups("50_Startups.csv")
    # g.preprocessing()
    g.fill_missing_values()
    g.delete_cols()
    scaler, data_scaled = g.std_scaler()
    X, y = g.get_Xy(data_scaled)
    x_train, x_test, y_train, y_test = g.train_test(X, y)
    #fit data to model
    algorithm = RandomForestRegressor()
    model = g.model_algorithm(algorithm, x_train, y_train)
    y_pred = model.predict(x_test)
    #show accuracy
    g.model_score(x_train, x_test, y_train, y_test)
    g.metrics(y_test, y_pred)

    # g.save_model(scaler,"scaler.h5")
    # print(g.load_model("scaler.h5"))
    #
    # g.save_model(model,"RandomForest.h5")
    # print(g.load_model("RandomForest.h5"))
