##################
#Import libraries#
##################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
#import Regression metric
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score

#class type of file
class Type:
    CSV = "csv"
    EXCEL = "excel"
    HTML = "html"

class Commercial:

    def __init__(self, file, type, predict_col = None):

        if type == Type.CSV:
            self.df = pd.read_csv(file)
        elif type == Type.EXCEL:
            self.df = pd.read_excel(file)
        else:
            print("Can not read this format")

        if predict_col != None:
            #make it deleted by user`s choice
            self.X = self.df.drop(["Avatar","Email","Address",predict_col], axis = 1 )
            self.y = self.df[predict_col]

    def train_test(self):
        """Return Train and split the DataFrame"""
        x_train, x_test, y_train, y_test = train_test_split(self.X, self.y, random_state = 5, test_size = 0.2)
        return x_train, x_test, y_train, y_test

    def model_predict(self, x_train, y_train):
        """Predict y_test; the hidden DataFrame"""
        model = LinearRegression()
        reg = model.fit(x_train, y_train)
        return reg

    def model_score(self, x_t, x_s, y_t, y_s):
        """Return score of the prediction"""
        print("X-y train model accuracy score", model.score(x_t, y_t))
        print("X-y test model accuracy score", model.score(x_s, y_s))

    def model_metric(self, y_test, y_pred):
        print("LinearRegression Metrics")
        print("mean_squared_error is: ", mean_squared_error(y_test, y_pred))
        print("mean_absolute_error is: ", mean_absolute_error(y_test, y_pred))
        print("meadian_absolute_error is: ", median_absolute_error(y_test, y_pred))
        print("R^2 is:", r2_score(y_test, y_pred))

    def delete_col(self, cols):
        self.df.drop(cols, axis = 1 , inplace = True)

        #optional function
    def display_frame(self):
        """Display all DataFrame and it`s train_test_split shapes"""
        x_train, x_test, y_train, y_test = self.train_test()
        print(self.df)
        print("x_train: ",x_train.shape)
        print("x_test: ",x_test.shape)
        print("y_train: ",y_train.shape)
        print("y_test: ",y_test.shape)

        #optional function
    def display_pairplot(self):
        """Display and compair all data to see their relationship"""
        sns.pairplot(data = self.df , kind = 'reg', diag_kind='auto')
        plt.show()

        #optional function
    def dsiplay_pred(self, predicted):
        """Display predicted value by model in histgraph"""
        sns.distplot([predicted], kde=True, hist = True, color ='g')
        plt.show()

    def save_model(self, model, name):
        """Save the appropriate model"""
        joblib.dump(model, name)

    def load_model(self, name):
        """Loading saved model"""
        return joblib.load(name)

if __name__ == "__main__":
    Ecom = Commercial("Ecommerce Customers.csv", "csv","Yearly Amount Spent")
    Ecom.delete_col(["Avatar","Email","Address"])
    x_train, x_test, y_train, y_test = Ecom.train_test()
    model = Ecom.model_predict(x_train, y_train)
    y_pred = model.predict(x_test)
    # print("predicted y:", y_pred)
    print("--"*20)
    Ecom.model_score(x_train, x_test, y_train, y_test)
    print("--"*20)
    Ecom.model_metric(y_test, y_pred)
    print("\n")
    # Ecom.save_model(model, "LinearRegression.h5")
    print(Ecom.load_model("LinearRegression.h5"))
