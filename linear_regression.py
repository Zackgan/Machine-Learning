import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
# from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")

# print(data.head())

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]  # this is an attribute
print(data.head())  # to print some data from the data set


predict = "G3"  # this is labels

# labels are what we want to get based on attributes
# we can predict more labels

X = np.array(data.drop([predict], 1))  # this is gonna return to us a new dataframe that just doesn't have G3 in it
# (this up here is training data) #based on this training data we're gonna predict another value

y = np.array(data[predict])  # this is going to be the labels

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

''' #to train the model
best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# essentially we're taking all of our attributes & labels and we're gonna split them up into four arrays
# x_train is gonna be a section of this variable "X" and y_train is gonna be a section of this variable "y"

    linear = linear_model.LinearRegression() # to choose the model for this data

    linear.fit(x_train, y_train)  # to find the best linear line (persamaan garis linear (y = mx + b))

    acc = linear.score(x_test, y_test)  # this is going to return us a value that is going to represent the accuracy of our
# model

    print(acc)  # print the accuracy

if acc > best:
    best = acc
    with open("studentmodel.pickle","wb") as f:  # this code function is to save our model
        pickle.dump(linear,f)'''

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)


print("Co: \n", linear.coef_)  # kode print variabel x dalam persamaan garis linear
print("intercept: \n", linear.intercept_)  # kode print variabel b dalam persamaan garis linear


predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p = "absences"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()

