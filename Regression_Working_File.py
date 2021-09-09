import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
from yellowbrick import regressor
# we import the necessary libraries

data = pd.read_csv('student2/student-mat.csv', sep=';')
# we load in our our csv (Comma Seperated Values) file, order it, and load it into a variable

data = data[["G1","G2","G3","studytime","failures","absences"]]
# I created a new data variable with a couple columns in the original variable
# much of the data in our csv file is categorical, so we load the values that are numerical
# all of the column names listed above (G1, G2,G3, studytime, ...) are all represented with numbers

predict = "G3"

# I made a variable, predict, and stored a column name in it for later use

X = np.array(data.drop([predict],1))
# Now, we need to divide our data into Features and Targets. Features are basically attributes while the Target is what you want to predict. 
# Example: If we are trying to classify animals. The Target will be the animal while the Features will the attributes. Note that the values for both Target and Features need to be represented with numbers. 
# As hinted from above, the G3 column is what we need to predict, and we can't have that column in our Features. So, we get rid of that column and store our changes into the variable X
# Also, it is a convention to label our Features variable with a capital X. 

y = np.array(data[predict])
# our y here represents the Target. 
# The Target is what we want to predict, so that's why we assign the variable only to the G3 column. 

x_train, x_test , y_train , y_test = sklearn.model_selection.train_test_split(X,y,test_size=0.1)
# Now that we have our Features and our Target, we want to split our data into training and testing. 
# Basically, training data is what our model learns from and the testing data is what it tests on. 
# So, training data can be pictured as homework or practice test that students take. In this case, the student is the model. 
# Testing data can be thought as the exam or the final were the student demonstrates his/her knowledge. 

# the x_train and x_test is the Features, however, they are named differently. A portion is for training and the rest is for testing.
# the y_test and y_train is the Target values, however, one part is for training, and the other is for testing. 

model = linear_model.LinearRegression()
# then we create our model 

model.fit(x_train,y_train)
# we fit our model using our training data
# we feed our model with the neccessary info. 

acc = model.score(x_test,y_test)
# now, we can score our model with the testing data. 

pred = model.predict(x_test)
# now, with our x_test, we start to make predictions 

print(acc)
# print the accuracy. 
# I ran it a bit earlier and it came out as 0.70. Not exactly the best but the score varies as the data will change. 

regressor.residuals_plot(model, x_train,y_train,x_test,y_test)
# this is just another way to see how well our model performed.

print('pred: ', pred, 'actual: ', y_test)
# now we print out our data. The pred variable stores the values that our model predicted while the y_test is the actual results. 
# we can just compare the model's predictions to the actual results to get a better idea of our Linear Regression model. 

# Hopes this helps
# I am open to any feedback
