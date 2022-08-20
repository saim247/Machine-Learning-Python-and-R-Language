#simple Linear Regression

# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd

 ###################################################
 
# Importing the data set 
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

 ###################################################
 
# Spliting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size = 1/3, #1/3 of whole dataset
                                                    random_state = 0)

 ###################################################
 
# Fitting Simple Linear Regression to the Trainig set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train) # The model will learn and ready to test

 ###################################################
 
# test the model by giving test dataset.
             #  or
# Predicting the Test set results
y_pred = regressor.predict(X_test) # this will predict by providing test dataset to model

 ###################################################
 
# Visualising the Training set results
# First plot X-axis and y-axis with labels
plt.scatter(X_train, y_train, color = 'red')
# Now plot the predictions point  
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience(Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

 ###################################################

# Visulizing the Test set results
#First plt X and Y asis with labels
plt.scatter(X_test, y_test, color = 'red')
# Now plot the predictions point
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

 ###################################################













