                          # Polynomial Regression
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2]
y = dataset.iloc[:, 2]

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#Fitting polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# Visualising the Linear REgression results
plt.scatter(X, y, color = 'red')
#plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.plot(X['Level'], lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear regression result)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X['Level'], lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
print("level 6.5 prediction of linear regression is {0}", lin_reg.predict([[6.5]]))

# Predicting a new result with Polynomial Regression
#print("level 6.5 prediction of linear regression is {0}", lin_reg2.predict(poly_reg.fit_transform(6.5)))
print("level 6.5 prediction of linear regression is {0}", lin_reg2.predict(poly_reg.fit_transform([[6.5]])))















