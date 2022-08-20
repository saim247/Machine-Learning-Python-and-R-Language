#Importing the libraries
import numpy as np
import pandas as pd

#-------------------------------------------------#

np.set_printoptions(threshold=np.inf, linewidth=np.nan)
# Importing the dataset
dataset =pd.read_csv('Data.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values

#-------------------------------------------------#

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:, 1:3])
X[:,1:3] = imputer.transform(X[:, 1:3])

#-------------------------------------------------#

# Encoding categorical data on independent variables
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ctx = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],   # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                                         # Leave the rest of the columns untouched
)
X = ctx.fit_transform(X)

#-------------------------------------------------#

# Encoding categorical data on dependent variables
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y)

#-------------------------------------------------#

# Splitting the data into test set and training set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#-------------------------------------------------#

# Feature Scaling on independent variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) 

#-------------------------------------------------#

# Feature Scaling on dependent variables
sc_y = StandardScaler()
y_train = y_train.reshape(-1,1)
y_train = sc_y.fit_transform(y_train)
y_test = y_test.reshape(-1,1)
y_test = sc_y.transform(y_test)   
                         



















