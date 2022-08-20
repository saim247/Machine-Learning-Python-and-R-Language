# Multiple regression using backward elimination method
# Backward Elimination
# 1. Select a significance level e.g for this model SL = 0.05
# 2. fit the full model with all possible predictors
# 3. Consider the preditor with the highest P-value. If P>SL, go to STEP 4, otherwise go to finish
# 4. Remove the predictor
# 5. Fit model without this variable

# Importing dataset
dataset = read.csv('50_Startups.csv')

# Encoding categorical data 
dataset$State = factor(dataset$State,
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1, 2, 3))

# Spliting the dataset into training set (80%) and test set (20%) of dataset.
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Backward Elimination steps
# <-------------------------------------------->

# 1. Select a significance level e.g for this model SL = 0.05

# <-------------------------------------------->

# 2. fit the full model with all possible predictors
# Fitting Multiple Linear Regression to the training set
# regressor = lm(formula = Profit ~ ., data = training_set)
# or
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = dataset)

# <-------------------------------------------->

# 3. Consider the preditor with the highest P-value. If P>SL, go to STEP 4, otherwise go to finish
summary(regressor)

# <-------------------------------------------->

# 4. Remove the predictor & 5. Fit model without this variable
# In this case remove most least significant variabel 'State' its value is 0.9
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
               data = dataset)

# <-------------------------------------------->

# Repeat the process from step 3 while removing the most least significant variable one by one
summary(regressor)

# remove Administration and fit model because its value is 0.6 which is greater than 0.05
regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
               data = dataset)
summary(regressor)

# remove Marketing.Spend and fit model because its value is 0.06 which is greater than 0.05
regressor = lm(formula = Profit ~ R.D.Spend,
               data = dataset)
summary(regressor)

# R.D.Spend is the only independent variable with highest significance
