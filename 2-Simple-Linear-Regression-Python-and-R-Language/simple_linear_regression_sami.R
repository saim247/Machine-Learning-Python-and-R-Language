                       # Simple Linear Regression
# Importing the data set
dataset = read.csv('Salary_Data.csv')

# Spliting the dataset into Training and Test datasets
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)



# Fitting Simple Linear Regression to the Training set
regressor = lm(formula = Salary ~ YearsExperience, 
               data = training_set)

# Predicting the Test set results
y_pred = predict(regressor, newdata = test_set)


# Visualizing the Training set results
#install.packages('ggplot2')
library(ggplot2)

#plot for training set
ggplot() + 
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary), #define x and y axis of training set
             color = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            color = 'blue') +
  ggtitle('Salary vs Experience (Training set)') +
  xlab('Years of experience') +
  ylab('Salary')

#plot for test set
ggplot() + 
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary), #define x and y axis of training set
             color = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            color = 'blue') +
  ggtitle('Salary vs Experience (Test set)') +
  xlab('Years of experience') +
  ylab('Salary')

















