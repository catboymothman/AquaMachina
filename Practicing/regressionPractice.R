# Practice with linear regression models without Cross-Validation
# Minimizing root mean squared error (RMSE) to determine accuracy
# Created by Ross Nygard on 1/25/24
# Last modified by Ross Nygard on 1/25/24 at 2:29 PM

# Load the Caret library to use the data within
library("caret")

data(diamonds) # loads the data set
model1 <- lm(price ~ ., diamonds) # Linear model excluding price from the data examined
p1 <- predict(model1, diamonds) # Generic prediction for model fitting functions

# In-sample error on prediction
error1 <- p1 - diamonds$price
rmse_in <- sqrt(mean(error1^2)) # in-sample RMSE
rmse_in

# Create out of sample RMSE by removing some percent to serve as testing data
set.seed(29) # Choose a seed to keep samples excluded to play with other parameters of model
per <- 0.90  # Choose a percent value, 80% is just a random number
n_test <- nrow(diamonds) * per # Calculate the number for [percent]% of the rows
test <- sample(nrow(diamonds), n_test) # Create a dataframe with 80% of the rows of diamonds to serve as the testing data
model2 <- lm(price ~ ., data = diamonds[test, ]) # Create a linear model using the same set up as before, but with less training data
p2 <- predict(model2, diamonds[-test, ]) # Predict the price of the testing data based on the linear model

# Calculate out-of-sample RMSE using the 20% for testing
error2 <- p2 - diamonds$price[-test] # calculate the difference in the predicted prices from the actual prices
rmse_out <- sqrt(mean(error2^2))
rmse_out
rmse_in




