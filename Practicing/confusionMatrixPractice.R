# Test accuracy with a confusion matric
# Sort results based on true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN)
# Created by Ross Nygard on 1/29/2024
# Last Modified by Ross Nygard on 1/29/2024 at 12:44 PM

install.packages("mlbench") # Need to install Rtools42 for this package to install correctly, following an update to R
# Rtools42 install link: https://cran.rstudio.com/bin/windows/Rtools/rtools42/rtools.html
library("mlbench") # Contains the Sonar dataset used in this model
library("caret") # Contains a single line function to create a confusion matrix
data(Sonar)

# 60/40 split
tr_perc <- 0.6
tr <- sample(nrow(Sonar), # take from the Sonar set
             round(nrow(Sonar)*tr_perc)) # Take 60% of rows from the Sonar set
train <- Sonar[tr,] # Define the training set as the rows from sonar
test <- Sonar[-tr,] # Define the test set as all rows not in the training set

model <- glm(Class ~ ., # Generalized linear model, 
             data = train,
             family = "binomial")
p <- predict(model, # Use model to predict
             test,  # Use test to determine the accuracy
             type = "response") # Gives the output as the predicted probabilities
summary(p)

# Return results as a table
cl <- ifelse(p > 0.1, "M", "R") # Sort the categories into "M" for mine or "R" for rock based on if they're greater or less than 0.5

table(cl,test$Class) # Horizontal are assignments that the model made, vertical are the real assignments made

# Using the carat package to get more information on the accuracy of this model
confusionMatrix(factor(cl), # The data
                test$Class) # The reference



