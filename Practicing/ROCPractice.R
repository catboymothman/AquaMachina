# Practicing with Reciever Operating Characteristic (ROC) curve
# Shows multiple thresholds and the accuracy therein to create determine a threshold to use
# Created by Ross Nygard on 1/29/2024
# Last modified by Ross Nygard on 1/29/2024 at 5:15 PM

install.packages("caTools") # Contains the colAUC function which will be used later on

library("caTools")
library("caret")   # Contains 'train' function used later
library("mlbench") # Contains Sonar data being used

data(Sonar)

# Create the trainControl
my_control <- trainControl(method = "cv", # Cross validation
                           number = 20, # 10 folds
                           summaryFunction = twoClassSummary, # Computes sensitivity, specificity, and AUC for the ROC curve
                           classProbs = TRUE, # Recalculates class probabilities in each resample
                           verboseIter = FALSE # Suppresses dialogue with iterations
                           )

# Create training and testing data
tr <- sample(nrow(Sonar), round(nrow(Sonar)*0.60))
train <- Sonar[tr, ] # Creates the training set
test <- Sonar[-tr, ] # Creates the testing

# Train the glm with trainControl model
model <- train(Class ~ .,
               train,
               method = "glm", # Use the Generalized linear regression method
               trControl = my_control
               )

# Pint model to console
print(model)

p <- predict(model, # Use the model developed above
             test,  # Use the test data to check accuracy
             type = "prob")
summary(p)

# Use the colAUC function to visualize results
caTools::colAUC(p, test[["Class"]], plotROC = TRUE)





