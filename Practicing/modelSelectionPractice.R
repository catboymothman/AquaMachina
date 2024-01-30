# Practicing Model Selection
# Using a common set of training data, determine which model is most accurate at predicting the testing data
# Created by Ross Nygard on 1/29/2024
# Last modified by Ross Nygard on 1/30/2024 at 11:50 am

# install.packages("C50")
# install.packages("modeldata")
# install.packages("glmnet")
# install.packages("Matrix")
# install.packages("caret")
# install.packages("kernlab")
# install.packages("naivebayes")


library("modeldata")  # Contains the 'mlc_churn' data used for practice
library("C50")        # Contains churn data
library("caret")      # Using the 'train' function
library("glmnet")     # Required to use the glmnet method
library("Matrix")     # Required to use the glmnet method
library("kernlab")    # Required to use the svm linear method
library("naivebayes") # Required for the naive bayes method

data(mlc_churn) # Loads in the data used for this practice

set.seed(43) # Set a specific seed to prevent randomness generating a new set of samples

samp_perc <- 0.6 # Defines the percent of the sample used for training

tr <- sample(nrow(mlc_churn),
             round(nrow(mlc_churn)*samp_perc))
churnTrain <- mlc_churn[tr, ] # Create the training set
churnTest <- mlc_churn[-tr, ] # Create the testing set

table(mlc_churn$churn)/nrow(mlc_churn) # Percent of churn in the original data

table(churnTrain$churn)/nrow(churnTrain) # percent of churn in the training data, should be close to 14.14% (percent of churn in full data set)

my_folds <- createFolds(churnTrain$churn, 
                        k = 5) # Defines the number of folds
str(my_folds) # lists out the folds

sapply(my_folds, function(i){ # Verify that the folds have a similar proportion of the yes/no results
  table(churnTrain$churn[i])/length(i)
})

my_control <- trainControl(summaryFunction = twoClassSummary, # The output of the training will have two different options
                           classProb = TRUE, # recalculate class probabilities for each re-sample
                           savePredictions = TRUE, # All hold-out predictions for each re-sample should be saved
                           index = my_folds # Lists each element for each re-sampling iteration
                           )

# Create the glm model
glm_model <- train(churn ~ ., # Determines the output
                   churnTrain, # All of the training data
                   metric = "ROC", # Use ROC to determine which classification threshold is best
                   method = "glmnet", # Use a generelized linear regression method for this model
                   tuneGrid = expand.grid( # Data frame with tuning values
                     alpha = 0:1,# One of the tuning parameters for glmnet, mixing percentage
                     lambda = 0:10/10), # The other tuning parameter for glmnet, regularization parameter
                   trControl = my_control # Sets training controls to pre-established parameters
                   )
print(glm_model)
glm_plot <- plot(glm_model)
glm_plot # blue line is alpha = 0, pink is alpha = 1, x-axis is value of lambda, and y-axis is measure of accuracy

# Create a random forest model
rf_model <- train(churn ~ .,
                  churnTrain,
                  metric = "ROC", # Same metric as before
                  method = "ranger", # Random forest method
                  tuneGrid = expand.grid(
                    mtry = c(2, 5, 10, 19), # Number of randomly selected predictors, max being 20
                    splitrule = c("gini", "extratrees"), # Defines the splitting rules as gini and extra trees 
                    min.node.size = 1 # Sets minimum node size to 1
                  ), 
                  trControl = my_control # Sets the training controls as the pre-established parameters
                  )
print(rf_model)
rf_plot <- plot(rf_model)
rf_plot # Blue is gini splitting rule and pink is extra trees splitting rule. y-axis is accuracy, and x-axis is number random predictors

# Create a kNN model
knn_model <- train(churn ~ .,
             churnTrain,
             metric = "ROC", # same metric as before
             method = "knn", # kNN method
             tuneLength = 20, # Number of tuning parameters, 20 because the data contains 20 columns
             trControl = my_control
             )
print(knn_model)
knn_plot <- plot(knn_model)
knn_plot # x-axis is the number of neighbors considered, and y-axis is accuracy

# Create a Support Vector Machine (svm) model, using a linear basis first
svm_lin_model <- train(churn ~ .,
                   churnTrain,
                   metric = "ROC", # Same metric as before
                   method = "svmLinear", # Will do a radial one next; assumes linear grid for data distribution
                   tuneGrid = expand.grid(
                     C = c(10, 20, 30, 40, 50, 60, 70, 80, 90, 100) # Values of C randomly selected, can be refined later if this method is chosen
                   ),
                   trControl = my_control
                   )
print(svm_lin_model)
svm_lin_plot <- plot(svm_lin_model) # y-axis is accuracy, x-axis is a penalty for inaccuracy when testing
svm_lin_plot

# Create an svm radial model
svm_rad_model <- train(churn ~ .,
                       churnTrain,
                       metric = "ROC", # Same as before
                       method = "svmRadial", # Radial basis for svm rather than linear; assumes radial grid for data distribution
                       tuneLength = 10, # randomly selected
                       trControl = my_control
                       )
print(svm_rad_model)
svm_rad_plot <- plot(svm_rad_model)
svm_rad_plot # y-axis is accuracy, x-axis is a penalty for inaccuracy when testing

# Create a Naive-Bayes model
nb_model <- train(churn ~ .,
                  churnTrain,
                  metric = "ROC",
                  method = "naive_bayes", # Naive bayes model type
                  trControl = my_control
                  )
print(nb_model)
nb_plot <- plot(nb_model)
nb_plot # x-axis is the yes/no for whether customers disconnected and the y-axis is how accurate the model is at predicting them

# Comparing all the different models
model_list <- list(glmnet = glm_model,
                   rf = rf_model,
                   knn = knn_model,
                   svm_lin = svm_lin_model,
                   svm_rad = svm_rad_model,
                   nb = nb_model)
resamp = resamples(model_list) # collects, analyzes, and visualizes a set of resampling results from a common data set, ie churnTest
print(resamp)
summary(resamp)

lattice::bwplot(resamp, metric = "ROC") # Graphically compare the different model types using a box and whiskers plot

# Using model on set aside test data
p <- predict(rf_model, churnTest)
confusionMatrix(p, churnTest$churn)





