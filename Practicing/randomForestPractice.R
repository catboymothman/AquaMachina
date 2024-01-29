# Practicing with Random Forest models
# Creating a bootstrapped dataset and multiple decision trees
# Created by Ross Nygard on 1/29/2024
# Last modified by Ross Nygard on 1/29/2024 at 1:33 PM

install.packages("rpart.plot", dependancies = TRUE) # Had to change Primary Repository to the one on UNC in Durham, NC to get this to install properly

library("rpart") # Recursive partitioning package
library("rpart.plot") # Plotting recursive partioning package
library("mlbench") # Sonar data being used
library("caret") # Contains the train function

# Plotting the first decision tree randomly made
data(Sonar)
# m <- rpart(Class ~ ., data = Sonar,
#            method = "class") # Decides based on the class of the object
# rpart.plot(m)
# 
# p <- predict(m, Sonar, type = "class") # Lists whether an object scanned is either a "M"ine or a "R"ock based on the prior decision tree
# table(p, Sonar$Class) # Displays the table to determine the accuracy of the decision tree

set.seed(42) # Used in practice to judge results and how they relate to the method given by tutorial

# Create a grid to use when creating the model
my_grid <- expand.grid(mtry = c(5, 10, 20, 40, 60), # Create a data frame, and define the number of variable randomly collected to sample for each split
                       splitrule = c("gini", "extratrees"), # Extratrees is a splitting rule for the decision trees that splits the nodes by choosing cut-points fully at random, and gini is a splitting rule that looks at the probability the model is wrong
                       min.node.size = 1 # Default is 1 for classification
                       )

# Create the model used
model <- train(Class ~ .,
               data = Sonar,
               method = "ranger", # Random forest method to create a model
               tuneGrid = my_grid,
               trControl = trainControl(method = "cv", # Resampling method; cross validation
                                        number = 5, # Number of folds for cross validation
                                        verboseIter = FALSE # Won't print the training log
                                        ),
               tuneLength = 5 # Sets number of hyperparameter values to test; commented out for practice elsewhere
               ) 

print(model) # Provide information about the model
plot(model) # Graphically represent the model's accuracy




