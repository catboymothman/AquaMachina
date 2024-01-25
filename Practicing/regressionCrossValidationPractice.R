# Practice with linear regression models with Cross-validation
# Minimizing root mean squared error (RMSE) to determine accuracy
# Created by Ross Nygard on 1/25/24
# Last modified by Ross Nygard on 1/25/24 at [TIME]

# Load the Caret library to use the data within
library("caret") # contains 'diamonds' data to be used, and 'train' function used to create folds
library("MASS")  # Contains data on Boston homes to use as practice

# Creating the folds and model to be used for diamond data
set.seed(29)
model_dia <- train(price ~ ., diamonds, # Train the model with the diamonds data set and use price as the output
               method = "lm", #linear model
               trControl = trainControl(method = "cv",      # Resampling method
                                        number = 10,        # Number of folds
                                        verboseIter = FALSE # verboseIter determines if the train function prints its iterations
                                        )
               )

# Using the model to predict the price of diamonds
p_dia <- predict(model_dia, diamonds)
error_dia <- p_dia - diamonds$price
rmse_xval_dia <- sqrt(mean(error_dia^2)) # Cross validated RMSE
rmse_xval_dia

# SWITCHING TO BOSTON HOME DATA AS INDEPENDENT PRACTICE

# Creating the folds and model to be used for Boston home data
set.seed(29)
model_Bos <- train(medv ~ ., Boston, # Train the model with the diamonds data set and use price as the output
                   method = "lm", #linear model
                   trControl = trainControl(method = "cv",      # Resampling method
                                            number = 10,        # Number of folds
                                            verboseIter = FALSE # verboseIter determines if the train function prints its iterations
                   )
)

# Using the model to predict the price of diamonds
p_Bos <- predict(model_Bos, Boston)
error_Bos <- p_Bos- Boston$medv
rmse_xval_Bos <- sqrt(mean(error_Bos^2)) # Cross validated RMSE
rmse_xval_Bos



