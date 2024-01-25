# First practice at coding a supervised machine learning model
# k-Nearest Neighbor, or knn model
# Measures distance from k numbers of nearby neighbors for an unclassified data point
# Created by Ross Nygard on 1/25/24
# Last modified by Ross Nygard on 1/25/24 at 1:42 PM

library("class") # Package containing knn function

# Drawing 50 random iris observations to train the knn ML model
set.seed(12L) # Specifies a seed
tr <- sample (150,50) # Creates training set of 50 points of data
nw <- sample (150,50) # Creates new set of data to determine accuracy of ML model

knnResult <- knn(iris[tr, -5],     # Assigns training data
                 iris[nw,-5],      # Assigns testing data
                 iris$Species[tr], # Assigns response variables
                 k=1,              # Assigns number of neighbors to check
                 prob = FALSE
                 ) # Runs the ML model

head(knnResult) # See the first part of the factor after running the knn function

# Compare observed knn product to the observed outcome
table(knnResult, iris$Species[nw]) # Compares the Species printed by the knn function to the expected species
accK5 <- mean(knnResult == iris$Species[nw]) # Modified manually to check different values of k

# Checking accuracy for multiple values of k
accK1 # accuracy when k = 1; 0.96 for the above seed
accK2 # accuracy when k = 2; 0.94 for the above seed
accK3 # accuracy when k = 3; 0.94 for the above seed
accK4 # accuracy when k = 4; 0.96 for the above seed
accK5 # accuracy when k = 5; 0.94 for the above seed

# Checking what happens with 'prob = true'
table(attr(knnResult5Prob, "prob"))

