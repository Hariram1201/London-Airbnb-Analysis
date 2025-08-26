#Loads the necessary libraries
library(tidyverse) #Collection of packages for data manipulation, visualisation and analysis 
library(ggplot2) #Collection of packages to create static, interactive and customisable plots
library(corrplot) #Collection of packages specialised for visualising correlation matrices
library(tidyr) #Collection of packages designed for data cleaning and reshaping
library(plotly) #Collection of packages used to create interactive visualisations 
library(caret) #Collection of packages used to build, train and evaluate machine learning models
library(glmnet) #Collection of packages used for regularisation regression
library(rpart) #Collection of packages used for implementing recursive partitioning for 
#               classification and regression trees (CART) models
library(rpart.plot) #Collection of packages that provide tools for visualizing and plotting decision 
#                    trees created using the rpart library
library(randomForest) #Collection of packages to implement the Random Forest algorithm
library(gbm) #Collection of packages to implement gradient boosting Machines
library(xgboost) #Collection of packages to implement a highly fficient and scalable implementation
#                 gradient boosting Machines, particularly for larger datasets
library(knitr) #Collection of packages to display prettier tables

#Loads the necessary functions
source("inspectData.R")
source("dataPreprocessing.R")
source("plotsAndGraphs.R")
source("createCorrMat.R")
source("priceTrends.R")
source("multiCollinarity.R")
source("featureEngineering.R")
source("createNewVar.R")
source("predModel_regularised.R")
source("evaluateModel.R")
source("predModel_treeBased.R")
source("predModel_ensemble.R")

source("plotActVsPred.R")

source("modelEval.R")

# ==============================
# DATA INSPECTION
# ==============================

#Loads the necessary data files used to train and test the models
trainData <- read.csv("/Users/hariramgnanachandran/Documents/Data Science : Machine Learning Work/Projects/Predicting House Prices with Regression Models/Data/train.csv") #Data used to train the model
testData <- read.csv("/Users/hariramgnanachandran/Documents/Data Science : Machine Learning Work/Projects/Predicting House Prices with Regression Models/Data/test.csv") #Data used to test the model

#Calls a function to analyse the training data - outputs the structure of the data and the summary
#statistics of each numerical column in the data
inspectData(trainData)

# ==============================
# PREPROCESSING
# ==============================

#Calls a function to clean the training data - displays a box and whisker plot of the data before and 
#after being cleaned. Then removes any outliers from the dataset to be used
trainDataCleaned <- cleanData(trainData, "SalePrice", "Boxplot of Sale Price", "Sale Price ($)") 

#Calls a function to change the NA values of numerical variables to 0 and categorical variables to 
#'none'
trainDataCleaned <- changeNA(trainDataCleaned)

# ==============================
# EXPLORATORY DATA ANALYSIS
# ==============================

#Calls a function to visualise the cleaned training data - displays a histogram on sale price against
#frequency and a correlation matrix of the different numerical variables
histogram(trainDataCleaned, "SalePrice", "Distribution of SalePrice", "Sale Price ($)", "Frequency")

#Calculates the correlation matrix for numeric variables in the dataset and 
#visualizes it using a circular plot to show the strength and direction of 
#relationships between the variables.
createCorrMat(trainDataCleaned)

#Calls a function to plot graphs of key variables identified from the heat map which show a strong 
#correlation with the sale price of the house 
priceTrends(trainDataCleaned)

#Calls a function to plot graphs of variables that show multicollinearity to have a better understanding
#of their relationship
multiCollinarity(trainDataCleaned)

# ==============================
# FEATURE ENGINEERING
# ==============================

# ------------------------------
# ONE-HOT ENCODE
# ------------------------------

#Calls a function to one-hot encode all categorical data variables
dfEncoded <- analyseCatData(trainDataCleaned, "SalePrice")

#Creates new matrices from which correlation matrices can be created to spot trends in the 
#categorical data
dfEncoded1 <- as.data.frame(dfEncoded[, 1:45])
dfEncoded1$SalePrice <- trainDataCleaned$SalePrice
dfEncoded2 <- as.data.frame(dfEncoded[, 46:90])
dfEncoded2$SalePrice <- trainDataCleaned$SalePrice
dfEncoded3 <- as.data.frame(dfEncoded[, 91:135])
dfEncoded3$SalePrice <- trainDataCleaned$SalePrice
dfEncoded4 <- as.data.frame(dfEncoded[, 136:180])
dfEncoded4$SalePrice <- trainDataCleaned$SalePrice
dfEncoded5 <- as.data.frame(dfEncoded[, 181:225])
dfEncoded5$SalePrice <- trainDataCleaned$SalePrice

#Creates correlation matrices to spot trends in the categorical variables with the sale price
createCorrMat(dfEncoded1)
createCorrMat(dfEncoded2)
createCorrMat(dfEncoded3)
createCorrMat(dfEncoded4)
createCorrMat(dfEncoded5)

# ------------------------------
# CREATE NEW VARIABLES
# ------------------------------

#Function to create more variables from the raw data to potentially find more patterns
trainDataClnFEng <- createNewVar(trainDataCleaned)
trainDataClnFEng <- changeNA(trainDataClnFEng)

# ------------------------------
# ONE-HOT ENCODE NEW VARIABLES
# ------------------------------

#Extracts the new variables ready to be analysed for correlation with sale price
newVar <- as.data.frame(trainDataClnFEng[, 81:88])

#Calls a function to one-hot encode the new categorical varaibles 
dfEncoded <- oneHotEncode(newVar)

#Adds the sale price to the matrix with the new variables stored
dfEncoded$SalePrice <- trainDataClnFEng$SalePrice

#Creates correlation matrices to spot trends in the new variables with the sale price
createCorrMat(dfEncoded)

# ------------------------------
# NORMALISE 
# ------------------------------

#Calls a function to normalise the final data set of the training to be used
trainDataNormalised <- normaliseData(trainDataClnFEng)

# ------------------------------
# ONE-HOT ENCODE
# ------------------------------

#Calls a function to one-hot encode the final data set of the training to be used
encodeData <- oneHotEncode(trainDataNormalised)
encodeData$SalePrice <- trainDataCleaned$SalePrice

# ------------------------------
# SPLIT TRAINING & TESTING 
# ------------------------------

#Calls a function to split the training data into a 80/20 split to train the model and evaluate its'
#performance
trainAndTestData <- splitData(encodeData)

#Converts the data into data frame format from list form
trainDataSplit <- as.data.frame(trainAndTestData[1])
testDataSplit <- as.data.frame(trainAndTestData[2])

# ==============================
# Linear Regression Models
# ==============================

#Determines the mean value of the house sale prices in the training data
meanSalePrice <- mean(trainDataSplit$SalePrice)

# ------------------------------
# INTERCEPT ONLY LINEAR REGRESSION MODEL
# ------------------------------

# Key points:
# - This is the simplest regression model possible: it predicts the average value 
#   of the target variable (SalePrice) for all data points, ignoring all features.
# - Acts as a baseline to compare the performance of more complex models.
# - Assumes that none of the input features affect the prediction.

# Call the function that fits the intercept-only model and makes predictions on test data
interceptData <- interceptOnlyModel(trainDataSplit, testDataSplit, "SalePrice")

# Extract predictions and the fitted model object from the returned list
interceptPredictions <- interceptData[[1]]
interceptModel <- interceptData[[2]]

# Evaluate the intercept-only model’s predictions against the actual values
modelEvaluation <- evaluateModel(testDataSplit, interceptPredictions, meanSalePrice)

# Extract evaluation metrics: MAE, RMSE, and R-squared from the evaluation results
interMAE <- unlist(modelEvaluation)[1]
interRMSE <- unlist(modelEvaluation)[2]
interRSqr <- unlist(modelEvaluation)[3]

# ------------------------------
# MULTIPLE LINEAR REGRESSION MODEL
# ------------------------------

# Key Points:
# - Extension of simple linear regression that models the relationship between 
#   the target variable (SalePrice) and multiple predictor variables (features).
# - Uses all available features to predict the target.

# Call the function that fits the multiple linear regression model and predicts on test data
multLinData <- multLinModel(trainDataSplit, testDataSplit, "SalePrice")

# Extract predictions and the fitted linear model from the returned list
multLinPredictions <- multLinData[[1]]
multLinModel <- multLinData[[2]]

# Check for complete cases to ensure predictions and actual values align without missing data
valid <- complete.cases(multLinPredictions, testDataSplit$SalePrice)

# Evaluate the multiple linear regression predictions on valid (non-missing) data only
modelEvaluation <- evaluateModel(testDataSplit[valid, ], multLinPredictions[valid], meanSalePrice)

# Extract evaluation metrics: MAE, RMSE, and R-squared
multLinMAE <- unlist(modelEvaluation)[1]
multLinRMSE <- unlist(modelEvaluation)[2]
multLinRSqr <- unlist(modelEvaluation)[3]

# Plot feature importance based on absolute coefficients of the linear regression model
linearFeatureImportance(multLinModel)

# ------------------------------
# Ridge Regression (L2 Regularisation)
# ------------------------------

# Key Points:
# - Ridge regression is a linear model that adds an L2 penalty to reduce 
#   overfitting and multicollinearity by shrinking coefficients toward zero.
# - Unlike Lasso, coefficients are not set exactly to zero, so all features remain.
# - Helps improve model generalization when predictors are correlated.

# Call the function to train Ridge regression and get predictions on test data
ridgeData <- ridgeReg(trainDataSplit, testDataSplit, "SalePrice")

# Extract predictions and the trained ridge model from the returned list
ridgePredictions <- ridgeData[[1]]
ridgeModel <- ridgeData[[2]]

# Evaluate the ridge regression model predictions
modelEvaluation <- evaluateModel(testDataSplit, ridgePredictions, meanSalePrice)

# Extract evaluation metrics: MAE, RMSE, and R-squared
ridgeMAE <- unlist(modelEvaluation)[1]
ridgeRMSE <- unlist(modelEvaluation)[2]
ridgeRSqr <- unlist(modelEvaluation)[3]

# Plot feature importance based on the ridge model’s coefficients
glmnetFeatureImportance(ridgeModel)

# ------------------------------
# Lasso Regression (L1 Regularisation)
# ------------------------------

# Key Points:
# - Lasso regression adds an L1 penalty which can shrink some coefficients exactly to zero.
# - This feature selection property helps identify the most important predictors by removing less relevant ones.
# - Useful for reducing model complexity and improving interpretability.

# Call the function to train Lasso regression and get predictions on test data
lassoData <- lassoReg(trainDataSplit, testDataSplit, "SalePrice")

# Extract predictions and the trained Lasso model from the returned list
lassoPredictions <- lassoData[[1]]
lassoModel <- lassoData[[2]]

# Evaluate the Lasso regression model predictions
modelEvaluation <- evaluateModel(testDataSplit, lassoPredictions, meanSalePrice)

# Extract evaluation metrics: MAE, RMSE, and R-squared
lassoMAE <- unlist(modelEvaluation)[1]
lassoRMSE <- unlist(modelEvaluation)[2]
lassoRSqr <- unlist(modelEvaluation)[3]

# Plot feature importance based on the Lasso model’s coefficients
glmnetFeatureImportance(lassoModel)

# ==============================
# Tree Based Models
# ==============================

# ------------------------------
# Decision Tree Regressor
# ------------------------------

# Call the function to train the decision tree regressor and make predictions on test data
decTreeData <- decTreeReg(trainDataSplit, testDataSplit)

# Extract predictions and the trained decision tree model from the returned list
decTreePredictions <- decTreeData[[1]]
decTreeModel <- decTreeData[[2]]

# Evaluate the decision tree model predictions
modelEvaluation <- evaluateModel(testDataSplit, decTreePredictions, meanSalePrice)

# Extract evaluation metrics: MAE, RMSE, and R-squared from the evaluation results
decTreeMAE <- unlist(modelEvaluation)[1]
decTreeRMSE <- unlist(modelEvaluation)[2]
decTreeRSqr <- unlist(modelEvaluation)[3]

# Plot feature importance using the final fitted model within the caret object
decisionTreeFeatureImportance(decTreeModel$finalModel)

# ------------------------------
# Random Forest Regressor
# ------------------------------

# Call the function to train the random forest regressor and make predictions on test data
randForData <- randForReg(trainDataSplit, testDataSplit)

# Extract evaluation metrics: MAE, RMSE, and R-squared from the returned list
randForMAE <- unlist(randForData)[1]
randForRMSE <- unlist(randForData)[2]
randForRSqr <- unlist(randForData)[3]

# Extract predictions made by the random forest model
randForPredictions <- randForData[[4]]

# Extract the trained random forest model object
randForModel <- randForData[[5]]

# Plot feature importance using the final fitted model within the caret random forest object
randomForestFeatureImportance(randForModel$finalModel)

# ==============================
# Ensemble & Advanced Models
# ==============================

# Prepare the training matrix for xgboost (exclude the target column)
trainMatrix <- as.matrix(trainDataSplit[, -which(names(trainDataSplit) == "SalePrice")])
testMatrix <- as.matrix(testDataSplit[, -which(names(testDataSplit) == "SalePrice")])

# Define the target variable
trainLabel <- trainDataSplit$SalePrice
testLabel <- testDataSplit$SalePrice

# ------------------------------
# Gradient Boosting Model
# ------------------------------

# Call the function 'gradBoost' to train the gradient boosting model using training data
# 'trainMatrix' contains features in matrix form, 'trainLabel' contains the target values
gradBoostData <- gradBoost(trainMatrix, trainLabel, testMatrix)

# Extract the predicted house prices on the test data from the returned list
gradBoostPredictions <- gradBoostData[[1]]

# Extract the trained gradient boosting model object from the returned list
gradBoostModel <- gradBoostData[[2]]

# Evaluate the model's predictions by comparing with actual house prices in 'testDataSplit'
# Uses a custom evaluation function 'evaluateModel' that calculates performance metrics
modelEvaluation <- evaluateModel(testDataSplit, gradBoostPredictions, meanSalePrice)

# Extract evaluation metrics: MAE, RMSE, and R-squared from the evaluation results
gradBoostMAE <- unlist(modelEvaluation)[1]
gradBoostRMSE <- unlist(modelEvaluation)[2]
gradBoostRSqr <- unlist(modelEvaluation)[3]

# Plot the feature importance using the final fitted model within the gradient boosting model object
# Here, 'finalModel' refers to the actual trained xgboost model inside the wrapper object
gradientBoostingFeatureImportance(gradBoostModel$finalModel)

# ------------------------------
# XGBoost Model
# ------------------------------

# Calls a function to train the XGBoost model and predict house prices
# 'trainMatrix' contains training features, 'trainLabel' contains training targets,
# 'testMatrix' contains test features to predict on
XGBoostData <- xgboost(trainMatrix, trainLabel, testMatrix)

# Extract predicted house prices on the test dataset from the returned list
XGBoostPredictions <- XGBoostData[[1]]

# Extract the trained XGBoost model object from the returned list
XBBoostModel <- XGBoostData[[2]]

# Evaluate the XGBoost model’s predictions against actual house prices
# 'evaluateModel' returns performance metrics such as MAE, RMSE, and R²
modelEvaluation <- evaluateModel(testDataSplit, XGBoostPredictions, meanSalePrice)

# Extract evaluation metrics: MAE, RMSE, and R-squared from the evaluation results
xgBoostMAE <- unlist(modelEvaluation)[1]
xgBoostRMSE <- unlist(modelEvaluation)[2]
xgBoostRSqr <- unlist(modelEvaluation)[3]

# Plot feature importance using the final trained XGBoost model and training data
# 'finalModel' contains the underlying xgb.Booster object used for importance calculation
xgboostFeatureImportance(XBBoostModel$finalModel, trainMatrix)

# ==============================
# Evaluate Models
# ==============================

# Create a data frame with your model results
modelName = c("Intercept-Only", "Linear Regression", "Ridge Regression", "Lasso Regression", 
            "Decision Tree", "Random Forest", "Gradient Boosting", "XGBoost")
model = list(multLinModel, ridgeModel, lassoModel, decTreeModel, randForModel, gradBoostModel$finalModel, XBBoostModel$finalModel)
MAE = c(interMAE, multLinMAE, ridgeMAE, lassoMAE, decTreeMAE, randForMAE, gradBoostMAE, xgBoostMAE)
RMSE = c(interRMSE, multLinRMSE, ridgeRMSE, lassoRMSE, decTreeRMSE, randForRMSE, gradBoostRMSE, xgBoostRMSE) 
R_Squared = c(interRSqr, multLinRSqr, ridgeRSqr, lassoRSqr, decTreeRSqr, randForRSqr, gradBoostRSqr, xgBoostRSqr) 

modelEval(modelName, model, MAE, RMSE, R_Squared, '/Users/hariramgnanachandran/Documents/Data Science : Machine Learning Work/Projects/Predicting House Prices with Regression Models/Outputs/Model Output/')




