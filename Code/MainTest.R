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
# DATA COLLECTION
# ==============================

#Loads the necessary data files used to train and test the models
trainData <- read.csv("/Users/hariramgnanachandran/Documents/Data Science : Machine Learning Work/Projects/Predicting House Prices with Regression Models/Data/train.csv") #Data used to train the model
testData <- read.csv("/Users/hariramgnanachandran/Documents/Data Science : Machine Learning Work/Projects/Predicting House Prices with Regression Models/Data/test.csv") #Data used to test the model

# ==============================
# PREPROCESSING
# ==============================

#Calls a function to change the NA values of numerical variables to 0 and categorical variables to 
#'none'
testDataCleaned <- changeNA(testData)

# ==============================
# FEATURE ENGINEERING
# ==============================

# ------------------------------
# CREATE NEW VARIABLES
# ------------------------------

#Function to create more variables from the raw data to potentially find more patterns
testDataClnFEng <- createNewVar(testDataCleaned)
testDataClnFEng <- changeNA(testDataClnFEng)

# ------------------------------
# NORMALISE 
# ------------------------------

#Calls a function to normalise the final data set of the training to be used
testDataNormalised <- normaliseData(testDataClnFEng)

# ------------------------------
# ONE-HOT ENCODE
# ------------------------------

#Calls a function to one-hot encode the final data set of the testing to be used
encodeData <- oneHotEncode(testDataNormalised)

# ==============================
# PREDICTION WITH HOUSE ID
# ==============================

library(xgboost)

# -----------------------------
# Step 0: Load test data and final model
# -----------------------------
# testData should contain the original Id column
# finalModel: trained xgb.Booster object
finalModel <- readRDS("/Users/hariramgnanachandran/Documents/Data Science : Machine Learning Work/Projects/Predicting House Prices with Regression Models/Outputs/Model Output/final_model_gradBoost.rds")
output_path <- "/Users/hariramgnanachandran/Documents/Data Science : Machine Learning Work/Projects/Predicting House Prices with Regression Models/Outputs/Test Data Predictions/house_price_predictions.csv"

# Extract Ids
house_ids <- testData$Id

# -----------------------------
# Step 1: Align test data features
# -----------------------------
# encodeData: your preprocessed & one-hot encoded test dataset (excluding Id)
train_features <- finalModel$feature_names

# Add missing columns in test data
missing_cols <- setdiff(train_features, colnames(encodeData))
for (col in missing_cols) encodeData[[col]] <- 0

# Remove extra columns not in training
extra_cols <- setdiff(colnames(encodeData), train_features)
if(length(extra_cols) > 0) encodeData <- encodeData[, !(colnames(encodeData) %in% extra_cols)]

# Reorder columns to match training
encodeData <- encodeData[, train_features]

# -----------------------------
# Step 2: Convert all columns to numeric
# -----------------------------
encodeData[] <- lapply(encodeData, function(x) {
  if(is.logical(x)) as.numeric(x)
  else if(is.factor(x)) as.numeric(as.integer(x))
  else if(is.character(x)) as.numeric(as.factor(x))
  else x
})

# -----------------------------
# Step 3: Convert to numeric matrix and create DMatrix
# -----------------------------
testMatrix <- do.call(cbind, lapply(encodeData, as.numeric))
stopifnot(all(sapply(testMatrix, is.numeric)))

dtest <- xgb.DMatrix(data = testMatrix)

# -----------------------------
# Step 4: Make predictions
# -----------------------------
predictions <- predict(finalModel, dtest)

# -----------------------------
# Step 5: Save predictions with Id
# -----------------------------
prediction_df <- data.frame(
  Id = house_ids,
  SalePrice = predictions
)

write.csv(prediction_df, output_path, row.names = FALSE)
cat("Predictions saved to:", output_path, "\n")
