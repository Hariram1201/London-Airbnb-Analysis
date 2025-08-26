analyseCatData <- function(dataset, targetVar = NULL){
  "
  One-hot encodes categorical variables in a dataset for analysis, preserving the target variable.
  This transformation is useful for identifying patterns or correlations between categorical features 
  and the target variable (e.g., SalePrice).

  Parameters:
  -----------
  dataset : data.frame  
      The dataset containing both numeric and categorical variables. Should be cleaned prior to use.
  
  targetVar : character or NULL (optional)  
      The name of the target variable (e.g., 'SalePrice') to be retained in the final encoded dataset.
      If NULL, no target variable is preserved.

  Returns:
  --------
  dfEncoded : data.frame  
      A new data frame where categorical variables have been one-hot encoded, and the target variable
      is preserved for analysis.
  "
  
  # Step 1: Remove all numeric columns except for the target variable
  dataset <- dataset[, !(sapply(dataset, is.numeric) & names(dataset) != targetVar)]
  
  # Step 2: Replace all missing values with the placeholder "none" and remove the target variable
  dataset[is.na(dataset)] <- "none"
  
  # Step 3: Apply one-hot encoding to all remaining categorical variables
  dfEncoded <- oneHotEncode(dataset)
  
  # Step 4: Reattach the target variable to the encoded dataset
  dfEncoded[targetVar] <- dataset[targetVar]
  
  # Step 5: Return the fully encoded data frame
  return(dfEncoded)
}

createCorrMat <- function(dataSet){
  
  "
  Creates and displays a correlation matrix to investigate relationships between numerical variables in the dataset.
  
  Parameters:
  -----------
  dataSet : data.frame or matrix
      The dataset containing variables to analyze correlation.
  
  Returns:
  --------
  None
      Displays a correlation matrix plot.
  "
  
  corr_matrix <- cor(dataSet %>% select(where(is.numeric)), use = "complete.obs")
  corrplot(corr_matrix, method = "circle")
  mtext("Correlation Matrix", side = 3, line = 2, cex = 1.5, col = "black")
}


oneHotEncode <- function(data){
  "
  Converts categorical variables in a dataset into one-hot encoded binary variables.
  This is essential for machine learning models that require numeric input, enabling
  categorical data to be represented in a model-friendly format.

  Parameters:
  -----------
  data : data.frame  
      A dataset containing categorical variables to be transformed. Character variables
      will be automatically converted to factors before encoding.

  Returns:
  --------
  dfEncoded : data.frame  
      A new data frame containing only the one-hot encoded variables. The original
      categorical variables are replaced with binary indicator columns.
  "
  
  # Step 1: Convert all character-type columns to factor type
  data[sapply(data, is.character)] <- 
    lapply(data[sapply(data, is.character)], factor)
  
  # Step 2: Set up the dummyVars model for one-hot encoding, excluding the target variable 'SalePrice'
  dummies <- dummyVars(~ ., data = data, fullRank = TRUE)
  
  # Step 3: Generate the one-hot encoded variables by applying the model to the dataset
  dfEncoded <- predict(dummies, newdata = data)
  
  # Step 4: Convert the encoded result to a standard data frame
  dfEncoded <- as.data.frame(dfEncoded)
  
  return(dfEncoded)
}
normaliseData <- function(dataSet){
  "
  Normalises numeric variables in the dataset except for the target variable 'SalePrice'.
  This scaling helps to standardise features for better performance in many machine learning models.

  Parameters:
  -----------
  dataSet : data.frame
      The dataset containing variables to be normalised.

  Returns:
  --------
  trainDataNormalised : data.frame
      The dataset with numeric variables scaled to have mean 0 and standard deviation 1,
      excluding the 'SalePrice' column which remains unchanged.
  "
  
  # Create a copy of the input dataset to preserve the original data
  datasetNormalised <- dataSet
  
  # Iterate over each column, normalising numeric variables except 'SalePrice'
  for (col in colnames(dataSet)) {
    if (is.numeric(dataSet[[col]]) && col != "SalePrice") {
      # Apply z-score normalization (mean=0, sd=1) to the column
      datasetNormalised[[col]] <- scale(dataSet[[col]])
    }
  }
  
  # Return the dataset with normalised numeric variables
  return(datasetNormalised)
}

splitData <- function(dataSet){
  "
  Splits the dataset into training and testing subsets using an 80/20 split.
  This allows the model to be trained on 80% of the data and evaluated on the remaining 20%.

  Parameters:
  -----------
  dataSet : data.frame
      The complete dataset to be split.

  Returns:
  --------
  list(trainData, testData) : list of data.frames
      A list containing two data frames:
      - trainData: 80% of the data used for training the model.
      - testData: 20% of the data used for evaluating the model’s performance.
  "
  
  # Set seed for reproducibility to ensure consistent train-test splits across runs
  set.seed(123)
  
  # Calculate the number of rows and the cutoff for 80% training data
  totalRows <- nrow(dataSet)
  numRowsTrained <- floor(0.8 * totalRows)
  
  # Randomly shuffle row indices to avoid any ordering bias
  shuffledData <- sample(1:totalRows)
  
  # Select the first 80% indices for training set
  trainRows <- shuffledData[1:numRowsTrained]
  trainData <- dataSet[trainRows, ]
  
  # Select the remaining 20% indices for testing set
  testRows <- shuffledData[(numRowsTrained + 1):totalRows]
  testData <- dataSet[testRows, ]
  
  # Return the train and test datasets as a list
  return(list(trainData, testData))
}
