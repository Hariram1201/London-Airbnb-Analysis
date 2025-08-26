decTreeReg <- function(trainData, testData) {
  "
  Trains a Decision Tree Regressor using grid search and 10-fold cross-validation 
  to predict house prices, then makes predictions on the test data.

  This function uses the 'caret' package to tune the complexity parameter (cp) 
  and selects the best model based on cross-validated RMSE. It also plots the 
  final tree and visualizes the RMSE vs cp values to aid interpretation.

  Parameters:
  -----------
  trainData : data.frame
      The training dataset (80% of the full dataset) used to train the model.

  testData : data.frame
      The testing dataset (20% of the full dataset) used to evaluate the model.

  Returns:
  --------
  predictions : numeric vector
      Predicted values of house sale prices for the test dataset.
  "
  
  # Set up training control with 10-fold cross-validation
  train_control <- trainControl(method = "cv", number = 10)
  
  # Define a grid of complexity parameters (cp) to tune
  grid <- expand.grid(
    cp = seq(0.001, 0.0500, by = 0.0050)  
  )
  
  # Train the Decision Tree Regressor using caret with grid search
  treeModel <- train(
    SalePrice ~ ., 
    data = trainData,
    method = "rpart",
    trControl = train_control,
    tuneGrid = grid
  )
  
  # Plot the final decision tree
  rpart.plot(treeModel$finalModel, type = 2, extra = 101, fallen.leaves = TRUE,
             main = "Tuned Decision Tree")
  
  # Plot RMSE vs Complexity Parameter (cp)
  results <- treeModel$results
  plot1 <- ggplot(results, aes(x = cp, y = RMSE)) +
    geom_line() +
    geom_point() +
    labs(title = "RMSE vs CP for Decision Tree Model",
         x = "Complexity Parameter (cp)",
         y = "Root Mean Squared Error (RMSE)") +
    theme_minimal()
  print(plot1)  # Display the plot
  
  # Output the best cp value chosen by grid search
  bestCp <- treeModel$bestTune$cp
  cat("The best cp value chosen by grid search is:", bestCp, "\n")
  
  # Predict on the test dataset using the best model
  predictions <- predict(treeModel, newdata = testData)
  
  return(list(predictions, treeModel))
}

decisionTreeFeatureImportance <- function(dtModel) {
  "
  Extracts and visualizes feature importance from a decision tree model 
  (rpart) based on the variable importance measure provided by the model.

  Parameters:
  -----------
  dtModel : rpart
      A trained decision tree model object created using rpart().

  Returns:
  --------
  A horizontal bar plot displaying the relative importance of features in the decision tree.
  "
  
  # Extract variable importance from the decision tree model
  importance <- dtModel$variable.importance
  
  # Convert to data frame for plotting
  importance_df <- data.frame(
    Feature = names(importance),
    Importance = as.numeric(importance)
  )
  
  # Order features by importance descending
  importance_df <- importance_df[order(-importance_df$Importance), ]
  
  # Plot horizontal bar chart of feature importance
  ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
    geom_bar(stat = "identity", fill = "darkorange") +
    coord_flip() +
    labs(title = "Decision Tree Feature Importance",
         x = "Feature", y = "Importance") +
    theme_minimal()
}

randForReg <- function(trainData, testData){
  "
  Trains and evaluates a Random Forest Regressor using grid search for `mtry` 
  and multiple values for `ntree`, returning performance metrics and predictions.

  This function performs hyperparameter tuning for `mtry` and loops over 
  different `ntree` values (number of trees in the forest) using 5-fold 
  cross-validation. For each model, it evaluates performance using MAE, 
  RMSE, and R-squared, and selects the best model based on highest R-squared.

  Parameters:
  -----------
  trainData : data.frame
      The training dataset (80% of the full data) used to train the models.

  testData : data.frame
      The testing dataset (20% of the full data) used to evaluate the models.

  Returns:
  --------
  list:
      - bestRandForMAE : numeric
          Mean Absolute Error of the best performing model.
      - bestRandForRMSE : numeric
          Root Mean Squared Error of the best performing model.
      - bestRandForRSqr : numeric
          R-squared value (coefficient of determination) of the best performing model.
      - prediction : numeric vector
          Predicted house prices from the best Random Forest model.
  "
  
  # Create a tuning grid for the 'mtry' hyperparameter (number of features at each split)
  tuneGrid <- expand.grid(mtry = c(1:20))
  
  # Define 5-fold cross-validation for training
  trainControl <- trainControl(
    method = "cv",
    number = 5,
    search = "grid"
  )
  
  # Initialize result matrix to store MAE, RMSE, and R-squared across different ntree values
  nTreeResult <- data.frame(matrix(NA, nrow = 3, ncol = 5))
  x <- 1  # Column index for nTreeResult
  results_df <- data.frame()  # To store all RMSE values for plotting
  
  # Loop over ntree values to tune number of trees
  for (ntree in c(50, 100, 200)){
    # Train Random Forest model with current ntree and mtry grid
    rfModel <- train(
      SalePrice ~ .,
      data = trainData,
      method = "rf",
      trControl = trainControl,
      tuneGrid = tuneGrid,
      ntree = ntree,
      nodesize = 5,
      importance = TRUE
    )
    
    print(rfModel)  # Print summary of model tuning results
    
    # Make predictions on the test set
    prediction <- predict(rfModel, newdata = testData)
    
    # Evaluate the model performance
    modelEvaluation <- evaluateModel(testData, prediction, meanSalePrice)
    
    # Extract and store MAE, RMSE, R²
    randForMAE <- as.data.frame(modelEvaluation[1])
    nTreeResult[1, x] <- randForMAE
    randForRMSE <- as.data.frame(modelEvaluation[2])
    nTreeResult[2, x] <- randForRMSE
    randForRSqr <- as.data.frame(modelEvaluation[3])
    nTreeResult[3, x] <- randForRSqr
    
    x <- x + 1
    
    # Add ntree info to model results for visualization
    rfModel$results$ntree <- ntree
    results_df <- rbind(results_df, rfModel$results)
  }
  
  # Identify best model based on highest R² value
  largestRSqr <- which.max(nTreeResult[3, ])
  bestnTree <- c(50, 100, 200)[largestRSqr]
  
  # Retrieve evaluation metrics for the best model
  bestRandForMAE <- nTreeResult[1, largestRSqr]
  bestRandForRMSE <- nTreeResult[2, largestRSqr]
  bestRandForRSqr <- nTreeResult[3, largestRSqr]
  
  # Plot RMSE vs mtry colored by ntree
  plot1 <- ggplot(results_df, aes(x = mtry, y = RMSE, color = as.factor(ntree))) +
    geom_line() +
    geom_point() +
    labs(title = "Effect of mtry and ntree on RMSE in Random Forest",
         x = "mtry (Number of Features Tried at Each Split)",
         y = "Root Mean Squared Error (RMSE)",
         color = "Number of Trees (ntree)") +
    theme_minimal()
  
  print(plot1)  # Display plot
  
  return(list(bestRandForMAE, bestRandForRMSE, bestRandForRSqr, prediction, rfModel))
}

randomForestFeatureImportance <- function(rfModel) {
  "
  Extracts and visualizes feature importance from a random forest model.

  Parameters:
  -----------
  rfModel : randomForest
      A trained random forest model object from the randomForest package.

  Returns:
  --------
  A horizontal bar plot displaying the relative importance of features in the random forest.
  "
  
  # Extract importance scores (MeanDecreaseGini or MeanDecreaseAccuracy)
  importance <- importance(rfModel)
  
  # Convert to data frame for plotting (using MeanDecreaseGini as importance)
  importance_df <- data.frame(
    Feature = rownames(importance),
    Importance = importance[, "IncNodePurity"]
  )
  
  # Order by descending importance
  importance_df <- importance_df[order(-importance_df$Importance), ][1:100, ]
  
  # Plot horizontal bar chart
  ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
    geom_bar(stat = "identity", fill = "forestgreen") +
    coord_flip() +
    labs(title = "Random Forest Feature Importance",
         x = "Feature", y = "Mean Decrease in Gini") +
    theme_minimal()
}

