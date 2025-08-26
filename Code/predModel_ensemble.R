gradBoost <- function(trainMatrix, trainLabel, testMatrix){
  "
  Trains and predicts house prices using a Gradient Boosting Machine (GBM) model 
  with hyperparameter tuning via cross-validation.

  Parameters:
  -----------
  trainMatrix : matrix
      Feature matrix for training data (80% of full dataset).
  trainLabel : numeric vector
      Target variable (house sale prices) corresponding to trainMatrix.
  testMatrix : matrix
      Feature matrix for testing data (20% of full dataset).

  Returns:
  --------
  predictions : numeric vector
      Predicted house prices from the tuned Gradient Boosting model on the test set.
  "
  
  # Define a grid of hyperparameters to tune:
  # - n.trees: number of boosting iterations (trees)
  # - interaction.depth: max depth of each tree (controls model complexity)
  # - shrinkage: learning rate (step size reduction at each boosting iteration)
  # - n.minobsinnode: minimum number of observations in terminal nodes (controls overfitting)
  gbm_grid <- expand.grid(
    n.trees = c(100, 200),
    interaction.depth = c(4, 6, 8),
    shrinkage = c(0.01, 0.05, 0.1),
    n.minobsinnode = c(10, 20)
  )
  
  # Set up 5-fold cross-validation to tune hyperparameters, with verbosity and parallel processing enabled
  gbm_control <- trainControl(
    method = "cv",
    number = 5,
    verboseIter = TRUE,
    allowParallel = TRUE
  )
  
  # Train the GBM model using caret’s train function with tuning grid and cross-validation
  gbm_tuned <- train(
    x = trainMatrix,
    y = trainLabel,
    method = "gbm",
    trControl = gbm_control,
    tuneGrid = gbm_grid,
    verbose = TRUE
  )
  
  # Print best combination of hyperparameters found by cross-validation
  print(gbm_tuned$bestTune)
  
  # Plot model performance for different hyperparameters
  print(plot(gbm_tuned))
  
  # Generate predictions on the test data using the best tuned model
  predictions <- predict(gbm_tuned, testMatrix)
  
  return(list(predictions, gbm_tuned))
}

gradientBoostingFeatureImportance <- function(gbmModel) {
  # Extract importance using gbm's relative influence
  imp <- summary(gbmModel, plotit = FALSE)  # summary.gbm gives rel. influence
  
  # Rename columns for clarity
  colnames(imp) <- c("Feature", "RelativeImportance")
  
  # Sort by importance
  imp <- imp[order(-imp$RelativeImportance), ][1:100, ]
  
  # Plot
  ggplot(imp, aes(x = reorder(Feature, RelativeImportance), 
                  y = RelativeImportance)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    coord_flip() +
    labs(title = "Gradient Boosting Feature Importance",
         x = "Feature",
         y = "Relative Importance") +
    theme_minimal()
}

xgboost <- function(trainMatrix, trainLabel, testMatrix){
  "
  Trains and predicts house prices using the XGBoost model with hyperparameter tuning via cross-validation.

  Parameters:
  -----------
  trainMatrix : matrix
      Feature matrix for training data (80% of the dataset).
  trainLabel : numeric vector
      Target variable (house sale prices) corresponding to trainMatrix.
  testMatrix : matrix
      Feature matrix for testing data (20% of the dataset).

  Returns:
  --------
  predictions : numeric vector
      Predicted house prices from the tuned XGBoost model on the test set.
  "
  
  # Define a grid of hyperparameters to tune:
  # - nrounds: number of boosting rounds (trees)
  # - max_depth: max depth of a tree (controls model complexity)
  # - eta: learning rate (step size shrinkage)
  # - gamma: minimum loss reduction required to make a split
  # - colsample_bytree: fraction of columns sampled for each tree (feature subsampling)
  # - min_child_weight: minimum sum of instance weight needed in a child (regularization)
  # - subsample: fraction of observations used for growing each tree (row subsampling)
  xgb_grid <- expand.grid(
    nrounds = c(100, 200),
    max_depth = c(6, 8, 10),
    eta = c(0.01, 0.05, 0.1),
    gamma = c(0, 1),
    colsample_bytree = c(0.6, 0.7),
    min_child_weight = 1,
    subsample = c(0.8, 0.9)
  )
  
  # Set up 5-fold cross-validation with verbosity and parallel processing enabled
  xgb_control <- trainControl(
    method = "cv",
    number = 5,
    verboseIter = TRUE,
    allowParallel = TRUE
  )
  
  # Train the XGBoost model with caret’s train function using the tuning grid and cross-validation
  xgb_tuned <- train(
    x = trainMatrix,
    y = trainLabel,
    method = "xgbTree",
    trControl = xgb_control,
    tuneGrid = xgb_grid,
    verbose = TRUE
  )
  
  # Print best hyperparameters found by tuning
  print(xgb_tuned$bestTune)
  
  # Plot model performance for different hyperparameter combinations
  print(plot(xgb_tuned))
  
  # Generate predictions on the test data using the best tuned model
  predictions <- predict(xgb_tuned, testMatrix)
  
  return(list(predictions, xgb_tuned))
}

xgboostFeatureImportance <- function(xgbModel, trainData) {
  importance_matrix <- xgb.importance(feature_names = colnames(trainData), model = xgbModel)
  print(importance_matrix)  # Optional: see the table
  
  # Plot feature importance
  xgb.plot.importance(importance_matrix)
}
