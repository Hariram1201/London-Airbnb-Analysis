library(ggplot2) #Collection of packages to create static, interactive and customisable plots

interceptOnlyModel <- function(trainData, testData, targetVar){
  "
  Fits an intercept-only linear regression model using the training data and 
  generates predictions on the test data. This model ignores all predictors 
  and predicts the mean of the target variable for every observation.

  Parameters:
  -----------
  trainData : data.frame
      The training dataset (typically 80% of original data).
  
  testData : data.frame
      The test dataset (typically 20% of original data).
  
  targetVar : character
      The name of the target variable (as a string) to predict.

  Returns:
  --------
  list containing:
    - predictions : numeric vector
        A vector containing predicted values for the target variable in the test set, 
        based only on the mean value from the training set.
    - model : lm object
        The fitted intercept-only linear regression model object, which can be used for 
        further analysis such as extracting coefficients or plotting feature importance.
  "
  
  # Dynamically create formula for intercept-only model
  formula <- as.formula(paste(targetVar, "~ 1"))
  
  # Fit the intercept-only linear regression model to training data
  model <- lm(formula, data = trainData)
  
  # Generate predictions on the test data using the trained model
  predictions <- predict(model, newdata = testData)
  
  # Return both predicted values and the fitted model object
  return(list(predictions, model))
}

multLinModel <- function(trainData, testData, targetVar){
  "
  Fits a multiple linear regression model using all available predictors 
  in the training dataset to predict a specified target variable. 
  Generates predictions on the test dataset.

  Parameters:
  -----------
  trainData : data.frame
      The dataset (typically 80% of original data) used to train the model.
  
  testData : data.frame
      The dataset (typically 20% of original data) used to test the model.
  
  targetVar : character
      The name of the target variable (as a string) to predict.

  Returns:
  --------
  list containing:
    - predictions : numeric vector
        A vector containing the predicted values of the target variable 
        in the test dataset using the multiple linear regression model.
    - model : lm object
        The fitted multiple linear regression model object, which can be used for 
        further analysis such as examining coefficients or diagnostic plots.
  "
  
  # Dynamically create a formula for multiple linear regression 
  formula <- as.formula(paste(targetVar, "~ ."))
  
  # Fit the multiple linear regression model on training data
  model <- lm(formula, data = trainData)
  
  # Generate predictions on the test data using the trained model
  predictions <- predict(model, newdata = testData)
  
  # Return both predicted values and the fitted model object
  return(list(predictions, model))
}

linearFeatureImportance <- function(lmModel){
  "
  Calculates and visualises feature importance for a linear regression model 
  based on the absolute values of the model's coefficients. The function 
  displays a bar chart of the top 100 most influential features.

  Parameters:
  -----------
  lmModel : lm
      A trained linear regression model (created using lm()).

  Returns:
  --------
  A horizontal bar plot displaying the top 100 most important features 
  based on the absolute magnitude of their coefficients.
  "
  
  # Extract coefficient summary statistics from the model
  coefs <- summary(lmModel)$coefficients
  
  # Build a data frame containing feature names and their absolute coefficient values
  importance_df <- data.frame(
    Feature = rownames(coefs),
    Importance = abs(coefs[, "Estimate"])
  )
  
  # Remove the intercept term, as it is not a feature
  importance_df <- importance_df[importance_df$Feature != "(Intercept)", ]
  
  # Select the top 100 most important features (by absolute coefficient value)
  top_features <- importance_df[order(-importance_df$Importance), ][1:100, ]
  
  # Plot a horizontal bar chart of top features
  ggplot(top_features, aes(x = reorder(Feature, Importance), y = Importance)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    coord_flip() +
    labs(title = "Top 100 Linear Regression Feature Importances",
         x = "Feature", y = "Absolute Coefficient") +
    theme_minimal()
}

ridgeReg <- function(trainData, testData, targetVar){
  "
  Fits a Ridge Regression (L2 regularisation) model to predict target variable 
  using training data, then makes predictions on the test data.

  Ridge regression reduces overfitting by penalising large coefficients, which 
  helps improve model generalisation.

  Parameters:
  -----------
  trainData : data.frame
      The dataset (80% split) used to train the model.

  testData : data.frame
      The dataset (20% split) used to test the model.

  targetVar : character
      The name of the target variable to be predicted.

  Returns:
  --------
  list containing:
    - predictions : numeric vector
        The predicted values of the target variable on the test data, 
        generated by the ridge regression model.
    - model : cv.glmnet object
        The fitted ridge regression model object (with cross-validation), 
        which can be used for further analysis such as coefficient extraction 
        or plotting.
  "
  
  # Extract feature matrix and target vector for training data
  XTrain <- as.matrix(trainData[, -which(names(trainData) == targetVar)]) 
  yTrain <- trainData[[targetVar]]
  
  # Extract feature matrix and target vector for test data
  XTest <- as.matrix(testData[, -which(names(testData) == targetVar)]) 
  yTest <- testData[[targetVar]]
  
  # Perform cross-validation to select the optimal lambda for ridge regression
  cvRidge <- cv.glmnet(XTrain, yTrain, alpha = 0)
  
  # Plot the cross-validation results to visualise performance vs lambda
  plot(cvRidge)
  mtext("Cross-Validation Curve for Ridge Regression", side = 3, line = 2.5, cex = 1.2, font = 2)
  
  # Fit ridge regression model using cross-validated lambda values
  cvRidgeModel <- cv.glmnet(XTrain, yTrain, alpha = 0)
  
  # Extract the lambda that gives the minimum cross-validation error
  bestLambda <- cvRidgeModel$lambda.min
  
  # Make predictions on the test set using the optimal lambda
  predictions <- predict(cvRidgeModel, XTest, s = bestLambda)
  
  # Return predictions and model object
  return(list(predictions, cvRidgeModel))
}

lassoReg <- function(trainData, testData, targetVar){
  "
  Fits a Lasso Regression (L1 regularization) model to predict the target variable 
  using training data, then makes predictions on the test data.

  Lasso regression reduces overfitting by penalizing the absolute values of 
  coefficients and can shrink some coefficients entirely to zero, effectively 
  performing feature selection.

  Parameters:
  -----------
  trainData : data.frame
      The dataset (80% split) used to train the model.

  testData : data.frame
      The dataset (20% split) used to test the model.

  targetVar : character
      The name of the target variable to be predicted.

  Returns:
  --------
  list containing:
    - predictions : numeric vector
        The predicted values of the target variable on the test data, 
        generated by the Lasso regression model.
    - model : cv.glmnet object
        The fitted Lasso regression model object (with cross-validation), 
        which can be used for further analysis such as coefficient extraction 
        or plotting.
  "
  
  # Extract feature matrix and target vector from training data
  XTrain <- as.matrix(trainData[, -which(names(trainData) == targetVar)]) 
  yTrain <- trainData[[targetVar]]
  
  # Extract feature matrix and target vector from testing data
  XTest <- as.matrix(testData[, -which(names(testData) == targetVar)]) 
  yTest <- testData[[targetVar]]
  
  # Perform cross-validation to choose the best lambda (regularization strength)
  cvLasso <- cv.glmnet(XTrain, yTrain, alpha = 1)
  
  # Plot cross-validation curve
  plot(cvLasso)
  mtext("Cross-Validation Curve for Lasso Regression", side = 3, line = 2.5, cex = 1.2, font = 2)
  
  # Fit Lasso model using cross-validation
  cvLassoModel <- cv.glmnet(XTrain, yTrain, alpha = 1)
  
  # Identify the lambda that minimizes cross-validation error
  bestLambda <- cvLassoModel$lambda.min
  
  # Generate predictions on test data using the optimal lambda
  predictions <- predict(cvLassoModel, XTest, s = bestLambda)
  
  # Return predictions and model object
  return(list(predictions, cvLassoModel))
}

glmnetFeatureImportance <- function(cvModel) {
  "
  Extracts and visualises the top 100 feature importances from a glmnet cross-validated model,
  based on the absolute values of coefficients at the optimal lambda.

  Parameters:
  -----------
  cvModel : cv.glmnet object
      A trained glmnet model with cross-validation.

  Returns:
  --------
  A horizontal bar plot showing the top 100 most important features based on absolute coefficient values.
  "
  
  # Extract the lambda value that gives minimum cross-validation error
  bestLambda <- cvModel$lambda.min
  
  # Get model coefficients at the best lambda (includes intercept)
  coefs <- coef(cvModel, s = bestLambda)
  
  # Convert the sparse matrix of coefficients to a data frame with feature names and coefficients
  coefs_df <- data.frame(
    Feature = rownames(coefs),
    Coefficient = as.vector(coefs)
  )
  
  # Remove intercept as it is not a feature
  coefs_df <- coefs_df[coefs_df$Feature != "(Intercept)", ]
  
  # Compute absolute coefficient values to represent feature importance
  coefs_df$Importance <- abs(coefs_df$Coefficient)
  
  # Order features by descending importance and keep top 100
  topFeatures <- coefs_df[order(-coefs_df$Importance), ][1:100, ]
  
  # Create a horizontal bar plot of the top features by importance
  ggplot(topFeatures, aes(x = reorder(Feature, Importance), y = Importance)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    coord_flip() +
    labs(title = "Top 100 Feature Importances from glmnet Model",
         x = "Feature", y = "Absolute Coefficient") +
    theme_minimal()
}
