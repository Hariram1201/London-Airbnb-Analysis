evaluateModel <- function(testData, predictions, meanSalePrice){
  "
  Evaluates the performance of a regression model using three metrics:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - R-squared (R²)

  MAE provides an average of absolute errors and is useful for understanding 
  overall prediction accuracy. RMSE penalizes large errors more than MAE, 
  making it sensitive to outliers. R² quantifies how well the model explains 
  variance in the data.

  Parameters:
  -----------
  testData : data.frame
      Dataset containing the true values of the target variable.

  predictions : numeric vector
      Model-generated predictions for the target variable.

  meanSalePrice : numeric
      The mean value of the target variable in the training data,
      used to calculate the R² score.

  Returns:
  --------
  list(MAEValue, rmseValue, rSquared) : list
      - MAEValue: Mean Absolute Error
      - rmseValue: Root Mean Squared Error
      - rSquared: R² score (coefficient of determination)
  "
  
  # Calculate Mean Absolute Error
  MAEValue <- mean(abs(testData$SalePrice - predictions))
  
  # Calculate Root Mean Squared Error
  rmseValue <- sqrt(mean((testData$SalePrice - predictions)^2))
  
  # Calculate R-squared
  rSquared <- 1 - (sum((testData$SalePrice - predictions)^2) / 
                     sum((testData$SalePrice - meanSalePrice)^2))
  
  # Return all evaluation metrics in a list
  return(list(MAEValue, rmseValue, rSquared))
}
