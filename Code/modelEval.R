modelEval <- function(modelName, model, MAE, RMSE, R_Squared, filePath){
  
  # Function to display a table comparing the different evaluation metrics, a bar chart comparing the 
  # MAE and RMSE of the different machine learning models, as well as a scatter graph of the actual
  # vs predicted results for the best model
  #
  #Input Parameter: - modelName: array storing the different machine learning model names
  #                 - model: array storing the models for different machine learning models
  #                 - MAE: array storing the MAE values for the different machine learning models
  #                 - RMSE: array storing the RMSE values for the different machine learning models
  #                 - R^Squared: array storing the R^2 values for the different machine learning models
  
  modelResults <- data.frame(modelName, MAE, RMSE, R_Squared)
  
  # Display the table nicely
  kable(modelResults, caption = "Comparison of Model Performance Metrics")
  
  # Create a data frame for plotting
  errorData <- data.frame(
    Model = rep(modelName, 2),
    ErrorType = rep(c("MAE", "RMSE"), each = length(modelName)),
    Value = c(MAE, RMSE)
  )
  
  # Plot the bar chart
  ggplot(errorData, aes(x = Model, y = Value, fill = ErrorType)) +
    geom_bar(stat = "identity", position = "dodge") +
    labs(y = "Error Value") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    scale_fill_manual(values = c("MAE" = "skyblue", "RMSE" = "orange")) +
    theme_minimal()
  
  #Determines the best machine learning model
  bestModel <- which.max(R_Squared)
  
  #Calls a function to plot the actual and predicted results of the best model
  if (bestModel == 1){
    plotActVsPred(testDataSplit, interceptPredictions)
    saveRDS(model[[bestModel]], paste0(filePath, "final_model_intercept.rds"))
  } else if (bestModel == 2){
    plotActVsPred(testDataSplit, multLinPredictions)
    saveRDS(model[[bestModel]], paste0(filePath, "final_model_multLin.rds"))
  } else if (bestModel == 3){
    plotActVsPred(testDataSplit, ridgePredictions)
    saveRDS(model[[bestModel]], paste0(filePath, "final_model_ridge.rds"))
  } else if (bestModel == 4){
    plotActVsPred(testDataSplit, lassoPredictions)
    saveRDS(model[[bestModel]], paste0(filePath, "final_model_lasso.rds"))
  } else if (bestModel == 5){
    plotActVsPred(testDataSplit, decTreePredictions)
    saveRDS(model[[bestModel]], paste0(filePath, "final_model_decTree.rds"))
  } else if (bestModel == 6){
    plotActVsPred(testDataSplit, randForPredictions)
    saveRDS(model[[bestModel]], paste0(filePath, "final_model_randFor.rds"))
  } else if (bestModel == 7){
    plotActVsPred(testDataSplit, gradBoostPredictions)
    saveRDS(model[[bestModel]], paste0(filePath, "final_model_gradBoost.rds"))
  } else{
    plotActVsPred(testDataSplit, XGBoostPredictions)
    saveRDS(model[[bestModel]], paste0(filePath, "final_model_xgBoost.rds"))
  }
}