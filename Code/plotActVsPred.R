plotActVsPred <- function(testData, predictions){
  
  #Function to plot the actual and predicted results of the best regression model
  
  #Input Parameter: - testData: table containing the data to be used to test the machine learning 
  #                              models
  #                 - predictions: data frame with the sales price predicted by the best 
  #                                regression model
  
  multLinComp <- data.frame(actual = testData$SalePrice, predicted = as.numeric(predictions))
  
  ggplot(multLinComp, aes(x = actual, y = predicted)) +
    geom_point(color = "red", size = 1) +         
    geom_abline(slope = 1, intercept = 0, color = "blue") +  # y = x line
    labs(title = "Linear Plot of the Actual vs Predicted Sales Price of the Best Regression Model",
         x = "Actual Sales Price ($)", y = "Predicted Sales Price ($)") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
}