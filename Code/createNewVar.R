createNewVar <- function(trainDataCleaned){
  "
  Creates new features from the existing cleaned dataset to potentially uncover additional patterns
  related to the sale price of the house.

  Parameters:
  -----------
  trainDataCleaned : data.frame
      The cleaned dataset used for training machine learning models.

  Returns:
  --------
  trainDataCleaned : data.frame
      The original dataset extended with newly created variables that may improve model performance.
  "
  
  # Replace GarageCars value of 0 with NA to avoid division by zero in efficiency calculation
  trainDataCleaned$GarageCars[trainDataCleaned$GarageCars == 0] <- NA
  
  # Calculate garage efficiency as garage area divided by number of cars the garage can hold
  trainDataCleaned$GarageEff <- trainDataCleaned$GarageArea / trainDataCleaned$GarageCars
  
  # Create an overall score by multiplying overall condition and overall quality of the house
  trainDataCleaned$OverallScore <- trainDataCleaned$OverallCond * trainDataCleaned$OverallQual
  
  # Calculate total porch area by summing different porch types
  trainDataCleaned$totalPorch <- trainDataCleaned$OpenPorchSF + trainDataCleaned$EnclosedPorch +
    trainDataCleaned$ScreenPorch + trainDataCleaned$`X3SsnPorch`
  
  # Calculate the age of the house as difference between current year (2025) and year built
  trainDataCleaned$HouseAge <- 2025 - trainDataCleaned$YearBuilt
  
  # Calculate years since last remodel similarly as difference between 2025 and year remodeled
  trainDataCleaned$YearSinceRemodel <- 2025 - trainDataCleaned$YearRemodAdd
  
  # Categorize month sold into seasons for potential seasonal sale effects
  trainDataCleaned$SeasonSold <- ifelse(trainDataCleaned$MoSold %in% c(12, 1, 2), 'Winter',
                                        ifelse(trainDataCleaned$MoSold %in% c(3, 4, 5), 'Spring',
                                               ifelse(trainDataCleaned$MoSold %in% c(6, 7, 8), 'Summer',
                                                      'Autumn')))
  
  # Calculate average room size as lot area divided by total rooms above ground
  trainDataCleaned$LotPerRoom <- trainDataCleaned$LotArea / trainDataCleaned$TotRmsAbvGrd
  
  return(trainDataCleaned)
}
