multiCollinarity <- function(trainDataCleaned){
  "
  Plots a series of graphs to explore multicollinearity between variables that may influence one another.

  Plots Generated:
  ----------------
  - Violin plot of Total Rooms Above Ground vs. Ground Living Area
  - Linear regression plot of Year Built vs. Garage Year Built
  - Linear regression plot of Total Basement Area vs. First Floor Surface Area
  - Violin plot of Full Baths vs. Ground Living Area

  Parameters:
  -----------
  trainDataCleaned : data.frame
      The cleaned dataset used for visual analysis. Must include variables such as
      TotRmsAbvGrd, GrLivArea, YearBuilt, GarageYrBlt, TotalBsmtSF, X1stFlrSF, and FullBath.

  Returns:
  --------
  None
      The function prints a series of plots but does not return a value.
  "
  
  # Ensure TotRmsAbvGrd is treated as a categorical variable
  trainDataCleaned$TotRmsAbvGrd <- as.factor(trainDataCleaned$TotRmsAbvGrd)
  
  # Remove rows with missing values to ensure clean plotting
  trainDataCleaned <- na.omit(trainDataCleaned)
  
  # Plot 1: Violin plot of Total Rooms Above Ground vs Ground Living Area
  violin(
    trainDataCleaned,
    "TotRmsAbvGrd", "GrLivArea",
    "Violin Plot of TotRmsAbvGrd against GrLivArea",
    "Total Rooms above Ground", "Ground Living Area"
  )
  
  # Remove rows with invalid garage year built (e.g., 0)
  trainDataCleaned <- trainDataCleaned %>% filter(GarageYrBlt != 0)
  
  # Plot 2: Linear regression plot of Year Built vs Garage Year Built
  linear(
    trainDataCleaned,
    "YearBuilt", "GarageYrBlt",
    "Linear Plot of Year Built against Garage Year Built",
    "Year Built", "Garage Year Built"
  )
  
  # Plot 3: Linear regression plot of Total Basement Area vs First Floor Area
  linear(
    trainDataCleaned,
    "TotalBsmtSF", "X1stFlrSF",
    "Linear Plot of Total Basement Surface Area against First Floor Surface Area",
    "Total Basement Surface Area (Square Feet)", "First Floor Surface Area (Square Feet)"
  )
  
  # Ensure FullBath is treated as a categorical variable
  trainDataCleaned$FullBath <- as.factor(trainDataCleaned$FullBath)
  
  # Plot 4: Violin plot of Full Baths vs Ground Living Area
  violin(
    trainDataCleaned,
    "FullBath", "GrLivArea",
    "Violin Plot of FullBath against GrLivArea",
    "Number of Full Baths", "Ground Living Area"
  )
}
