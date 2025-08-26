source("plotsAndGraphs.R")

priceTrends <- function(trainDataCleaned){
  "
  Plots a series of graphs to explore how key variables relate to house sale price.
  These variables were selected based on strong correlations identified in prior analysis.

  Plots Generated:
  ----------------
  - Violin plot of Overall Quality vs. Sale Price
  - Facet plot of surface area (by floor type) vs. Sale Price
  - Linear regression plot of Above Ground Living Area vs. Sale Price
  - Two-line linear plot of Year Built and Year Remodeled vs. Sale Price
  - 3D scatter plot of Garage Area and Number of Cars vs. Sale Price

  Parameters:
  -----------
  trainDataCleaned : data.frame
      The cleaned dataset used for visual analysis and model training. Must include
      variables such as OverallQual, SalePrice, GrLivArea, BsmtFinSF1, X1stFlrSF,
      X2ndFlrSF, YearBuilt, YearRemodAdd, GarageArea, and GarageCars.

  Returns:
  --------
  None
      The function prints a series of plots but does not return a value.
  "
  
  # Ensure OverallQual is treated as a categorical variable for plotting
  trainDataCleaned$OverallQual <- as.factor(trainDataCleaned$OverallQual)
  
  # Remove any rows with missing values to avoid plotting issues
  trainDataCleaned <- na.omit(trainDataCleaned)
  
  # Plot 1: Violin plot of Overall Quality vs Sale Price
  violin(
    trainDataCleaned,
    "OverallQual", "SalePrice",
    "Violin Plot of Overall Quality against Sale Price",
    "Overall Quality", "Sale Price ($)"
  )
  
  # Prepare data for Plot 2: Reshape floor area columns into long format
  trainDataCleaned_long <- trainDataCleaned %>%
    select(SalePrice, BsmtFinSF1, `X1stFlrSF`, `X2ndFlrSF`) %>%
    pivot_longer(
      cols = c(BsmtFinSF1, `X1stFlrSF`, `X2ndFlrSF`),
      names_to = "FloorType",
      values_to = "Area"
    )
  
  # Plot 2: Facet plot of floor surface areas vs Sale Price
  facet(
    trainDataCleaned_long,
    "Area", "SalePrice", "FloorType",
    "Surface Area vs Sale Price by Floor Type",
    "Area (sq ft)", "Sale Price ($)"
  )
  
  # Plot 3: Linear plot of Above Ground Living Area vs Sale Price
  linear(
    trainDataCleaned,
    "GrLivArea", "SalePrice",
    "Linear Plot of Above Ground Living Area against Sale Price",
    "Above Ground Living Area (Square Feet)", "Sale Price ($)"
  )
  
  # Prepare data for Plot 4: Reshape year columns into long format
  trainDataCleaned_long <- trainDataCleaned %>%
    select(SalePrice, YearBuilt, YearRemodAdd) %>%
    pivot_longer(
      cols = c(YearBuilt, YearRemodAdd),
      names_to = "YearType",
      values_to = "Year"
    )
  
  # Plot 4: Two-line linear plot of Year Built vs Year Remodeled vs Sale Price
  twoLinear(
    trainDataCleaned_long,
    "Year", "SalePrice", "YearType",
    "Sale Price Trends: Year Built vs. Year Remodeled",
    "Year", "Sale Price ($)", "Year Type"
  )
  
  # Plot 5: 3D scatter plot of Garage Area and Garage Cars vs Sale Price
  threeDimLin(
    trainDataCleaned,
    "GarageArea", "GarageCars", "SalePrice",
    "Garage Area & Number of Cars vs Sale Price",
    "Garage Area (sq ft)", "Number of Cars", "Sale Price ($)"
  )
}
