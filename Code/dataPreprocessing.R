source("plotsAndGraphs.R")

cleanData <- function(dataset, targetVar, title, xTitle){
  
  "
  Cleans the dataset by removing outliers based on percentile thresholds, and displays boxplots
  before and after cleaning to visualize the effect.
  
  Parameters:
  -----------
  dataset : data.frame
      The dataset containing the data to clean.
  targetVar : character
      The name of the target variable (column) to analyze and clean.
  title : character
      The title used for the boxplots.
  xTitle : character
      The label for the x-axis on the boxplots.
  
  Returns:
  --------
  data.frame
      The cleaned dataset with outliers removed.
  "
  
  # Plot box and whisker plot of original data to visually identify outliers
  boxPlot(dataset, dataset[[targetVar]], title, xTitle)
  
  # Calculate the 1st and 99th percentiles of the target variable
  lower_percentile <- quantile(dataset[[targetVar]], 0.01)
  upper_percentile <- quantile(dataset[[targetVar]], 0.99)
  
  # Filter dataset to keep only rows where target variable is within bounds
  datasetCleaned <- dataset %>%
    filter(.data[[targetVar]] >= lower_percentile & .data[[targetVar]] <= upper_percentile)
  
  # Plot box and whisker plot of cleaned data to show effect of outlier removal
  boxPlot(datasetCleaned, datasetCleaned[[targetVar]], title, xTitle)
  
  # Return the cleaned dataset
  return(datasetCleaned)
}

changeNA <- function(dataset){
  
  "
  Replaces all NA values in the dataset: numeric columns get 0, categorical/non-numeric get 'none'.
  
  Parameters:
  -----------
  dataset : data.frame
      The dataset to process for NA value replacement.
  
  Returns:
  --------
  data.frame
      The dataset with NAs replaced: numeric NAs as 0, categorical NAs as 'none'.
  "
  
  for (col_name in colnames(dataset)) {
    
    # Check if the column is numeric
    if (is.numeric(dataset[[col_name]])) {
      # Replace NA with 0 in numeric columns
      dataset[[col_name]][is.na(dataset[[col_name]])] <- 0
    } else {
      # Replace NA with 'none' in non-numeric columns
      dataset[[col_name]][is.na(dataset[[col_name]])] <- 'none'
    }
  }
  
  return(dataset)
}
