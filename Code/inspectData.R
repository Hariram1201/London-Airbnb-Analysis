inspectData <- function(dataset) {
  
  # Function used to analyse the data - outputs the structure of the data 
  # and the summary statistics of each numerical column in the data.
  #
  # Parameters:
  #   dataset : data.frame or tibble
  #     The table containing the data to be inspected.
  #
  # Returns:
  #   NULL (prints output to the console)
  
  # Inspects the structure of the data
  print(str(dataset))
  
  # Gets summary statistics of the numerical columns
  print(summary(dataset))
}
