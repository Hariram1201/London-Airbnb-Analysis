import pandas as pd #Library used for data manipulation and analysis 

def dataInspection(dataset):
    """
    Provides an initial analysis of the input dataset to aid in understanding its structure and content.

    Parameters:
    -----------
    dataset : pandas.DataFrame
        The dataset to be analyzed, expected as a 2D DataFrame containing rows and columns.

    Functionality:
    --------------
    - Displays a concise summary of the dataset including index range, column names, data types, 
      number of non-null entries, and memory usage using `dataset.info()`.
    - Prints the first 5 rows of the dataset for a quick preview.
    - Generates descriptive statistics of all numerical columns such as count, mean, standard deviation,
      min, quartiles, and max values using `dataset.describe()`.
    - Saves the numerical summary statistics to a CSV file for further reference or reporting.

    Output:
    -------
    - Console output: Dataset info and first 5 rows.
    - CSV file: summary statistics saved to a specified path.
    """

    #Provides a concise summary of the dataset:
    # - Index Range: The range of the dataset's index
    # - Column Details: Each column's name, data type (dtype), and the number of non-null entries
    # - Memory Usage: The total memory consumed by the dataset
    dataset.info()

    #Returns the first 5 rows of dataset, useful for quickly inspecting the top rows of the dataset
    print(dataset.head(5))

    #Generates descriptive statistics of the dataset's numerical columns, this inlcudes:
    # - Count: Number of non-null entries
    # - Mean: Average value
    # - Std: Standard deviation
    # - Min: Minimum value
    # - 25%: 25th percentile
    # - 50%: Median (50th percentile)
    # - 75%: 75th percentile
    # Max: Maximum value
    summaryNum = dataset.describe()
    summaryNum.to_csv('/Users/hariramgnanachandran/Documents/Data Science : Machine Learning Work/Projects/London AirBnB Analysis/Outputs/Statistical Data/summary_statistics.csv', index=True)
