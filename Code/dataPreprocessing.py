import pandas as pd #Library used for data manipulation and analysis 
import seaborn as sns #Library for creating attractive and informative statistical graphics
import matplotlib.pyplot as plt #Library used to create static, interactive and animated visualisations

from dataStructures import dropColumn

def handleEmptyCol(dataset):

    """
    Removes columns from the dataset that contain only missing (NaN) values.

    Parameters:
    -----------
    dataset : pandas.DataFrame
        The input DataFrame to be processed.

    Returns:
    --------
    pandas.DataFrame
        The DataFrame with any entirely empty columns removed.
    """

    #Removes any empty columns from the dataset
    for col in dataset.columns:
        if dataset[col].isna().all():
            dropColumn(dataset, col)

    return dataset

def handleMissing(dataset):

    """
    Handle missing values in a dataset by imputing:
    - Numeric columns: replace NaNs with the column mean.
    - Categorical columns: replace NaNs with the string 'Unknown'.

    Parameters:
    -----------
    dataset : pandas.DataFrame
        The input DataFrame containing missing values.

    Returns:
    --------
    None
        The function modifies the DataFrame in place.
    """

    for col in dataset.columns:
        #Replace missing values in numeric columns with the mean of that column
        if dataset[col].dtype in ['float64', 'int64']:
            dataset[col].fillna(dataset[col].mean(), inplace=True)
        #Replace missing values in non-numeric (categorical) columns with 'Unknown'
        else:
            dataset[col].fillna('Unknown', inplace=True)

    return dataset

def convertMoney(df, targetVariable):

    """
    Convert a monetary column with currency symbols and commas to a numeric float type.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    targetVariable : str
        The name of the column to convert from string to float.

    Returns:
    --------
    None
        The function modifies the DataFrame in place.
    """

    #Remove dollar signs and commas from the target column
    df[targetVariable] = df[targetVariable].replace({'\$': '', ',': ''}, regex=True)

    #Convert the cleaned string values to numeric, coercing errors to NaN
    df[targetVariable] = pd.to_numeric(df[targetVariable], errors='coerce')

    return df

def convertPercent(df, targetVariable):

    """
    Convert percentage strings in a column to decimal float values.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    targetVariable : str
        The name of the column to convert from percentage strings to decimals.

    Returns:
    --------
    pandas.DataFrame
        The DataFrame with the converted column.
    """

    #Remove percentage signs from the target column
    df[targetVariable] = df[targetVariable].replace({'%': ''}, regex=True)

    #Convert the cleaned string values to numeric and divide by 100 to get decimals
    df[targetVariable] = pd.to_numeric(df[targetVariable], errors='coerce') / 100

    return df

def plotBox(dataset, targetVariable):

    """
    Plot a boxplot for the target variable to visualize its distribution and display summary statistics.

    Parameters:
    -----------
    dataset : pandas.DataFrame
        DataFrame containing the data.
    targetVariable : str
        The numerical column to plot.
    """

   #Calculate summary statistics for the target variable
    min_value = dataset[targetVariable].min()
    q1 = dataset[targetVariable].quantile(0.25)
    median = dataset[targetVariable].median()
    q3 = dataset[targetVariable].quantile(0.75)
    max_value = dataset[targetVariable].max()

    #Set up the figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    #Plot the boxplot for the target variable
    sns.boxplot(x=dataset[targetVariable], color='lightgray', ax=ax)

    #Customize x-axis ticks and limits
    ax.set_xticks(range(0, 1000, 100))
    ax.set_xlim(0, 1000)
    ax.grid(True)

    #Shrink the plot area to make room for summary stats
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

    #Prepare summary stats text and add to the figure
    stats_text = f'''Summary Stats:
Min: {min_value:.2f}
Q1: {q1:.2f}
Median: {median:.2f}
Q3: {q3:.2f}
Max: {max_value:.2f}
'''

    fig.text(0.75, 0.5, stats_text, ha='left', va='center', fontsize=12)

    #Show the plot
    plt.show()

def removeOutliers(dataset, targetVariable):

    """
    Remove outliers from the dataset based on the IQR (Interquartile Range) method.

    Parameters:
    -----------
    dataset : pandas.DataFrame
        DataFrame containing the original data.
    targetVariable : str
        Column name of the numerical variable to check for outliers.

    Returns:
    --------
    datasetCleaned : pandas.DataFrame
        DataFrame with outliers removed based on the IQR boundaries.
    """

    #Calculate Q1 (25th percentile) and Q3 (75th percentile)
    q1 = dataset[targetVariable].quantile(0.25)
    q3 = dataset[targetVariable].quantile(0.75)

    #Calculate the Interquartile Range (IQR)
    IQR = q3 - q1

    #Define upper and lower bounds for outliers
    upperBound = q3 + IQR
    lowerBound = q1 - IQR

    #Filter out rows outside the bounds to remove outliers
    datasetCleaned = dataset[(dataset[targetVariable] >= lowerBound) & (dataset[targetVariable] <= upperBound)]

    return datasetCleaned

def dateTimeFeatEng(dataset):

    """
    Extract date (year, month, day) and time (hour, minute, second) features from columns containing date/time data.

    Parameters:
    -----------
    dataset : pandas.DataFrame
        DataFrame containing the cleaned data with date/time columns.

    Returns:
    --------
    dataset : pandas.DataFrame
        DataFrame with new date/time feature columns added and original date columns removed.
    """

    #Define regex pattern for date format YYYY-MM-DD
    datePattern = r'^\d{4}-\d{2}-\d{2}$'

    #Define a regex pattern for time format HH-MM-SS
    timePattern = r'^\d{2}:\d{2}:\d{2}$'

    #Lists to store detected date and time columns
    dateCols = []
    timeCols = []

    #Detect date and time columns by matching regex patterns
    for col in dataset.columns:
        if dataset[col].astype(str).str.match(datePattern).any():
            dateCols.append(col)
        elif  dataset[col].astype(str).str.match(timePattern).any():
            timeCols.append(col)
    
    print('The date columns are', dateCols)
    print('The time columns are', timeCols)

    #Process each detected date column
    for col in dateCols:
        #Convert column to datetime, invalid dates coerced to NaT
        dataset[col] = pd.to_datetime(dataset[col], errors='coerce')  # Coerce invalid dates to NaT

        #Define new column names for extracted features
        yearCol = f'{col}_year'
        monthCol = f'{col}_month'
        dayCol = f'{col}_day'

        #Get index of the original date column
        colIndex = dataset.columns.get_loc(col)

        #Insert new feature columns immediately after original column
        dataset.insert(colIndex + 1, yearCol, dataset[col].dt.year)
        dataset.insert(colIndex + 2, monthCol, dataset[col].dt.month)
        dataset.insert(colIndex + 3, dayCol, dataset[col].dt.day)

        #Drops the original column from the dataset
        colName = dataset.columns[colIndex]
        dataset.drop(colName, axis=1, inplace=True)

    return dataset

def onehotEncode(dataset, ignoreCols):

    """
    One-hot encode categorical variables in the dataset, excluding specified columns.

    Parameters:
    -----------
    dataset : pandas.DataFrame
        DataFrame containing the cleaned data.
    ignoreCols : list of str
        List of column names to exclude from one-hot encoding.

    Returns:
    --------
    pandas.DataFrame
        A new DataFrame with specified categorical variables one-hot encoded.
    """

    #Create a copy of the dataset to avoid modifying the original
    datasetCopy = dataset.copy()

    #Select categorical columns excluding those in ignoreCols
    categoricalCols = datasetCopy.select_dtypes(include=['object', 'category']).columns
    categoricalCols = [col for col in categoricalCols if col not in ignoreCols]
    
    #Loop through each categorical column to encode
    for col in categoricalCols:
        #Generate one-hot encoded dummy variables with column prefix
        dummies = pd.get_dummies(datasetCopy[col], prefix=col)
        
        #Find the position of the original column
        col_index = datasetCopy.columns.get_loc(col)
        
        # Insert dummy columns immediately after the original column # Insert dummies immediately after the original column
        for i, dummy_col in enumerate(dummies.columns):
            datasetCopy.insert(col_index + 1 + i, dummy_col, dummies[dummy_col])

    return datasetCopy

def labelEncode(dataset, ordinalCols, mappingDict):
    """
    Label encodes specified ordinal categorical columns and inserts the encoded column
    immediately after the original column.
    
    Parameters:
    - dataset: DataFrame to encode
    - ordinalCols: List of column names to encode
    - mappingDict: Optional dict with custom mappings for each ordinal column
    
    Returns:
    - datasetCopy: DataFrame with new encoded columns inserted
    """
    datasetCopy = dataset.copy()

    for col in ordinalCols:
        if mappingDict and col in mappingDict:
            encoded_col = datasetCopy[col].map(mappingDict[col]).fillna(-1).astype(int)
        else:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            encoded_col = le.fit_transform(datasetCopy[col].astype(str))

        # Get the index of the original column
        col_index = datasetCopy.columns.get_loc(col)
        
        # Insert the encoded column right after the original
        datasetCopy.insert(col_index + 1, f"{col}_encoded", encoded_col)

    return datasetCopy

def mergeDataset(dataset1, dataset2, column1, column2):

    """
    Merge two datasets based on matching values in specified columns.

    Parameters:
    -----------
    dataset1 : pandas.DataFrame
        The first DataFrame to merge.
    dataset2 : pandas.DataFrame
        The second DataFrame to merge.
    column1 : str
        The column name in dataset1 to merge on.
    column2 : str
        The column name in dataset2 to merge on.

    Returns:
    --------
    pandas.DataFrame
        A new DataFrame resulting from merging dataset1 and dataset2 on the specified columns.
    """

    #Merge dataset1 and dataset2 on specified columns with inner join
    mergedDataset = pd.merge(dataset1, dataset2, left_on=column1, right_on=column2, how='inner')
    
    #Drop the duplicate column from the merged DataFrame
    mergedDataset.drop(column2, axis=1, inplace=True)    

    return mergedDataset

def mergeConsistency(mergedDataset, uniqueIdentifier):

    """
    Verify data consistency in the merged dataset.

    Parameters:
    -----------
    mergedDataset : pandas.DataFrame
        The DataFrame containing merged data from two datasets.
    uniqueIdentifier : str
        The column name that should uniquely identify each row in the merged dataset.

    Returns:
    --------
    None
        Prints information about missing values, uniqueness of the identifier, and duplicates.
    """

    #Print count of missing values in each column
    print(mergedDataset.isnull().sum())

    #Check if the uniqueIdentifier column has unique values for every row
    print(mergedDataset[uniqueIdentifier].nunique() == mergedDataset.shape[0])

    #Display any duplicate rows based on the uniqueIdentifier
    print(mergedDataset[mergedDataset.duplicated(subset=uniqueIdentifier)])

    #Print the columns and the dataset
    print("Merged dataset preview:")
    print(mergedDataset)