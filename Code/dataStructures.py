#==============================
#Dictionary
#==============================

def createDictionary(df, key, pair, filePath):

    """
    Creates a dictionary of grouped DataFrame columns based on a key and saves each group to a CSV file.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing the data to be grouped.

    key : str
        The name of the column in df to group by 

    pair : str
        The name of the column in df whose data will be stored in the dictionary and saved (e.g., 'comments').

    filePath : str
        The directory path and file name prefix where the CSV files will be saved.
        Each group will be saved with a filename constructed as "{filePath}{key_value}.csv".

    Returns:
    --------
    keys : list
        A list of unique keys found in the DataFrame (the unique values from the 'key' column).

    grouped : dict
        A dictionary where each key corresponds to a unique key value from the DataFrame,
        and each value is a pandas Series containing the specified column (pair) for that group.
    """

    #List to hold all unique keys
    keys = []

    #Dictionary to hold one column per group
    grouped = {}

    #Group the database based on the key
    groups = df.groupby(key)

    for key, group in groups:
        keys.append(key)

        # Store only the specified column (pair) for this group
        grouped[key] = group[pair]

        #Save just that column to CSV
        filename = f"{filePath}{key}.csv"
        grouped[key].to_csv(filename, index=False)
        print(f"Saved reviews for listing {key} to {filename}")

    return keys, grouped

def searchDict(input, dictionary):

    """
    Searches for a given input value within the values of a dictionary and returns the corresponding key.

    Parameters:
    -----------
    input_value : any
        The value to search for within the dictionary’s values.

    dictionary : dict
        A dictionary where each key maps to a list (or iterable) of items.

    Returns:
    --------
    key : any or None
        The key in the dictionary whose value list contains the input_value.
        Returns None if the input_value is not found in any of the dictionary’s values.
    """
    
    #Finds the input in the value of the dictionary and returns the key
    for key, value in dictionary.items():
        if input in value:
            return key
    return None

def searchDictValue(input, dictionary):

    """
    Searches for a given input key within a dictionary and returns the corresponding value.

    Parameters:
    -----------
    input_value : any
        The key to search for in the dictionary.

    dictionary : dict
        A dictionary where each key maps to a value.

    Returns:
    --------
    value : any or None
        The value corresponding to the input_key.
        Returns None if the input_key is not found in the dictionary.
    """
    
    #Finds the input in the value of the dictionary and returns the key
    for key, value in dictionary.items():
        if input == key:
            return value
    return None

#==============================
#DataFrame
#==============================

def dropColumn(df, targetColumn):

    """
    Drops the specified column from the DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame from which the column will be removed.
    targetColumn : str
        The name of the column to drop.

    Returns:
    --------
    None
        The DataFrame is modified in place.
    """

    #Find the index of the target column
    colIndex = df.columns.get_loc(targetColumn)

    #Get the exact column name by index
    colName = df.columns[colIndex]

    #Drop the column from the DataFrame inplace
    df.drop(colName, axis=1, inplace=True)