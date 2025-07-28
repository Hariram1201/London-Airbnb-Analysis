import pandas as pd #Library used for data manipulation and analysis 
import matplotlib.pyplot as plt #Library used to create static, interactive and animated visualisations
import seaborn as sns #Library for creating attractive and informative statistical graphics

from statsmodels.stats.outliers_influence import variance_inflation_factor #Library to calculate the VIF for multicollinearity detection
from statsmodels.tools.tools import add_constant #Library adds a column of 1s to your data, which is the intercept term used in regression models.

def corrMatrix(df, numericalCols):

    """
    Plot a correlation matrix heatmap for the specified numerical columns in a DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    numericalCols : list of str
        List of column names (numerical variables) to include in the correlation matrix.

    Returns:
    --------
    None
        Displays a heatmap plot of the correlation matrix with annotated correlation coefficients.
    """

    #Calculate correlation matrix
    correlation_matrix = df[numericalCols].corr()

    #Plots correlation matrix
    plt.figure(figsize=(12, 12))
    sns.heatmap(correlation_matrix, annot=True, annot_kws={"size":8}, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()

def pairplots(df, numericalCols):

    """
    Generate and display pairplots for the specified numerical columns in a DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    numericalCols : list of str
        List of numerical column names to include in the pairplot.

    Returns:
    --------
    None
        Displays pairwise scatterplots and histograms for the specified columns.
    """

    #Create and plots pairwise scatterplots and histograms
    sns.pairplot(df[numericalCols])
    plt.show()

def detMulticollinearity(df, numericalCols):

    """
    Calculate and display the Variance Inflation Factor (VIF) for specified numerical features to detect multicollinearity.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the dataset.
    numericalCols : list of str
        List of numerical column names for which to calculate VIF.

    Returns:
    --------
    None
        Prints a DataFrame showing each feature with its corresponding VIF value.
        A high VIF indicates potential multicollinearity issues among the predictors.
    """
    
    # Prepare data by dropping missing values and adding intercept column for VIF calculation
    X = df[numericalCols].dropna()
    X = add_constant(X)

    # Create DataFrame with feature names and their corresponding VIF values
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    print(vif_data)