from sklearn.preprocessing import StandardScaler  # Library to standardise features by removing the mean and scaling to unit variance
from sklearn.cluster import DBSCAN  # Imports density-based clustering algorithm to find clusters of varying shapes
import matplotlib.pyplot as plt  #Plotting library for creating static, interactive, and animated visualisations
import seaborn as sns  #Statistical data visualisation library built on matplotlib for attractive plots


def scaleDf(df, targetVar, clusterVar):

    """
    Scale selected features from the DataFrame after dropping rows with missing values.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame.
    targetVar : str
        Name of the first feature column to scale.
    clusterVar : str
        Name of the second feature column to scale.

    Returns:
    --------
    scaledFeatures : numpy.ndarray
        Scaled feature array suitable for clustering.
    cleanDf : pandas.DataFrame
        DataFrame with non-missing values in the selected columns.
    """

    #Remove rows with NaNs in the specified columns
    cleanDf = df[[targetVar, clusterVar]].dropna()

    #Standardise features for clustering algorithms sensitive to scale
    scaler = StandardScaler()
    scaledFeatures = scaler.fit_transform(cleanDf)

    return scaledFeatures, cleanDf

def dbscan(cleanDf, scaledFeatures, xcol, ycol):

    """
    Perform DBSCAN clustering and visualize the results.

    Parameters:
    -----------
    cleanDf : pandas.DataFrame
        DataFrame with features used for clustering.
    scaledFeatures : numpy.ndarray
        Scaled feature array for clustering.
    xcol : str
        Column name for the x-axis in the scatter plot.
    ycol : str
        Column name for the y-axis in the scatter plot.

    Returns:
    --------
    None
        Displays a scatter plot of clusters.
    """

    #Run DBSCAN clustering algorithm with specified parameters
    db = DBSCAN(eps=1.5, min_samples=2)
    labels = db.fit_predict(scaledFeatures)

    #Add cluster labels to the DataFrame
    cleanDf = cleanDf.copy()
    cleanDf['cluster'] = labels

    #Plot clusters with seaborn scatterplot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=cleanDf, x=xcol, y=ycol, hue='cluster', palette='tab10', s=100)
    plt.title(f'DBSCAN Clustering: {xcol} vs {ycol}')
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.legend(title='Cluster')
    plt.grid(True)
    plt.show()
