import matplotlib.pyplot as plt #Library used to create static, interactive and animated visualisations
import seaborn as sns #Library for creating attractive and informative statistical graphics
import numpy as np #library that enables fast, efficient numerical computations using powerful array operations and linear algebra tools

def pltHistogram(df, targetVariable, title, xAxis):

    """
    Plots a histogram with a kernel density estimate (KDE) for a specified variable.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data.
    targetVariable : str
        Column name in df to plot the histogram for.
    title : str
        Title of the plot.
    xAxis : str
        Label for the x-axis.

    Returns:
    --------
    None
    """

    plt.figure(figsize=(8, 4))
    sns.histplot(df[targetVariable], bins=50, kde=True)
    plt.title(title)
    plt.xlabel(xAxis)
    plt.show()

def pltLogHistogram(df, targetVariable, title, xAxis):

    """
    Plots a histogram with a kernel density estimate (KDE) for the logarithm (log1p) of a specified variable.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data.
    targetVariable : str
        Column name in df to plot the log-transformed histogram for.
    title : str
        Title of the plot.
    xAxis : str
        Label for the x-axis.

    Returns:
    --------
    None
    """

    plt.figure(figsize=(8, 4))
    sns.histplot(np.log1p(df[targetVariable]), bins=50, kde=True)
    plt.title(title)
    plt.xlabel(xAxis)
    plt.show()

def pltViolin(df, xVariable, yVariable, title, xLabel, yLabel):

    """
    Plots a violin plot with a logarithmic scale on the y-axis.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data.
    xVariable : str
        Column name for the x-axis categorical variable.
    yVariable : str
        Column name for the y-axis numerical variable.
    title : str
        Title of the plot.
    xLabel : str
        Label for the x-axis.
    yLabel : str
        Label for the y-axis.

    Returns:
    --------
    None
    """

    plt.figure(figsize=(10, 5))
    sns.violinplot(x=xVariable, y=yVariable, data=df)
    plt.yscale('log') 
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.show()

def pltBar(xVariable, yVariable, title, xLabel, yLabel):

    """
    Plots a bar chart with customized axis labels and title.

    Parameters:
    -----------
    xVariable : array-like
        Data or categories to plot on the x-axis.
    yVariable : array-like
        Numerical values to plot on the y-axis.
    title : str
        Title of the plot.
    xLabel : str
        Label for the x-axis.
    yLabel : str
        Label for the y-axis.

    Returns:
    --------
    None
    """

    plt.figure(figsize=(12, 6))
    sns.barplot(x=xVariable, y=yVariable)
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def pltBox(df, xVariable, yVariable, title, xLabel, yLabel):

    """
    Plots a box plot with logarithmic scaling on the y-axis.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data to plot.
    xVariable : str
        Column name in df to be plotted on the x-axis (categorical).
    yVariable : str
        Column name in df to be plotted on the y-axis (numerical).
    title : str
        Title of the plot.
    xLabel : str
        Label for the x-axis.
    yLabel : str
        Label for the y-axis.

    Returns:
    --------
    None
    """

    plt.figure(figsize=(10, 5))
    sns.boxplot(x=xVariable, y=yVariable, data=df)
    plt.yscale('log')  
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.show()

def pltHeatMap(df,title, xLabel, yLabel):

    """
    Plots a heatmap showing the missing values in a DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame whose missing values are to be visualized.
    title : str
        Title of the heatmap.
    xLabel : str
        Label for the x-axis.
    yLabel : str
        Label for the y-axis.

    Returns:
    --------
    None
    """
    
    plt.figure(figsize=(12,6))
    sns.heatmap(df.isnull(), cbar=True, cmap = 'viridis', yticklabels=False)
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.xticks(rotation=45, ha='right')
    plt.show()

def pltPie(labels, distributions, title, legendTitle):

    """
    Plots a pie chart with given labels and distribution values.

    Parameters:
    -----------
    labels : list of str
        The categories or names corresponding to each slice of the pie chart.
    distributions : list or array-like of numeric
        The numerical values representing the size of each slice.
    title : str
        The title displayed at the top of the pie chart.
    legendTitle : str
        The title for the legend that explains the labels.

    Functionality:
    --------------
    - Creates a pie chart with percentage values shown on slices.
    - Displays a legend on the side instead of labels directly on the slices.
    - Sets the chart title and displays the plot.
    """
    
    fig, ax = plt.subplots()
    ax.pie(distributions, startangle=90, autopct='%1.1f%%')  # autopct shows percentages on slices

    # Add legend instead of labels on slices
    ax.legend(labels, title=legendTitle, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    plt.title(title)
    plt.show()

def pltGroupedBar(df, xVariable, yVariable, hue, title, xLabel, yLabel):

    """
    Creates a grouped bar plot with annotated value labels on top of each bar.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data to plot.
    xVariable : str
        Column name in `df` to be used on the x-axis.
    yVariable : str
        Column name in `df` representing the height of the bars.
    hue : str
        Column name in `df` for grouping bars by different categories (color coding).
    title : str
        Title of the plot.
    xLabel : str
        Label for the x-axis.
    yLabel : str
        Label for the y-axis.

    The function displays a bar plot where each bar is annotated with its numeric value.
    """

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df, x=xVariable, y=yVariable, hue=hue)

    # Add value labels on top of each bar
    for p in ax.patches:
        height = p.get_height()
        if height is not None and height > 0:
            ax.annotate(f'{height:.0f}',  # format as integer
                        (p.get_x() + p.get_width() / 2, height),
                        ha='center', va='bottom',
                        fontsize=9,
                        xytext=(0, 3),  # slight offset above the bar
                        textcoords='offset points')

    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def pltScatter(x, y, title, xlabel, ylabel):

    """
    Plots a scatter plot with given x and y values.

    Parameters:
    -----------
    x : list or array-like of numeric
        The values for the x-axis.
    y : list or array-like of numeric
        The values for the y-axis.
    title : str
        The title displayed at the top of the scatter plot.
    xlabel : str
        The label for the x-axis.
    ylabel : str
        The label for the y-axis.

    Functionality:
    --------------
    - Plots a scatter plot of x vs y.
    - Sets axis labels and title, then displays the plot.
    """
    
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=10, alpha=0.7, color='teal')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.title(title)
    plt.show()