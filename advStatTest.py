import pandas as pd #Library used for data manipulation and analysis 
import numpy as np #Library used for handling large multi-dimensional arrays and matrices

import scipy.stats as stats #Library to import statistical functions from SciPy, such as t-tests, ANOVA, and more

from scipy.stats import chi2_contingency #Library to import chi2_contingency for performing the Chi-Square test of independence
from scipy.stats import ttest_ind #Imports the function to perform an independent two-sample t-test to compare the means of two groups

from statsmodels.stats.multicomp import pairwise_tukeyhsd #Imports the function to conduct Tukey's HSD test for multiple pairwise comparisons.

def anova(df, tarVar, testVar):

    """
    Perform a one-way ANOVA (Analysis of Variance) test to assess whether the 
    means of a numerical target variable differ significantly across groups 
    defined by a categorical variable.

    This function tests the null hypothesis that all group means are equal 
    versus the alternative that at least one group mean is different. It is 
    useful for understanding whether categorical factors (e.g. room type, 
    neighbourhood) significantly influence numerical outcomes such as price.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data.

    tarVar : str
        The name of the numerical target variable (e.g., 'price').

    testVar : str
        The name of the categorical variable used to define the groups 
        (e.g., 'room_type').

    Returns:
    --------
    tuple of (float, float)
        A tuple containing:
        - ANOVA F-statistic: indicates the ratio of between-group to within-group variance.
        - p-value: indicates whether the group means are statistically significantly different 
                   (typically p < 0.05 denotes significance).
    """
    #Group the DataFrame by the categorical variable (testVar) and extract the values of the target numerical variable 
    #(tarVar) for each group to prepare for ANOVA.
    groups = [group[tarVar].values for name, group in df.groupby(testVar)]

    #Run one-way ANOVA
    f_stat, p_val = stats.f_oneway(*groups)

    print(f"ANOVA F-statistic: {f_stat:.3f}, p-value: {p_val:.3f}")

    return f_stat, p_val

def tTest(df, tarVar, testVarCol, testVar1, testVar2):

    """
    Conduct an independent two-sample t-test to evaluate whether there is a statistically significant difference between the 
    means of a numerical variable across two distinct groups.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    tarVar : str
        The name of the numerical target variable to compare.
    testVarCol : str
        The name of the categorical column used to define the two groups.
    testVar1 : str
        The first category value to compare.
    testVar2 : str
        The second category value to compare.

    Returns:
    --------
    t_stat : float
        The calculated t-statistic, which measures the size of the difference relative to the variation in the sample data. 
        A larger absolute value of the t-statistic indicates a greater difference between group means.
    p_val : float
        The p-value indicating the significance of the difference between the two groups. A p-value less than 0.05 typically 
        indicates a statistically significant difference, meaning the group means differ beyond what might be expected by 
        chance. Conversely, a p-value greater than or equal to 0.05 suggests no significant difference.
"""

    #Filter the DataFrame to create two groups based on the specified categorical values, extracting the target numerical 
    #variable for each group.
    group1 = df[df[testVarCol] == testVar1][tarVar]
    group2 = df[df[testVarCol] == testVar2][tarVar]

    #Run independent t-test
    t_stat, p_val = stats.ttest_ind(group1, group2, nan_policy='omit')

    print(f"T-test statistic: {t_stat:.3f}, p-value: {p_val:.3f}")

    return t_stat, p_val

def chiSqrTest(df, testVar1, testVar2):

    """
    Perform a Chi-Square test to evaluate the association between two categorical variables.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    testVar1 : str
        The name of the first categorical variable.
    testVar2 : str
        The name of the second categorical variable.

    Returns:
    --------
    chi2 : float
        The Chi-square test statistic.
    p_val : float
        The p-value indicating the significance of the association.
    """

    #Create a contingency table
    contingency_table = pd.crosstab(df[testVar1], df[testVar2])

    #Run chi-square test
    chi2, p_val, dof, expected = chi2_contingency(contingency_table)

    print(f"Chi-square statistic: {chi2:.3f}, p-value: {p_val:.3f}")

    return chi2, p_val

def hypothTest(df, categoryCol, value1, value2, targetCol):
    """
    Perform a hypothesis test (independent t-test) to determine if the mean target value
    (e.g., price) differs significantly between two groups of a categorical variable.

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset containing the data.
    category_col : str
        Categorical column to split groups.
    value1 : str
        First category value to compare.
    value2 : str
        Second category value to compare.
    target_col : str, optional
        Target numeric column to compare.

    Returns:
    --------
    t_stat : float
        T-test statistic.
    p_val : float
        p-value from the test.
    """

    # Extract non-missing target values for each group based on category values
    group1 = df[df[categoryCol] == value1][targetCol].dropna()
    group2 = df[df[categoryCol] == value2][targetCol].dropna()

    #Performs the hypothesis test and displays the results
    t_stat, p_val = ttest_ind(group1, group2, equal_var=False)  
    print(f"Hypothesis Test between '{value1}' and '{value2}' for '{targetCol}':")
    print(f"T-statistic: {t_stat:.3f}, p-value: {p_val:.3f}")

    return t_stat, p_val

def tukeyHSD(df, tarVar, groupVar, alpha=0.05):
    """
    Perform Tukey’s Honest Significant Difference (HSD) post-hoc test after ANOVA.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    tarVar : str
        Numerical target variable (e.g., 'price').
    groupVar : str
        Categorical variable used to define the groups (e.g., 'room_type').
    alpha : float, optional
        Significance level. Default is 0.05.

    Returns:
    --------
    result : TukeyHSDResults
        Results of the Tukey HSD test.
    """
    
    # Filter out rows with missing values and ensure group labels are strings
    subset = df[[tarVar, groupVar]].dropna()  
    subset = subset[subset[groupVar].apply(lambda x: isinstance(x, str))]  

    # Run Tukey's HSD test and print the summary table
    tukey = pairwise_tukeyhsd(endog=subset[tarVar], groups=subset[groupVar], alpha=alpha)
    print(tukey.summary())
    return tukey

def cohens_d(df, tarVar, testVarCol, testVar1, testVar2):
    """
    Calculate Cohen's d effect size between two groups.

    Cohen’s d is a measure of effect size that quantifies the difference between two group means in terms of standard deviation units.
    It complements hypothesis tests by indicating the practical significance or magnitude of the difference, regardless of p-value.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    tarVar : str
        The name of the numerical target variable.
    testVarCol : str
        The name of the categorical column used to define groups.
    testVar1 : str
        The first group/category to compare.
    testVar2 : str
        The second group/category to compare.

    Returns:
    --------
    d : float
        Cohen's d value. Interpret as:
        - 0.2 = small effect
        - 0.5 = medium effect
        - 0.8+ = large effect
        Values close to 0 imply a negligible difference.
    """

    # Extract non-missing target values for each group
    group1 = df[df[testVarCol] == testVar1][tarVar].dropna()
    group2 = df[df[testVarCol] == testVar2][tarVar].dropna()

    # Calculate pooled standard deviation
    nx, ny = len(group1), len(group2)
    pooled_std = np.sqrt(((nx - 1)*np.var(group1, ddof=1) + (ny - 1)*np.var(group2, ddof=1)) / (nx + ny - 2))

    # Compute Cohen's d
    d = (np.mean(group1) - np.mean(group2)) / pooled_std

    # Print and return the effect size
    print(f"Cohen's d: {d:.3f}")
    return d