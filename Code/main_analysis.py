import pandas as pd #Library used for data manipulation and analysis 

import matplotlib.pyplot as plt #Library used to create plots and visualisations

from dataStructures import searchDictValue

from visualisations import pltBar
from visualisations import pltPie
from visualisations import pltGroupedBar

from correlationAnalysis import corrMatrix
from correlationAnalysis import pairplots
from correlationAnalysis import detMulticollinearity

from advStatTest import anova
from advStatTest import tTest
from advStatTest import chiSqrTest
from advStatTest import hypothTest
from advStatTest import tukeyHSD
from advStatTest import cohens_d

from predictiveModelling import predModSetup
from predictiveModelling import linearReg
from predictiveModelling import ridgeReg
from predictiveModelling import lassoReg
from predictiveModelling import decTree
from predictiveModelling import randForest
from predictiveModelling import gradBoost
from predictiveModelling import xgBoost
from predictiveModelling import plotFeatureImportance

from clusterAlg import scaleDf
from clusterAlg import dbscan

def categorise_sentiment(score):
    if score > 0.05:  # tweak thresholds as needed
        return 'Positive'
    elif score < -0.05:
        return 'Negative'
    else:
        return 'Neutral'
    
#Loads the main data files needed
listings = pd.read_csv('/Users/hariramgnanachandran/Documents/Data Science : Machine Learning Work/Projects/London AirBnB Analysis/Outputs/Datasets/listings_cleaned.csv')
mergedListRev = pd.read_csv('/Users/hariramgnanachandran/Documents/Data Science : Machine Learning Work/Projects/London AirBnB Analysis/Outputs/Datasets/cleaned_analysis.csv')

#=====================================
#Textual Analysis - Insight Generation
#=====================================

#------------------------------
#Identify Key Factors
#------------------------------

#Dictionary mapping topic numbers to their value names assigned
topicHeadersDict = {
    0: "House Cleanliness and General Impressions",
    1: "Apartment Quality and Host Experience",
    2: "Proximity to Transport and Cleanliness",
    3: "Comfort and Overall Stay Satisfaction",
    4: "Excellent Hosts and Prime Location"
}

mergedListRev['top_topic_header'] = mergedListRev['top_topic'].apply(lambda x: searchDictValue(x, topicHeadersDict))

#Determine the mean sentiment score for each of the topics
averageTopic = mergedListRev.groupby('top_topic')['review_sentiment'].mean().to_dict()
averageTopicList = list(averageTopic.values())

# Convert the dictionary values of the topic names to a list
topicNames = list(topicHeadersDict.values())

#Plots bar chart of the average sentiment scores for each topic identified
pltBar(topicNames, averageTopicList, 'Average Sentiment Score for each Topic', 'Topic', 'Average Sentiment Score')

#------------------------------
#Summarise Overall Sentiment
#------------------------------

#Determines whether each review is positive, neutral or negative
mergedListRev['sentiment_category'] = mergedListRev['review_sentiment'].apply(categorise_sentiment)

#Determines the proportion of each category and stores in an array as a percentage value
sentimentPercent = mergedListRev['sentiment_category'].value_counts(normalize=True) * 100
valuesArray = sentimentPercent.values
labelsList = sentimentPercent.index.tolist()

#Displays a pie chart of the proportion of positive, neutral and negative comments
pltPie(labelsList, valuesArray, 'Overall Sentiment Distribution', 'Sentiment')

#------------------------------
#Common Topics - Pos vs Neg
#------------------------------

# Define thresholds for sentiment
positiveReviews = mergedListRev[mergedListRev['review_sentiment'] > 0.1]
negativeReviews = mergedListRev[mergedListRev['review_sentiment'] < -0.1]

# Count topics in positive reviews
positiveTopicCounts = positiveReviews['top_topic_header'].value_counts()

# Count topics in negative reviews
negativeTopicCounts = negativeReviews['top_topic_header'].value_counts()

#DataFrame storing the number of positive and negative reviews of each opic
topic_comparison_df = pd.DataFrame({
    'Positive Reviews': positiveTopicCounts,
    'Negative Reviews': negativeTopicCounts
}).fillna(0)

#Transforms dataset from wide format to long format
melted_df = topic_comparison_df.reset_index().melt(
    id_vars='top_topic_header',
    var_name='Sentiment',
    value_name='Count'
)

#Plots a grouped bar chart of the number of positive and negative reviews
pltGroupedBar(melted_df, 'top_topic_header', 'Count', 'Sentiment', 'Topic Mentions in Positive vs. Negative Reviews', 'Topic', 'Number of Mentions')

#------------------------------
#Compare Sentiment and Topic by Borough
#------------------------------

#Groups the listing by their relative boroughs and finds the average sentiment score for each borough
avgSentimentByLocation = mergedListRev.groupby('neighbourhood_cleansed')['review_sentiment'].mean().sort_values()

#Plots a bar chart of the average sentiment scores for each borough
pltBar(avgSentimentByLocation.index, avgSentimentByLocation.values, 'Average Sentiment by Borough', 'Borough', 'Average Sentiment')

#Groups the listing by their relative boroughs and finds the top topic for each borough
avgTopicByLocation = mergedListRev.groupby('neighbourhood_cleansed')['top_topic'].agg(lambda x: x.mode()[0])

#Plots a bar chart of the top topic for each borough
pltBar(avgTopicByLocation.index, avgTopicByLocation.values, 'Top Topic by Borough', 'Borough', 'Top Topic')

#------------------------------
#Compare Sentiment and Topic by Room Type
#------------------------------

#Groups the listing by their relative boroughs and finds the average sentiment score for each borough
avgSentimentByLocation = mergedListRev.groupby('room_type')['review_sentiment'].mean().sort_values()

#Plots a bar chart of the average sentiment scores for each borough
pltBar(avgSentimentByLocation.index, avgSentimentByLocation.values, 'Average Sentiment by Room Type', 'Room Type', 'Average Sentiment')

#Groups the listing by their relative boroughs and finds the top topic for each borough
avgTopicByLocation = mergedListRev.groupby('room_type')['top_topic'].agg(lambda x: x.mode()[0])

#Plots a bar chart of the top topic for each borough
pltBar(avgTopicByLocation.index, avgTopicByLocation.values, 'Top Topic by Room Type', 'Room Type', 'Top Topic')

#------------------------------
#Find Consistently High & Low Listings
#------------------------------

listingSentimentStats = mergedListRev.groupby('id_x')['review_sentiment'].agg(
    mean_sentiment='mean',
    review_count='count',
    std_sentiment='std'
).reset_index()

#Set thresholds for high and low listings
highThreshold = 0.7
lowThreshold = -0.5
minReviews = 1  

# Consistently high sentiment
consistentlyHigh = listingSentimentStats[
    (listingSentimentStats['mean_sentiment'] > highThreshold) &
    (listingSentimentStats['review_count'] >= minReviews) &
    (listingSentimentStats['std_sentiment'] < 0.2)  
]

#Saves results
listingSentimentStats.to_csv('/Users/hariramgnanachandran/Documents/Data Science : Machine Learning Work/Projects/London AirBnB Analysis/Outputs/Datasets/1.csv', index=False)
consistentlyHigh.to_csv('/Users/hariramgnanachandran/Documents/Data Science : Machine Learning Work/Projects/London AirBnB Analysis/Outputs/Datasets/2.csv', index=False)


#Plots a bar chart of the average sentiment score for highly performing listings
pltBar(consistentlyHigh['id_x'], consistentlyHigh['mean_sentiment'], 'Mean Sentiments of Highly Performing Listings', 'Listing ID', 'Average Sentiment')


# Consistently low sentiment
consistentlyLow = listingSentimentStats[
    (listingSentimentStats['mean_sentiment'] < lowThreshold) &
    (listingSentimentStats['review_count'] >= minReviews) &
    (listingSentimentStats['std_sentiment'] < 0.2)
]

#Plots a bar chart of the average sentiment score for highly performing listings
pltBar(consistentlyLow['id_x'], consistentlyLow['mean_sentiment'], 'Mean Sentiments of Lowly Performing Listings', 'Listing ID', 'Average Sentiment')


#=====================================
#CORRELATION ANALYSIS
#=====================================

# List of numerical columns selected for detailed correlation analysis
numericalCols = ['price', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'host_response_rate' ,'host_acceptance_rate', 
                 'number_of_reviews', 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 
                 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value', 
                 'availability_365', 'minimum_nights', 'maximum_nights', 'distance_to_charing_cross', 'distance_to_canary_wharf', 'distance_to_soho_&_chinatown', 'distance_to_wembley_stadium',
                 'distance_to_hyde_park', 'distance_to_greenwich_park', 'distance_to_heathrow_airport', 
                 'min_night_to_price_ratio', 'review_sentiment']

# Generate and display a correlation matrix heatmap for the selected numerical features
corrMatrix(mergedListRev, numericalCols)

# A simplified subset of numerical columns to visualise pairwise relationships and check multicollinearity
numericalColsSimp = ['price', 'accommodates', 'host_response_rate', 'review_scores_rating', 'minimum_nights', 
                     'distance_to_charing_cross', 'review_sentiment']

# Create pairplots for the simplified list of numerical variables to visualise distributions and bivariate relationships
pairplots(mergedListRev, numericalColsSimp)

# Calculate and print Variance Inflation Factor (VIF) for the simplified numerical columns to detect multicollinearity issues
detMulticollinearity(mergedListRev, numericalColsSimp)

#=====================================
#ADVANCED STATISTICAL ANALYSIS
#=====================================

#------------------------------
#One-Way Anova Test
#------------------------------

#Performs one way anova to assess overall differences in mean prices across groups
anova(mergedListRev, 'price', 'neighbourhood_cleansed')
anova(mergedListRev, 'price', 'property_type')
anova(mergedListRev, 'price', 'room_type')

#------------------------------
#Tukey's HSD
#------------------------------

# Post-hoc analysis (Tukey's HSD) conducted following significant ANOVA results to identify specific group differences.
tukeyHSD(mergedListRev, 'price', 'neighbourhood_cleansed')
tukeyHSD(mergedListRev, 'price', 'property_type')
tukeyHSD(mergedListRev, 'price', 'room_type')

#------------------------------
#T-Test
#------------------------------

#T-tests for testing mean differences between two groups for categorical variables
filtered_df = mergedListRev[mergedListRev['host_is_superhost'].isin(['t', 'f'])]
tTest(filtered_df, 'price', 'host_is_superhost', 't', 'f')
tTest(mergedListRev, 'price', 'instant_bookable', 't', 'f')

#------------------------------
#Cohen's d 
#------------------------------

#Calculation of effect sizes to quantify the magnitude of significant differences identified in t-tests 
cohens_d(filtered_df, 'price', 'host_is_superhost', 't', 'f')
cohens_d(mergedListRev, 'price', 'instant_bookable', 't', 'f')

#------------------------------
#Chi-Square Test
#------------------------------

#Performs chi-square test to examine associations between categorical variables
chiSqrTest(mergedListRev, 'room_type', 'neighbourhood_cleansed')
chiSqrTest(mergedListRev, 'room_type', 'property_type')
chiSqrTest(mergedListRev, 'room_type', 'host_is_superhost')
chiSqrTest(mergedListRev, 'neighbourhood_cleansed', 'property_type')

#=====================================
#PREDICTIVE MODELLING
#=====================================

# Define the list of feature columns to use for prediction, mostly numerical and encoded categorical variables
features = ['accommodates', 'bedrooms', 'review_scores_rating', 'availability_365', 'distance_to_charing_cross', 
            'min_night_to_price_ratio', 'host_is_superhost_f', 'host_is_superhost_t', 'neighbourhood_cleansed_Barnet',
            'neighbourhood_cleansed_Brent', 'neighbourhood_cleansed_Camden', 'neighbourhood_cleansed_Greenwich',
            'neighbourhood_cleansed_Hackney', 'neighbourhood_cleansed_Hammersmith and Fulham', 'neighbourhood_cleansed_Haringey',
            'neighbourhood_cleansed_Havering', 'neighbourhood_cleansed_Islington', 'neighbourhood_cleansed_Kensington and Chelsea',
            'neighbourhood_cleansed_Lambeth', 'neighbourhood_cleansed_Merton', 'neighbourhood_cleansed_Newham',
            'neighbourhood_cleansed_Richmond upon Thames', 'neighbourhood_cleansed_Southwark', 'neighbourhood_cleansed_Tower Hamlets',
            'neighbourhood_cleansed_Waltham Forest', 'neighbourhood_cleansed_Wandsworth', 'neighbourhood_cleansed_Westminster',
            'latitude', 'longitude', 'property_type_Entire condo', 'property_type_Entire home', 'property_type_Entire rental unit',
            'property_type_Entire serviced apartment', 'property_type_Entire townhouse', 'property_type_Private room in bed and breakfast',
            'property_type_Private room in condo', 'property_type_Private room in home', 'property_type_Private room in loft', 
            'property_type_Private room in rental unit', 'property_type_Private room in townhouse', 'property_type_Room in aparthotel',
            'property_type_Room in serviced apartment', 'property_type_Shared room in home', 'room_type_Entire home/apt',
            'room_type_Hotel room', 'room_type_Private room', 'room_type_Shared room', 'bathrooms', 'minimum_nights', 
            'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication',
            'review_scores_location', 'review_scores_value']

# Define the target variable to predict
target = 'price'

# Split the data into training and testing sets using a custom function predModSetup
xTrain, xTest, yTrain, yTest = predModSetup(listings, features, target)

# Train and evaluate different regression models on the data
linModel, linRMSE, linMAE, linR2 = linearReg(xTrain, yTrain, xTest, yTest)
ridgeModel, ridgeRMSE, ridgeMAE, ridgeR2 = ridgeReg(xTrain, yTrain, xTest, yTest)
lassoModel, lassoRMSE, lassoMAE, lassoR2 = lassoReg(xTrain, yTrain, xTest, yTest)
decTreeModel, decTreeRMSE, decTreeMAE, decTreeR2 = decTree(xTrain, yTrain, xTest, yTest)
randForModel, randForRMSE, randForMAE, randForR2 = randForest(xTrain, yTrain, xTest, yTest)
gradBoostMod, gradBoostRMSE, gradBoostMAE, gradBoostR2 = gradBoost(xTrain, yTrain, xTest, yTest)
xgBoostModel, xgBoostRMSE, xgBoostMAE, xgBoostR2 = xgBoost(xTrain, yTrain, xTest, yTest)

# Aggregate the evaluation results into a DataFrame for easy comparison
results = pd.DataFrame({
    'Model_Name': ['Linear Regression', 'Ridge', 'Lasso', 'Decision Tree', 'Random Forest', 'Gradient Boost', 'XGBoost'],
    'RMSE': [linRMSE, ridgeRMSE, lassoRMSE, decTreeRMSE, randForRMSE, gradBoostRMSE, xgBoostRMSE],
    'MAE': [linMAE, ridgeMAE, lassoMAE, decTreeMAE, randForMAE, gradBoostMAE, xgBoostMAE],
    'R2': [linR2, ridgeR2, lassoR2, decTreeR2, randForR2, gradBoostR2, xgBoostR2],
    'Model': [linModel, ridgeModel, lassoModel, decTreeModel, randForModel, gradBoostMod, xgBoostModel]
})

# Aggregate the evaluation results into a DataFrame for easy comparison
pltBar(results['Model_Name'], results['R2'], 'Model R² Scores Comparison', 'Model', 'R² Score')

# Identify the model with the highest R² score
bestRow = results.loc[results['R2'].idxmax()]

# Extract the best model's name, R2, and model object
bestModelName = bestRow['Model_Name']
bestR2Score = bestRow['R2']
bestModel = bestRow['Model']

# Visualize the feature importance for the best performing model
plotFeatureImportance(bestModel, features, bestModelName)

#=====================================
#CLUSTERING 
#=====================================

# For each pair of variables (mostly price vs another feature), scale the data and apply DBSCAN clustering.

scaledFeatures, cleanDf = scaleDf(mergedListRev, 'price', 'number_of_reviews')
dbscan(cleanDf, scaledFeatures, 'price', 'number_of_reviews')

scaledFeatures, cleanDf = scaleDf(mergedListRev, 'price', 'review_scores_rating')
dbscan(cleanDf, scaledFeatures, 'price', 'review_scores_rating')

scaledFeatures, cleanDf = scaleDf(mergedListRev, 'price', 'accommodates')
dbscan(cleanDf, scaledFeatures, 'price', 'accommodates')

scaledFeatures, cleanDf = scaleDf(mergedListRev, 'price', 'bedrooms')
dbscan(cleanDf, scaledFeatures, 'price', 'bedrooms')

scaledFeatures, cleanDf = scaleDf(mergedListRev, 'price', 'beds')
dbscan(cleanDf, scaledFeatures, 'price', 'beds')

scaledFeatures, cleanDf = scaleDf(mergedListRev, 'price', 'minimum_nights')
dbscan(cleanDf, scaledFeatures, 'price', 'minimum_nights')

scaledFeatures, cleanDf = scaleDf(mergedListRev, 'price', 'availability_365')
dbscan(cleanDf, scaledFeatures, 'price', 'availability_365')

scaledFeatures, cleanDf = scaleDf(mergedListRev, 'latitude', 'longitude')
dbscan(cleanDf, scaledFeatures, 'latitude', 'longitude')
