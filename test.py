import pandas as pd #Library used for data manipulation and analysis 

#Loads the necessary functions required 
from dataInspection import dataInspection

from dataPreprocessing import handleEmptyCol
from dataPreprocessing import handleMissing
from dataPreprocessing import convertMoney
from dataPreprocessing import convertPercent
from dataPreprocessing import plotBox
from dataPreprocessing import removeOutliers
from dataPreprocessing import dateTimeFeatEng
from dataPreprocessing import onehotEncode
from dataPreprocessing import labelEncode
from dataPreprocessing import mergeDataset
from dataPreprocessing import mergeConsistency

from textProcessing import cleanText
from textProcessing import tokenise
from textProcessing import get_similarityMatrix
from textProcessing import translateToEnglish
from textProcessing import wordCloud
from textProcessing import wordFreq
from textProcessing import vaderSentiment
from textProcessing import lda

from visualisations import pltHistogram
from visualisations import pltLogHistogram
from visualisations import pltViolin
from visualisations import pltBar
from visualisations import pltBox
from visualisations import pltHeatMap
from visualisations import pltScatter

from geospatAnal import plotListings
from geospatAnal import groupListings
from geospatAnal import calculateDistance
from geospatAnal import getDistColours
from geospatAnal import plotByDist

from dataStructures import createDictionary
from dataStructures import searchDict
from dataStructures import dropColumn

#==============================
# DATA COLLECTION
#==============================

#Loads the main data files needed
calender = pd.read_csv('/Users/hariramgnanachandran/Documents/Data Science : Machine Learning Work/Projects/London AirBnB Analysis/Dataset/calendar.csv')
listings = pd.read_csv('/Users/hariramgnanachandran/Documents/Data Science : Machine Learning Work/Projects/London AirBnB Analysis/Dataset/listings.csv')
neighbourhoods = pd.read_csv('/Users/hariramgnanachandran/Documents/Data Science : Machine Learning Work/Projects/London AirBnB Analysis/Dataset/neighbourhoods.csv')
reviews = pd.read_csv('/Users/hariramgnanachandran/Documents/Data Science : Machine Learning Work/Projects/London AirBnB Analysis/Dataset/reviews.csv')

# Extract the first 1000 rows of reviews
reviews = reviews.sample(n=1000, random_state=42)

#==============================
#DATA INSPECTION
#==============================

#CProvides an initial inspection on the datasets
dataInspection(listings)

#==============================
# DATA PREPROCESSING
#==============================

#------------------------------
#HANDLING MISSING VALUES
#------------------------------

#Calls a function to handle the missing values in the dataset
pltHeatMap(listings,'Missing Data Heatmap', 'Columns', 'Rows')
listings = handleEmptyCol(listings)
listings = handleMissing(listings)
pltHeatMap(listings,'Missing Data Heatmap', 'Columns', 'Rows')

#Calls a function to convert 'price' to numerical format to be processed appropriately
listings = convertMoney(listings, 'price')

#Calls a function to convert 'host_acceptance_rate' to numerical format to be processed appropriately
listings = convertPercent(listings, 'host_acceptance_rate')

#Calls a function to convert 'host_response_rate' to numerical format to be processed appropriately
listings = convertPercent(listings, 'host_response_rate')

#Calls a function to display a boxplot to help visualise the spread of the data
plotBox(listings, 'price')

#Calls a function to determine the boundaries and removes the outliers
listings = removeOutliers(listings, 'price')

#Calls a function to extract date features (year, month and day) and time features (hour, minute and second) from date/time 
#data columns
listings = dateTimeFeatEng(listings)
reviews = dateTimeFeatEng(reviews)

#Calls a function to one hot encode the categorical variables
ignoreCols = ['listing_url', 'name', 'description', 'neighborhood_overview', 'picture_url', 'host_url', 'host_about', 
              'host_name', 'host_thumbnail_url', 'host_picture_url', 'amenities', 'price', 'calender_last_scraped',
              'host_since', 'last_scraped', 'calender_last_scraped', 'first_review', 'last_review']
listings = onehotEncode(listings, ignoreCols)

ordinalCols = ['host_response_time']
customMappings = {
    'host_response_time': {
        'within an hour': 0,
        'within a few hours': 1,
        'within a day': 2,
        'a few days or more': 3
    }
}

listings = labelEncode(listings, ordinalCols, customMappings)

print("1")

#Calls a function to tokenise and lemmatise the columns from the listings database 
customStopwords = {}
listings = cleanText(listings, 'name', 'name_tokenised')
listings = tokenise(listings, 'name_tokenised', customStopwords)

print("2")

listings = cleanText(listings, 'description', 'description_tokenised')
listings = tokenise(listings, 'description_tokenised', customStopwords)

print("3")

listings = cleanText(listings, 'neighborhood_overview', 'neighborhood_overview_tokenised')
listings = tokenise(listings, 'neighborhood_overview_tokenised', customStopwords)

print("4")

listings = cleanText(listings, 'host_about', 'name_host_abouttokenised')
listings = tokenise(listings, 'name_host_abouttokenised', customStopwords)

print("5")

#Calls a function to tokenise and lemmatise the reviews
reviews['comments_translated'] = reviews['comments'].apply(translateToEnglish)
print("6")

reviews = cleanText(reviews, 'comments_translated', 'comments_tokenised') 
print("7")


customStopwords = {'stay', 'would', 'room', 'br', 'us', 'get', 'like', 'also', 'gideon', 'one' 'etc', 'etc.', 'really'}
reviews = tokenise(reviews, 'comments_tokenised', customStopwords)
print("8")


#==============================
#MERGE DATASETS
#==============================

#Calls a function to merge the 'listings' and 'reviews' dataset 
mergedListRev = mergeDataset(listings, reviews, 'id', 'listing_id')

#Calls a function to verify data consistency across the merged datasets
mergeConsistency(mergedListRev, 'id_y')

#==============================
#Advanced Feature Engineering
#==============================

#------------------------------
#SEASONAL TRENDS
#------------------------------

#Dictionary with the seasons as key and the months as values
seasonWithMonth = {
    'Spring': [3, 4, 5],
    'Summer': [6, 7, 8],
    'Autumn': [9, 10, 11],
    'Winter': [12, 1, 2]
}

# Function to map each month to its season based on the dictionary
mergedListRev['review_season'] = mergedListRev['date_month'].apply(lambda x: searchDict(x, seasonWithMonth))

#Plots a violin plot of the different room types against price 
pltViolin(mergedListRev, 'review_season', 'price', 'Distribution of Prices by Review Season', 'Review Season', 'Price')

#------------------------------
#MINIMUM NIGHTS / PRICE RATIO
#------------------------------

#Helps detect high-price, short-term listings (possibly tourist traps).
listings['min_night_to_price_ratio'] = mergedListRev['minimum_nights'] / mergedListRev['price']

#Helps detect high-price, short-term listings (possibly tourist traps).
mergedListRev['min_night_to_price_ratio'] = mergedListRev['minimum_nights'] / mergedListRev['price']

#------------------------------
#REVIEW RATE
#------------------------------

#Provides proxy for demand and popularity
listings['review_rate'] = mergedListRev['number_of_reviews'] / mergedListRev['availability_365']

#Provides proxy for demand and popularity
mergedListRev['review_rate'] = mergedListRev['number_of_reviews'] / mergedListRev['availability_365']

#------------------------------
#PRICE PER BEDROOM
#------------------------------

#Normalises price by capacity
listings['price_per_bedroom'] = mergedListRev['price'] / mergedListRev['bedrooms']

#Normalises price by capacity
mergedListRev['price_per_bedroom'] = mergedListRev['price'] / mergedListRev['bedrooms']

#------------------------------
#PRICE PER PERSON
#------------------------------

#Indicates cost per guest, useful for assessing affordability
listings['price_per_person'] = mergedListRev['price'] / mergedListRev['accommodates']

#Indicates cost per guest, useful for assessing affordability
mergedListRev['price_per_person'] = mergedListRev['price'] / mergedListRev['accommodates']

#------------------------------
#REVENUE POTENTIAL
#------------------------------

#Estimates potential annual revenue if booked every available day
listings['revenue_potential'] = mergedListRev['price'] * mergedListRev['availability_365']

#Estimates potential annual revenue if booked every available day.
mergedListRev['price_per_person'] = mergedListRev['price'] * mergedListRev['availability_365']

#==============================
#EXPLORATORY DATA ANALYSIS
#==============================

#Plots a histogram of sale price against frequency
pltHistogram(listings, 'price', 'Price Distribution', 'Price')

#Plots a histogram of the logarithm of sale price against frequency
pltLogHistogram(listings, 'price', 'Log-Transformed Price Distribution', 'Log(1 + Price)')

#Plots a violin plot of the different room types against price 
pltViolin(listings, 'room_type', 'price', 'Distribution of Prices by Room Type', 'Room Type', 'Price')

#Plots a scatter plot of price against minimum nights
pltScatter(listings['minimum_nights'], listings['price'], 'Distribution of Price by Minimum Nights', 'Minimum Nights', 'Price')

#Plots a box plot of host response time against price
pltBox(listings, 'host_response_time', 'price', 'Distribution of Prices by Host Response Time', 'Host Response Time', 'Price')

#==============================
#GEOSPATIAL ANALYSIS
#==============================

#------------------------------
#PLOT LISTINGS
#------------------------------

londonCoords = [51.5074, -0.1278]  # Latitude & Longitude of Central London

#Plots a map of Central London with the AirBnb's plotted as markers with the price, property type and number of reviews shown
#as popups
columnsToShow = ['price', 'property_type', 'number_of_reviews']
londonMap = plotListings(listings, londonCoords, columnsToShow)

# Save or display the map of the AirBnbs in Central London
londonMap.save("/Users/hariramgnanachandran/Documents/Data Science : Machine Learning Work/Projects/London AirBnB Analysis/Outputs/Plots & Images/Geospatial Results/airbnb_map.html")

#------------------------------
#PLOT LISTINGS BY ROOM TYPE
#------------------------------

#Dictionary to determine the colour of the marker based on the property type
roomColors = {
    'Entire home/apt': 'blue',
    'Private room': 'green',
    'Shared room': 'red',
    'Hotel room': 'purple'
}

#Create and save a map of the AirBnb's in Central London with the markers colour coded based on property type
propTypeMap = groupListings(listings, londonCoords, columnsToShow, roomColors, 'room_type')
propTypeMap.save("/Users/hariramgnanachandran/Documents/Data Science : Machine Learning Work/Projects/London AirBnB Analysis/Outputs/Plots & Images/Geospatial Results/airbnb_map_roomtype.html")

#------------------------------
#PLOT BY DISTANCE TO KEY LOCATIONS
#------------------------------

#Dictionary to store the key locations in Central London and their latitudes and longitudes
keyLocations = {
    #Central and Business Locations
    'Charing Cross': (51.5074, -0.1278),
    'Trafalgar Square': (51.5080, -0.1281),
    'London Bridge': (51.5079, -0.0877),
    'Canary Wharf': (51.5055, -0.0235),
    'Westminster Abbey': (51.4993, -0.1273),

    #Famous Historical and Cultural Landmarks
    'London Eye': (51.5033, -0.1196),
    'Tower of London': (51.5081, -0.0759),
    'Big Ben': (51.5007, -0.1246),
    'Tower Bridge': (51.5056, -0.0753),

    #Shopping, Dining & Entertainment
    'Oxford Circus': (51.5154, -0.1410),
    'Covent Garden': (51.5115, -0.1231),
    'Camden Market': (51.5416, -0.1456),
    'Soho & Chinatown': (51.5135, -0.1312),

    #Parks and Open Spaces
    'Hyde Park': (51.5073, -0.1657),
    'Greenwich Park': (51.4769, 0.0005),
    'Kew Gardens': (51.4780, -0.2956),
    'Richmond Park': (51.4426, -0.2733),

    #Educational and Cultural Institutions
    'British Museum': (51.5188, -0.1262),
    'Natural History Museum': (51.4967, -0.1765),
    'Victoria & Albert Museum': (51.4966, -0.1722),
    'Royal Observatory & Prime Meridian': (51.4769, -0.0005),

    #Sports and Events 
    'Wembley Stadium': (51.5560, -0.2795),
    'Wimbledon Centre': (51.4344, -0.2140),

    #Transportation Hubs
    "King's Cross Station": (51.5308, -0.1238),
    'Heathrow Airport': (51.4700, -0.4543),
    'London City Airport': (51.5053, 0.0553)
}

#List to store the boundary values to determine how far the AirBnb property is from the key locations
distanceClassify = [1,5]

#Creates new columns in the DataFrame to store the distances between each AirBnb and the key locations and colour codes 
#them based on their distance
for place, coords in keyLocations.items():
    cleanDist = place.replace(" ", "_").lower()
    mergedListRev[f'distance_to_{cleanDist}'] = mergedListRev.apply(
        lambda row: calculateDistance(row, locationCoords=coords),
        axis=1
    )
    
    mergedListRev[f'distance_to_{cleanDist}_colour'] = mergedListRev.apply(
        lambda row: getDistColours(row, distVar=f'distance_to_{cleanDist}', distanceColors=distanceClassify),
        axis=1
    )

#Plots a map of Central London with the AirBnb's and colour codes them based on their distance to the first item in the list
place1 = list(keyLocations.keys())[0]
cleanPlace1 = place1.replace(" ", "_").lower()
distMap1 = plotByDist(mergedListRev, londonCoords, f'distance_to_{cleanPlace1}_colour', f'distance_to_{cleanPlace1}')
distMap1.save("/Users/hariramgnanachandran/Documents/Data Science : Machine Learning Work/Projects/London AirBnB Analysis/Outputs/Plots & Images/Geospatial Results/dist_map1.html")

#==============================
#Textual Analysis
#==============================

#Creates a word cloud of the most common words in the reviews 
reviewData = mergedListRev['comments_tokenised']
wordCloud(reviewData, 'Word Cloud of Reviews')
wordFreq(reviewData, 'Top 20 most common words in reviews')

keys, groupedListing = createDictionary(mergedListRev, 'id_x', 'comments', '/Users/hariramgnanachandran/Documents/Data Science : Machine Learning Work/Projects/London AirBnB Analysis/Outputs/Datasets/Reviews Listing/listing_')

# Prepare a dictionary to store embeddings
simMatDict = {}

#Create a similarity matrix for the comments of each listing
for id_x in keys:

    # Get the reviews for this listing as a list of strings
    reviews = groupedListing[id_x].tolist()

    # Pass the list of reviews to the embedding function
    simMatDict[id_x] = get_similarityMatrix(reviews)

#Performs a sentiment analysis on the guests' reviews
mergedListRev['review_sentiment'] = mergedListRev['comments_translated'].apply(vaderSentiment)

#Uses Latent Dirichlet Allocation (LDA) to extract topics
mergedListRev, topics, topTopics = lda(mergedListRev, 'comments_tokenised')

# Convert list to DataFrame
reviews = pd.DataFrame(reviews)

#Saves the preprocessed DataFrame 
listings.to_csv('/Users/hariramgnanachandran/Documents/Data Science : Machine Learning Work/Projects/London AirBnB Analysis/Outputs/Datasets/listings_cleaned.csv', index=False)
reviews.to_csv('/Users/hariramgnanachandran/Documents/Data Science : Machine Learning Work/Projects/London AirBnB Analysis/Outputs/Datasets/reviews_cleaned.csv', index=False)
mergedListRev.to_csv('/Users/hariramgnanachandran/Documents/Data Science : Machine Learning Work/Projects/London AirBnB Analysis/Outputs/Datasets/cleaned_analysis.csv', index=False)