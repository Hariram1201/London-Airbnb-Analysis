import pandas as pd #Library used for data manipulation and analysis 
import sqlite3 #Library to connect to SQLite databases and run SQL queries
import os #Library to import the operating system interface module
import sqlalchemy #Library that provides a high-level, flexible way to interact with databases

from visualisations import pltBar

# Load the dataset
df = pd.read_csv('/Users/hariramgnanachandran/Documents/Data Science : Machine Learning Work/Projects/London AirBnB Analysis/Dataset/listings.csv')

# Clean the 'price' column: remove $ sign and convert to numeric
df['price'] = pd.to_numeric(df['price'].replace('[\$,]', '', regex=True), errors='coerce')

# Fill missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna('Unknown')
    else:
        df[col] = df[col].fillna(df[col].median())

# Connect to SQLite DB and upload the DataFrame
db_path = '/Users/hariramgnanachandran/Documents/Data Science : Machine Learning Work/Projects/London AirBnB Analysis/Dataset/airbnb_london.db'
conn = sqlite3.connect(db_path)
df.to_sql('listingsSQL', conn, if_exists='replace', index=False)

#------------------------------
#AVERAGE PRICE BY NEIGHBOURHOOD
#------------------------------

# Run SQL query: Average price by neighbourhood
query_neighbourhood_price = """
SELECT 
    neighbourhood_cleansed, 
    ROUND(AVG(price), 2) AS avg_price
FROM 
    listingsSQL
GROUP BY 
    neighbourhood_cleansed
"""

avg_neighbourhood_df = pd.read_sql(query_neighbourhood_price, conn)

#------------------------------
#AVERAGE PRICE BY PROPERTY TYPE
#------------------------------

# Run SQL query: Average price by property type
query_property_price = """
SELECT 
    property_type, 
    ROUND(AVG(price), 2) AS avg_price
FROM 
    listingsSQL
GROUP BY 
    property_type
"""

avg_property_df = pd.read_sql(query_property_price, conn)

#------------------------------
#MISSING REVIEWS
#------------------------------

# Run SQL query: Missing Reviews
query_missing_reviews = """
SELECT 
    id,
    name,
    neighbourhood_cleansed,
    number_of_reviews,
    last_review
FROM 
    listingsSQL
WHERE 
    number_of_reviews = 0 OR last_review IS NULL
ORDER BY
    neighbourhood_cleansed, id
"""

missing_reviews_df = pd.read_sql(query_missing_reviews, conn)

#------------------------------
#SAVE TO CSV FILE
#------------------------------

# Export the result to CSV
os.makedirs('outputs', exist_ok=True)
avg_neighbourhood_df.to_csv('/Users/hariramgnanachandran/Documents/Data Science : Machine Learning Work/Projects/London AirBnB Analysis/Outputs/SQL Results/SQL - Practice Tasks/avg_price_by_neighbourhood.csv', index=False)
print("✅ Exported filtered results to Outputs/SQL Results/avg_price_by_neighbourhood.csv")
avg_property_df.to_csv('/Users/hariramgnanachandran/Documents/Data Science : Machine Learning Work/Projects/London AirBnB Analysis/Outputs/SQL Results/SQL - Practice Tasks/avg_price_by_property_type.csv', index=False)
print("✅ Exported filtered results to Outputs/SQL Results/avg_price_by_property_type.csv")
missing_reviews_df.to_csv('/Users/hariramgnanachandran/Documents/Data Science : Machine Learning Work/Projects/London AirBnB Analysis/Outputs/SQL Results/SQL - Practice Tasks/missing_reviews.csv', index=False)
print("✅ Exported filtered results to Outputs/SQL Results/missing_reviews.csv")

# Close the connection
conn.close()

#------------------------------
#VISUALISATIONS
#------------------------------

#Plots a bar chart of the different London boroughs against price
pltBar(avg_neighbourhood_df['neighbourhood_cleansed'], avg_neighbourhood_df['avg_price'], 'Neighbourhoods by Average Price', 'Neighbourhood', 'Average Price')
pltBar(avg_property_df['property_type'], avg_property_df['avg_price'], 'Property Type by Average Price', 'Property Type', 'Average Price')

#==============================
#SQL VIEWS
#==============================

# Connect to your database
engine = sqlalchemy.create_engine('/Users/hariramgnanachandran/Documents/Data Science : Machine Learning Work/Projects/London AirBnB Analysis/Dataset/listings.csv')
conn = engine.connect()

# Create Views (run these once; IF NOT EXISTS used to avoid errors if they already exist)
create_views_sql = """
CREATE VIEW IF NOT EXISTS median_price_neighborhood_roomtype AS
SELECT 
    neighbourhood_cleansed,
    room_type,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price) AS median_price
FROM 
    listingsSQL
GROUP BY 
    neighbourhood_cleansed, room_type;

CREATE VIEW IF NOT EXISTS room_type_distribution AS
SELECT 
    room_type,
    COUNT(*) AS listing_count
FROM 
    listingsSQL
GROUP BY 
    room_type;

CREATE VIEW IF NOT EXISTS missing_reviews AS
SELECT 
    id,
    name,
    neighbourhood_cleansed,
    number_of_reviews,
    last_review
FROM 
    listingsSQL
WHERE 
    number_of_reviews = 0 OR last_review IS NULL;
"""

 # Execute the create view statements
with engine.begin() as connection:
    connection.execute(create_views_sql)

# Query views and export results
median_price_df = pd.read_sql("SELECT * FROM median_price_neighborhood_roomtype", conn)
median_price_df.to_csv('/Users/hariramgnanachandran/Documents/Data Science : Machine Learning Work/Projects/London AirBnB Analysis/Outputs/SQL Results/SQL - Repeatable Queries/median_price_neighborhood_roomtype.csv', index=False)

room_type_dist_df = pd.read_sql("SELECT * FROM room_type_distribution", conn)
room_type_dist_df.to_csv('/Users/hariramgnanachandran/Documents/Data Science : Machine Learning Work/Projects/London AirBnB Analysis/Outputs/SQL Results/SQL - Repeatable Queries/room_type_distribution.csv', index=False)

missing_reviews_df = pd.read_sql("SELECT * FROM missing_reviews ORDER BY neighbourhood_cleansed, id", conn)
missing_reviews_df.to_csv('/Users/hariramgnanachandran/Documents/Data Science : Machine Learning Work/Projects/London AirBnB Analysis/Outputs/SQL Results/SQL - Repeatable Queries/missing_reviews.csv', index=False)

print("✅ Views created, queried, and results exported.")

conn.close()