import folium #Library to create interactive leaflet maps
from folium.plugins import MarkerCluster #Library to help group nearby markers into clusters
from geopy.distance import geodesic #Library used for geographic calculations

def plotListings(df, baseCoord, columnsToShow):

    """
    Plot a map with markers for each location in the DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing location data with 'latitude' and 'longitude' columns.
    baseCoord : tuple or list
        Coordinates (latitude, longitude) to center the map.
    columnsToShow : list of str
        List of column names to display in the popup for each marker.

    Returns:
    --------
    folium.Map
        Folium map object with all location markers added.
    """

    #Create a map centered on the base coordinates, with marker clustering enabled
    map, markerCluster = createMap(baseCoord)

    #Iterate through each row in the DataFrame to add markers to the map
    for i, row in df.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=createPopup(row, columnsToShow),
            icon=folium.Icon(color='blue', icon='home')
        ).add_to(markerCluster)

    return map

def groupListings(df, baseCoord, columnsToShow, roomColours, sortVar):

    """
    Plot a map with color-coded markers based on a classification variable.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing location data with 'latitude' and 'longitude' columns.
    baseCoord : tuple or list
        Coordinates (latitude, longitude) to center the map.
    columnsToShow : list of str
        List of column names to display in the popup for each marker.
    roomColours : dict
        Dictionary mapping categories of the sortVar to marker colors.
    sortVar : str
        Column name used to determine marker color classification.

    Returns:
    --------
    folium.Map
        Folium map object with color-coded location markers added.
    """

    #Create a map centered on the base coordinates, with marker clustering enabled
    map, markerCluster = createMap(baseCoord)

    #Iterate over DataFrame rows to add colored circle markers based on sortVar
    for i, row in df.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']], #Marker location coordinates
            radius=5, #Marker size
            color=roomColours.get(row[sortVar], 'gray'), #Color from dict, default gray
            fill=True, #Enable fill color
            fill_opacity=0.7, #Marker opacity
            popup=createPopup(row, columnsToShow), #Popup showing selected info
    ).add_to(markerCluster) #Add marker to cluster group


    return map

def createMap(baseCoord):

    """
    Create a folium map centered on the provided base coordinates with marker clustering enabled.

    Parameters:
    -----------
    baseCoord : list or tuple
        Coordinates (latitude, longitude) to center the map.

    Returns:
    --------
    map : folium.Map
        Folium map object centered at baseCoord.
    markerCluster : folium.plugins.MarkerCluster
        MarkerCluster object for adding clustered markers to the map.
    """
  
   #Initialise the folium map centered at the base coordinates with a default zoom level
    map = folium.Map(location=baseCoord, zoom_start=11)

    #Create a MarkerCluster to efficiently manage multiple markers on the map
    markerCluster = MarkerCluster().add_to(map)

    return map, markerCluster

def createPopup(row, columns):

    """
    Create a formatted HTML popup text for a map marker based on specified columns from a data row.

    Parameters:
    -----------
    row : pandas.Series
        A single row from the dataset containing the data for one location.
    columns : list of str
        List of column names whose values will be included in the popup.

    Returns:
    --------
    popupText : str
        A string containing HTML formatted lines for each specified column and its value.
    """

    #Create a string joining each column name and its corresponding value from the row, separated by line breaks
    popupText = "<br>".join([f"{col}: {row[col]}" for col in columns])
    return popupText

def calculateDistance(row, locationCoords):

    """
    Calculate the geodesic distance in kilometers between a location in a data row and a key location.

    Parameters:
    -----------
    row : pandas.Series
        A row from the dataset containing 'latitude' and 'longitude' of the location.
    locationCoords : tuple or list
        Coordinates of the key location as (latitude, longitude).

    Returns:
    --------
    float
        Distance between the location in the row and the key location in kilometers.
    """
    
    #Extract latitude and longitude from the row as a tuple
    location = (row['latitude'], row['longitude'])

    # Calculate geodesic distance between the location and the key location
    return geodesic(location, locationCoords).km

def plotByDist(df, baseCoord, colour, dist):

    """
    Create a map with locations colour-coded based on their distance from a key location.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing location data with latitude, longitude, and distance.
    baseCoord : tuple or list
        Coordinates (latitude, longitude) for the center of the map.
    colour : str
        Name of the column in df containing color codes based on distance.
    dist : str
        Name of the column in df containing distance values from the key location.

    Returns:
    --------
    folium.Map
        Folium map object with color-coded location markers.
    """

    #Create a map centered on the base coordinates
    map, markerCluster = createMap(baseCoord)

    #Add locations as circle markers, colored based on distance to the key location
    for i, row in df.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color=row[colour],
            fill=True,
            fill_opacity=0.6,
            popup=f"Distance to Key Location: {row[dist]:.2f} km"
        ).add_to(markerCluster)

    return map

def getDistColours(row, distVar, distanceColors):

    """
    Assign a color based on the distance of a location from a key location.

    Parameters:
    -----------
    row : pandas.Series
        A row from the DataFrame containing location data.
    distVar : str
        Column name for the distance value in the DataFrame.
    distanceColors : list or tuple
        Two-element list/tuple with lower and upper bounds to assign colors.

    Returns:
    --------
    str
        Color ('green', 'yellow', or 'red') based on the distance thresholds.
    """
    
    #Extracts the distance from a location in the DataFrame to the key location
    dist = row[distVar]

    #Extract the lower and upper bounds from distanceColors
    lowDist = distanceColors[0]
    highDist = distanceColors[1]

    #Determines the colour to be assigned based on distance between location and key location       
    if dist <= lowDist:
        return 'green'
    elif lowDist < dist <= highDist:
        return 'yellow'
    else:
        return 'red'