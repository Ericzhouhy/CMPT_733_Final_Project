import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point

# Load the property data
property_data = pd.read_csv('cleaned_data/filteredPropertyInVancouver.csv')

# Filter the properties for sale only
property_data = property_data[property_data['Price Type'] == 'For Sale']

# Load the neighborhood boundary data
neighborhoods = gpd.read_file('raw_data/local-area-boundary.geojson')

# Convert the property coordinates to GeoDataFrame
geometry = [Point(xy) for xy in zip(property_data['Longitude'], property_data['Latitude'])]
property_gdf = gpd.GeoDataFrame(property_data, geometry=geometry)

# Ensure the CRS (coordinate reference system) is the same as the neighborhood boundaries
property_gdf = property_gdf.set_crs(neighborhoods.crs, allow_override=True)

# Function to find the neighborhood for each property
def find_neighborhood(row):
    # Check which neighborhood the property belongs to
    for _, neighborhood in neighborhoods.iterrows():
        if neighborhood['geometry'].contains(row['geometry']):
            return neighborhood['name']  # Assuming 'name' is the column for neighborhood names
    return 'others'

# Assign the neighborhood to each property
property_gdf['Neighborhood'] = property_gdf.apply(find_neighborhood, axis=1)

# Calculate the percentage of properties in each neighborhood
neighborhood_count = property_gdf['Neighborhood'].value_counts()
total_properties = len(property_gdf)
neighborhood_percentage = (neighborhood_count / total_properties) * 100

# Calculate the price statistics for each neighborhood
property_gdf['Price'] = property_gdf['Price'].replace({',': ''}, regex=True).astype(float)  # Remove commas and convert to float

price_stats = property_gdf.groupby('Neighborhood')['Price'].agg(
    mean='mean',
    median='median',
    q25=lambda x: np.percentile(x, 25),
    q50=lambda x: np.percentile(x, 50),
    q100=lambda x: np.percentile(x, 100)
)

# Merge the percentage and price stats into a single DataFrame
neighborhood_summary = pd.DataFrame({
    'Percentage': neighborhood_percentage,
}).join(price_stats)

# Output the result to a CSV file
neighborhood_summary.to_csv('neighborhood_summary.csv')

print("Neighborhood summary has been saved to 'neighborhood_summary.csv'")
