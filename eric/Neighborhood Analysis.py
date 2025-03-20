import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point

# Load the property data
property_data = pd.read_csv('cleaned_data/filteredPropertyInVancouver.csv')

# Filter the properties for sale only
property_data = property_data[property_data['Price Type'] == 'For Sale']

# Load the neighborhood boundary data
neighborhoods = gpd.read_file('raw_data/merged_vancouver_burnaby.geojson')

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

# Remove commas from the Price column and convert to float
property_gdf['Price'] = property_gdf['Price'].replace({',': ''}, regex=True).astype(float)

# Create a new column for bedroom categories with 4 or more combined
property_gdf['Beds_Category'] = property_gdf['Beds'].apply(lambda x: '4+' if x >= 4 else str(x))

# Group by Neighborhood and Beds_Category
grouped = property_gdf.groupby(['Neighborhood', 'Beds_Category'])

# Calculate price statistics for each group
price_stats = grouped['Price'].agg(
    mean='mean',
    median='median',
    q25=lambda x: np.percentile(x, 25),
    q75=lambda x: np.percentile(x, 75),
    q100=lambda x: np.percentile(x, 100)
)

# Calculate the percentage of properties in each group
neighborhood_beds_count = grouped.size()
total_properties = len(property_gdf)
neighborhood_beds_percentage = (neighborhood_beds_count / total_properties) * 100

# Merge the percentage and price stats into a single DataFrame
neighborhood_beds_summary = pd.DataFrame({
    'Percentage': neighborhood_beds_percentage,
}).join(price_stats)

# Output the result to a CSV file
neighborhood_beds_summary.to_csv('neighborhood_beds_summary.csv')

print("Neighborhood and bedroom summary has been saved to 'neighborhood_beds_summary.csv'")
