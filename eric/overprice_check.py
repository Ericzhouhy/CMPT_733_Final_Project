import pandas as pd

# Load the property data
property_data = pd.read_csv('cleaned_data/property_with_neighborhood.csv')
print(property_data.columns)

# Remove rows with any missing values (NA)
property_data.dropna(inplace=True)

# Load the neighborhood summary data
neighborhood_summary = pd.read_csv('cleaned_data/neighborhood_beds_summary.csv')

# Function to tag each property based on its price
def tag_price(row, summary):
    # Find the corresponding price stats for the property's neighborhood and beds category
    neighborhood = row['Neighborhood']
    beds = row['Beds']
    
    # Handle the case for 4+ beds
    beds_category = '4+' if beds >= 4 else str(beds)
    
    # Filter the summary for the specific neighborhood and beds category
    stats = summary[(summary['Neighborhood'] == neighborhood) & (summary['Beds_Category'] == beds_category)]
    
    if not stats.empty:
        q25 = stats['q25'].values[0]
        q75 = stats['q75'].values[0]
        price = row['Price']
        
        if price < q25:
            return 'Lower priced'
        elif price > q75:
            return 'Over priced'
        else:
            return 'Fair price'
    
    return 'Unknown'

# Add a column for the price tag
property_data['Price Tag'] = property_data.apply(lambda row: tag_price(row, neighborhood_summary), axis=1)

# Save the tagged data to a new CSV file
property_data.to_csv('cleaned_data/filteredPropertyWithPriceTags.csv', index=False)

print("Processed property data with price tags has been saved to 'filteredPropertyWithPriceTags.csv'")
