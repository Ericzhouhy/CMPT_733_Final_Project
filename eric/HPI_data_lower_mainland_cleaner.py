import pandas as pd

# Load the data from the CSV file
df = pd.read_csv('raw_data/property_HPI_data.csv')

# Forward fill the missing date values
df['Date'] = df['Date'].fillna(method='ffill')

# Initialize a new column for Residential Type
df['Residential Type'] = None

# Fill the 'Residential Type' column based on the category rows
current_type = None
for index, row in df.iterrows():
    if row.iloc[1] in ['Residential - All Types', 'Detached', 'Townhouse', 'Apartment']:
        current_type = row.iloc[1]
    else:
        df.at[index, 'Residential Type'] = current_type

# Remove the category rows
df = df.dropna(subset=['Residential Type'])

# Sort the DataFrame by 'Date' in descending order
df = df.sort_values(by='Date', key=lambda x: pd.to_datetime(x, errors='coerce'), ascending=False)
# List of columns to remove
columns_to_remove = ['Price Index', '1 Month +/-', '6 Month +/-', '1 Year +/-', '3 Year +/-', '5 Year +/-']

# Remove the specified columns
df = df.drop(columns=columns_to_remove)
# Reset the index
df = df.reset_index(drop=True)

# Save the cleaned data to a new CSV file
df.to_csv('cleaned_lower_mainland_property_data.csv', index=False)

print("Data cleaning complete. Cleaned data saved to 'Cleaned_lower_mainland_property_data.csv'.")
