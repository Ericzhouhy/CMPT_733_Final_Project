import pandas as pd

# Load the data
file_path = 'raw_data/vancouver_areas_property_data.csv'
df = pd.read_csv(file_path)

# Find indices where 'Benchmark' is 'Residential - All Types'
indices = df.index[df['Benchmark'] == 'Residential - All Types']

# Select rows immediately after each 'Residential - All Types' row
rows_to_keep = indices + 1
df_filtered = df.loc[rows_to_keep]

# Keep only the necessary columns: Date, Area, Benchmark, Price Index
df_cleaned = df_filtered[['Date', 'Area', 'Benchmark', 'Price Index']]

# Sort the data by Date
df_cleaned = df_cleaned.sort_values(by='Date')

# Reset index for cleanliness
df_cleaned = df_cleaned.reset_index(drop=True)

# Save the cleaned data
df_cleaned.to_csv('cleaned_vancouver_areas_property_data.csv', index=False)

print("Data cleaned and saved to cleaned_vancouver_areas_property_data.csv")
