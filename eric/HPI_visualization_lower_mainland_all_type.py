import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('cleaned_data/Cleaned_property_HPI_data.csv')

# Convert the 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Filter data for each residential type
apartment_data = data[data['Residential Type'] == 'Apartment']
townhouse_data = data[data['Residential Type'] == 'Townhouse']
detached_data = data[data['Residential Type'] == 'Detached']
all_types_data = data[data['Residential Type'] == 'Residential - All Types']

# Create a single plot
fig, ax = plt.subplots(figsize=(12, 8))

# Plot each residential type on the same graph
ax.plot(apartment_data['Date'], apartment_data['Price Index'], label='Apartment', color='b')
ax.plot(townhouse_data['Date'], townhouse_data['Price Index'], label='Townhouse', color='g')
ax.plot(detached_data['Date'], detached_data['Price Index'], label='Detached', color='r')
ax.plot(all_types_data['Date'], all_types_data['Price Index'], label='All Types', color='m')

# Set titles and labels
ax.set_title('Price Index for Different Residential Types')
ax.set_xlabel('Date')
ax.set_ylabel('Price Index')

# Add a legend to distinguish between the lines
ax.legend()

# Adjust layout
plt.tight_layout()

# Save the plot to a file
plt.savefig('price_index_comparison.png')

# Show the plot
plt.show()
