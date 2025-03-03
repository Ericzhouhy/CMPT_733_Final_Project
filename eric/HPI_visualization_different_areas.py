import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = 'cleaned_data/cleaned_vancouver_areas_property_data.csv'
df = pd.read_csv(file_path)

# Convert 'Date' to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m')

# Remove currency symbols and commas, then convert to numeric
df['Benchmark'] = df['Benchmark'].replace({'\$': '', ',': ''}, regex=True)
df['Benchmark'] = pd.to_numeric(df['Benchmark'], errors='coerce')

# Sort the data by date
df = df.sort_values('Date')

# Calculate a rolling mean to smooth the trend for each area
df['RollingBenchmark'] = df.groupby('Area')['Benchmark'].transform(lambda x: x.rolling(window=12, min_periods=1).mean())

# Create a single plot
fig, ax = plt.subplots(figsize=(12, 8))

# Plot each area on the same graph
areas = df['Area'].unique()
colors = ['b', 'g', 'r', 'm', 'c', 'y', 'k', 'orange', 'purple']

for area, color in zip(areas, colors):
    area_data = df[df['Area'] == area]
    ax.plot(area_data['Date'], area_data['RollingBenchmark'], label=area, color=color)

# Set titles and labels
ax.set_title('Benchmark Trend Over Time by Area')
ax.set_xlabel('Date')
ax.set_ylabel('Benchmark')

# Add a legend to distinguish between the lines
ax.legend(title='Area', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout
plt.tight_layout()

# Save the plot to a file
plt.savefig('benchmark_trend_by_area.png')

# Show the plot
plt.show()
