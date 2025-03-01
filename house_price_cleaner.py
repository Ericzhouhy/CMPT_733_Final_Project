import pandas as pd

# Read the dataset
df = pd.read_csv("Raw_data/synthetic_house_prices_20_years.csv", index_col="Neighborhood")

# Display first few rows to inspect
print(df.head())

# ---- 1. Handle Missing Values ---- #
# Check for missing values
missing_values = df.isnull().sum()
print("Missing values:\n", missing_values)

# Fill missing values in "Renovation Year" (assuming 0 means no renovation)
df["Renovation Year"].fillna(0, inplace=True)

# Fill missing values in "Garage Type" (assuming "None" means no garage)
df["Garage Type"].fillna("None", inplace=True)

# Drop rows where "Market Price" is missing (since it's a critical feature)
df.dropna(subset=["Market Price"], inplace=True)

# ---- 2. Convert Data Types ---- #
# Convert year columns to integer
df["Year Built"] = df["Year Built"].astype(int)
df["Renovation Year"] = df["Renovation Year"].astype(int)

# Convert "Market Price" to float
df["Market Price"] = df["Market Price"].astype(float)

# ---- 3. Handle Outliers ---- #
# Detect and remove outliers in "Market Price" using the IQR method
q1 = df["Market Price"].quantile(0.25)
q3 = df["Market Price"].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# Keep only rows within the acceptable range
df = df[(df["Market Price"] >= lower_bound) & (df["Market Price"] <= upper_bound)]

# ---- 4. Standardize Categorical Data ---- #
# Trim spaces and convert "Garage Type" to lowercase
df["Garage Type"] = df["Garage Type"].str.strip().str.lower()

# Capitalize "Property Type" for consistency
df["Property Type"] = df["Property Type"].str.strip().str.capitalize()

# ---- 5. Remove Duplicates ---- #
df.drop_duplicates(inplace=True)

# ---- 6. Save the cleaned dataset ---- #
df.to_csv("cleaned_data/house_price20.csv", index=True)

# ---- 7. Display cleaned data ---- #
import ace_tools as tools
tools.display_dataframe_to_user(name="Cleaned Data", dataframe=df)
