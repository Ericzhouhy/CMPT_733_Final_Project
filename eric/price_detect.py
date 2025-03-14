import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the data
data_path = "cleaned_data/propertyInVancouver.csv"
df = pd.read_csv(data_path)

# Convert Price to numeric (remove commas and convert to float)
df['Price'] = df['Price'].replace({',': ''}, regex=True).astype(float)

# Create the 'Price Category' column for binary classification (overpriced vs underpriced)
average_price = df['Price'].mean()
df['Price Category'] = (df['Price'] > average_price).astype(int)

# Simplify the dataset by rounding latitude and longitude to group by neighborhood
df['Latitude Rounded'] = df['Latitude'].round(3)
df['Longitude Rounded'] = df['Longitude'].round(3)

# Group by the rounded latitude and longitude and calculate the average price in each group
df['Avg Price in Neighborhood'] = df.groupby(['Latitude Rounded', 'Longitude Rounded'])['Price'].transform('mean')

# Features for binary classification
X_binary = df[['Beds', 'Baths', 'Avg Price in Neighborhood']]
y_binary = df['Price Category']

# Train-test split for binary classification
X_train_binary, X_test_binary, y_train_binary, y_test_binary = train_test_split(X_binary, y_binary, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model for binary classification
model_binary = LogisticRegression()
model_binary.fit(X_train_binary, y_train_binary)

# Make predictions for binary classification
y_pred_binary = model_binary.predict(X_test_binary)

# Evaluate the binary classification model
binary_accuracy = accuracy_score(y_test_binary, y_pred_binary)
print(f"Binary Classification Accuracy: {binary_accuracy:.2f}")

# Define price bins for multi-class classification
price_bins = [0, df['Price'].quantile(0.33), df['Price'].quantile(0.66), df['Price'].max()]
price_labels = ['Low-priced', 'Mid-range', 'High-end']

# Create the 'Price Category Multi' column for multi-class classification
df['Price Category Multi'] = pd.cut(df['Price'], bins=price_bins, labels=price_labels)

# Drop rows with NaN values in the multi-class target variable (Price Category Multi)
df = df.dropna(subset=['Price Category Multi'])

# Features for multi-class classification
X_multi = df[['Beds', 'Baths', 'Avg Price in Neighborhood']]
y_multi = df['Price Category Multi']

# Train-test split for multi-class classification
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

# Initialize and train the SVM model for multi-class classification
model_multi = SVC(kernel='linear', decision_function_shape='ovr')
model_multi.fit(X_train_multi, y_train_multi)

# Make predictions for multi-class classification
y_pred_multi = model_multi.predict(X_test_multi)

# Evaluate the multi-class classification model
print("\nMulti-Class Classification Report:")
print(classification_report(y_test_multi, y_pred_multi))
