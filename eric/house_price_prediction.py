import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Step 1: Load the dataset from the specified path
file_path = "cleaned_data/cleaned_property_HPI_data.csv"
df = pd.read_csv(file_path)

# Step 2: Filter the data for 'Residential Type' == 'Residential - All Types'
df = df[df['Residential Type'] == 'Residential - All Types']

# Step 3: Clean the 'Benchmark' column (remove dollar signs and commas)
df['Benchmark'] = df['Benchmark'].replace({'\$': '', ',': ''}, regex=True).astype(float)

# Step 4: Feature engineering
# Convert 'Date' to datetime and extract year and month
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m')
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# Step 5: Encode categorical variables (for this filtered dataset, Residential Type is constant)
df = pd.get_dummies(df, columns=['Residential Type'], drop_first=True)

# Step 6: Split the data into features (X) and target (y)
X = df.drop(['Date', 'Benchmark'], axis=1)  # Features
y = df['Benchmark']  # Target variable

# Step 7: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 8: Normalize the data (optional, but helps with some models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 9: Train a model (using Random Forest in this case)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Step 10: Predict and evaluate the model
y_pred = model.predict(X_test_scaled)

# Step 11: Model evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {r2}')

# Step 12: Plotting the true vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='True Values', color='blue', marker='o')
plt.plot(y_test.index, y_pred, label='Predicted Values', color='red', linestyle='--', marker='x')
plt.title('True vs Predicted Benchmark for Residential - All Types')
plt.xlabel('Time (Date)')
plt.ylabel('Benchmark ($)')
plt.legend()
plt.grid(True)
plt.show()

# Optional: Hyperparameter tuning using GridSearchCV (for Random Forest)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)
print("Best Parameters:", grid_search.best_params_)
