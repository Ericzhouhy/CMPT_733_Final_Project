import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("cleaned_data/Cleaned_property_HPI_data.csv")

# Convert Date to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Extract useful time features
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Quarter'] = data['Date'].dt.quarter

# Clean Benchmark column (use raw string for regex)
data['Benchmark'] = data['Benchmark'].replace(r'[\$,]', '', regex=True).astype(float)

# One-hot encode Residential Type
encoder = OneHotEncoder(sparse_output=False, drop='first')
res_type_encoded = encoder.fit_transform(data[['Residential Type']])
res_type_df = pd.DataFrame(res_type_encoded, columns=encoder.get_feature_names_out(['Residential Type']))
data = pd.concat([data[['Date', 'Year', 'Month', 'Quarter']], res_type_df, data['Benchmark']], axis=1)

# Train-test split
train_data = data[data['Year'] < 2024]
test_data = data[data['Year'] >= 2024]

# Separate Date for plotting
train_dates = train_data['Date']
test_dates = test_data['Date']

X_train = train_data.drop(columns=['Benchmark', 'Date'])
y_train = train_data['Benchmark']
X_test = test_data.drop(columns=['Benchmark', 'Date'])
y_test = test_data['Benchmark']

# Hyperparameter tuning with cross-validation
tscv = TimeSeriesSplit(n_splits=5)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=tscv, scoring='neg_root_mean_squared_error', verbose=1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

y_pred = best_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse:.2f}')

# Plot actual vs. predicted
plt.figure(figsize=(10, 5))
plt.plot(test_dates, y_test, label="Actual Prices", marker='o')
plt.plot(test_dates, y_pred, label="Predicted Prices", linestyle='dashed', marker='x')
plt.legend()
plt.title("Actual vs Predicted Benchmark Prices")
plt.xlabel("Date")
plt.ylabel("Benchmark Price")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
