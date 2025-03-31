import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings

# 屏蔽一些警告（如果需要）
warnings.filterwarnings("ignore", category=UserWarning)

# 读取合并好数据的 CSV 文件
df = pd.read_csv('../cleaned_data/mergeData.csv')
df.sort_values("Date", inplace=True)
df.reset_index(drop=True, inplace=True)

# 创建滞后特征：1期、2期、3期、6期、12期以及一阶差分（以部分变量为例）
for col in ['Benchmark', 'Rate', 'CPI_Value', 'AvgMonthlyEarnings', 'Unemployment_rate']:
    df[f'{col}_lag1'] = df[col].shift(1)
    df[f'{col}_lag2'] = df[col].shift(2)
    df[f'{col}_lag3'] = df[col].shift(3)
    df[f'{col}_lag6'] = df[col].shift(6)
    df[f'{col}_lag12'] = df[col].shift(12)
    df[f'{col}_diff'] = df[col].diff()

# 转换日期格式，并提取时间特征
df['Date'] = pd.to_datetime(df['Date'])
df['month'] = df['Date'].dt.month
df['quarter'] = df['Date'].dt.quarter
df['year'] = df['Date'].dt.year

# 创建滚动均值和滚动标准差特征（3个月窗口），示例仅针对 Benchmark 和 Rate
df['Benchmark_roll3'] = df['Benchmark'].rolling(window=3).mean()
df['Rate_roll3'] = df['Rate'].rolling(window=3).mean()
df['Benchmark_roll_std3'] = df['Benchmark'].rolling(window=3).std()
df['Rate_roll_std3'] = df['Rate'].rolling(window=3).std()

# 删除因滞后、滚动计算产生的 NaN 行
df = df.dropna().copy()

# 构造目标变量：预测下个月的 Benchmark
df['Next_Month_Benchmark'] = df['Benchmark'].shift(-1)
df_model = df.dropna(subset=['Next_Month_Benchmark']).copy()

# 定义输入特征
feature_cols = [
    'Benchmark', 'Price Index', 'Rate', 'CPI_Value', 'AvgMonthlyEarnings', 'Unemployment_rate',
    'Benchmark_lag1', 'Benchmark_lag2', 'Benchmark_lag3', 'Benchmark_lag6',
    'Rate_lag1', 'Rate_lag2', 'Rate_lag3', 'Rate_lag6',
    'Benchmark_roll3', 'Benchmark_roll_std3', 'Rate_roll3', 'Rate_roll_std3',
    'month', 'quarter', 'year',
    'Benchmark_lag12', 'Benchmark_diff', 'Rate_lag12', 'Rate_diff'
]

# 确保特征为数值型
for col in feature_cols:
    df_model[col] = pd.to_numeric(df_model[col], errors='coerce')
df_model = df_model.dropna().copy()

X = df_model[feature_cols]
y = df_model['Next_Month_Benchmark']

# 划分训练集和测试集，保持时间顺序（80%训练，20%测试）
train_size = int(len(df_model) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

#############################
# 方案1：使用 PCA 的 RandomForest
#############################
# pipeline_pca = Pipeline([
#     ('scaler', StandardScaler()),
#     ('pca', PCA()),
#     ('rf', RandomForestRegressor(random_state=42))
# ])

# param_dist_pca = {
#     'pca__n_components': [5, 7, 10, 15, 18],
#     'rf__n_estimators': [100, 300, 500, 700],
#     'rf__max_depth': [None, 10, 20, 30],
#     'rf__min_samples_split': [2, 5, 10],
#     'rf__min_samples_leaf': [1, 2, 4],
#     'rf__max_features': ['sqrt', 0.5]
# }

tscv = TimeSeriesSplit(n_splits=5)
# random_search_pca = RandomizedSearchCV(
#     pipeline_pca,
#     param_distributions=param_dist_pca,
#     cv=tscv,
#     scoring='neg_mean_squared_error',
#     n_iter=20,
#     n_jobs=1,
#     random_state=42
# )

# random_search_pca.fit(X_train, y_train)
# print("使用 PCA 的最佳参数：", random_search_pca.best_params_)

# y_pred_pca = random_search_pca.predict(X_test)
# rmse_pca = np.sqrt(mean_squared_error(y_test, y_pred_pca))
# r2_pca = r2_score(y_test, y_pred_pca)

# print("使用 PCA 测试集 RMSE:", rmse_pca)
# print("使用 PCA 测试集 R²:", r2_pca)

#############################
# 方案2：不使用 PCA 的 RandomForest
#############################
pipeline_no_pca = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(random_state=42))
])

param_dist_no_pca = {
    'rf__n_estimators': [100, 300, 500, 700],
    'rf__max_depth': [None, 10, 20, 30],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4],
    'rf__max_features': ['sqrt', 0.5]
}

random_search_no_pca = RandomizedSearchCV(
    pipeline_no_pca,
    param_distributions=param_dist_no_pca,
    cv=tscv,
    scoring='neg_mean_squared_error',
    n_iter=20,
    n_jobs=1,
    random_state=42
)

random_search_no_pca.fit(X_train, y_train)
print("不使用 PCA 的最佳参数：", random_search_no_pca.best_params_)

y_pred_no_pca = random_search_no_pca.predict(X_test)
rmse_no_pca = np.sqrt(mean_squared_error(y_test, y_pred_no_pca))
r2_no_pca = r2_score(y_test, y_pred_no_pca)

print("不使用 PCA 测试集 RMSE:", rmse_no_pca)
print("不使用 PCA 测试集 R²:", r2_no_pca)

# 添加下面的代码查看特征重要性
best_rf = random_search_no_pca.best_estimator_.named_steps['rf']
importances = best_rf.feature_importances_
features = X_train.columns
feature_importance = pd.DataFrame(
    {'feature': features, 'importance': importances})
print(feature_importance.sort_values('importance', ascending=False))
