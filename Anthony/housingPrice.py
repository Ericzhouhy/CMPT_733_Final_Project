# import pandas as pd
# import numpy as np
# from scipy import stats
# from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# import warnings
# import xgboost as xgb
# import matplotlib.pyplot as plt

# warnings.filterwarnings("ignore", category=UserWarning)

# # 读取数据
# df = pd.read_csv('../cleaned_data/mergeData.csv')
# df.sort_values("Date", inplace=True)
# df.reset_index(drop=True, inplace=True)

# # 创建滞后、差分特征
# for col in ['Benchmark', 'Rate', 'CPI_Value', 'AvgMonthlyEarnings', 'Unemployment_rate']:
#     df[f'{col}_lag1'] = df[col].shift(1)
#     df[f'{col}_lag2'] = df[col].shift(2)
#     df[f'{col}_lag3'] = df[col].shift(3)
#     df[f'{col}_lag6'] = df[col].shift(6)
#     df[f'{col}_lag12'] = df[col].shift(12)
#     df[f'{col}_diff'] = df[col].diff()

# # 时间特征
# df['Date'] = pd.to_datetime(df['Date'])
# df['month'] = df['Date'].dt.month
# df['quarter'] = df['Date'].dt.quarter
# df['year'] = df['Date'].dt.year

# # 滚动特征
# df['Benchmark_roll3'] = df['Benchmark'].rolling(window=3).mean()
# df['Rate_roll3'] = df['Rate'].rolling(window=3).mean()
# df['Benchmark_roll_std3'] = df['Benchmark'].rolling(window=3).std()
# df['Rate_roll_std3'] = df['Rate'].rolling(window=3).std()

# # 删除因特征构造产生的 NaN 行
# df = df.dropna().copy()

# # 构造目标变量：预测下个月的 Benchmark
# df['Next_Month_Benchmark'] = df['Benchmark'].shift(-1)
# df_model = df.dropna(subset=['Next_Month_Benchmark']).copy()

# # 对目标变量做对数变换：log(Next_Month_Benchmark + 1)
# df_model['log_Next_Month_Benchmark'] = np.log(
#     df_model['Next_Month_Benchmark'] + 1)

# # 对 Price Index 和 CPI_Value 做对数变换
# df_model['log_Price_Index'] = np.log(df_model['Price Index'])
# df_model['log_CPI_Value'] = np.log(df_model['CPI_Value'])

# # 定义初始特征列表（名称必须与 CSV 中一致）
# feature_cols_all = [
#     'Next_Month_Benchmark', 'log_Price_Index', 'Rate', 'log_CPI_Value', 'AvgMonthlyEarnings', 'Unemployment_rate',
#     'Benchmark_lag1', 'Benchmark_lag2', 'Benchmark_lag3', 'Benchmark_lag6',
#     'Rate_lag1', 'Rate_lag2', 'Rate_lag3', 'Rate_lag6',
#     'Benchmark_roll3', 'Benchmark_roll_std3', 'Rate_roll3', 'Rate_roll_std3',
#     'month', 'quarter', 'year',
#     'Benchmark_lag12', 'Benchmark_diff', 'Rate_lag12', 'Rate_diff',
#     'Number of Commercial Building Permits',
#     'Number of Industrial Building Permits',
#     'Number of Institutional and Government Building Permits',
#     'Number of single dwelling units', 'Number of apartment units',
#     'row dwelling units', 'totalNumber'
# ]

# # 根据之前的特征重要性结果，移除贡献极低的 Rate 相关特征及 Rate_diff
# features_to_remove = [
#     'Rate', 'Rate_lag1', 'Rate_lag2', 'Rate_lag3', 'Rate_lag6',
#     'Rate_roll3', 'Rate_roll_std3', 'Rate_lag12', 'Rate_diff'
# ]

# # 新的特征列表（注意此处暂不使用移除后的列表训练模型，用全部特征进行对比，后面可按需求调整）
# feature_cols_selected = [
#     feat for feat in feature_cols_all if feat not in features_to_remove]
# # print("移除低贡献特征后的特征列表：", feature_cols_selected)

# # 将所有特征转换为数值型
# for col in feature_cols_all:
#     df_model[col] = pd.to_numeric(df_model[col], errors='coerce')
# df_model = df_model.dropna(
#     subset=feature_cols_selected + ['Next_Month_Benchmark']).copy()

# # 构造特征矩阵 X 与目标 y（这里用 log_Next_Month_Benchmark 作为目标）
# X = df_model[feature_cols_all]
# y = df_model['log_Next_Month_Benchmark']

# # 划分训练集和测试集（保持时间顺序）
# train_size = int(len(df_model) * 0.8)
# X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
# y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# # 定义时间序列交叉验证
# tscv = TimeSeriesSplit(n_splits=5)

# ###########################################
# # 模型1：随机森林（不使用 PCA）改进版
# ###########################################
# pipeline_rf = Pipeline([
#     ('scaler', StandardScaler()),
#     ('rf', RandomForestRegressor(random_state=42))
# ])
# param_dist_rf = {
#     'rf__n_estimators': [100, 300, 500, 700, 1000],
#     'rf__max_depth': [None, 10, 20, 30, 40],
#     'rf__min_samples_split': [2, 5, 10],
#     'rf__min_samples_leaf': [1, 2, 4, 6],
#     'rf__max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7]
# }
# random_search_rf = RandomizedSearchCV(
#     pipeline_rf,
#     param_distributions=param_dist_rf,
#     cv=tscv,
#     scoring='neg_mean_squared_error',
#     n_iter=40,
#     n_jobs=1,
#     random_state=42
# )
# random_search_rf.fit(X_train, y_train)
# y_pred_rf = random_search_rf.predict(X_test)
# rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
# r2_rf = r2_score(y_test, y_pred_rf)
# print("改进后的 RandomForest 测试集 RMSE (log尺度):", rmse_rf)
# print("改进后的 RandomForest 测试集 R² (log尺度):", r2_rf)

# ###########################################
# # 模型2：XGBoost 模型示例
# ###########################################
# pipeline_xgb = Pipeline([
#     ('scaler', StandardScaler()),
#     ('xgb', xgb.XGBRegressor(random_state=42))
# ])
# param_dist_xgb = {
#     'xgb__n_estimators': [100, 300, 500, 700],
#     'xgb__max_depth': [3, 5, 7, 10],
#     'xgb__learning_rate': [0.01, 0.05, 0.1, 0.2],
#     'xgb__subsample': [0.6, 0.8, 1.0],
#     'xgb__colsample_bytree': [0.6, 0.8, 1.0]
# }
# random_search_xgb = RandomizedSearchCV(
#     pipeline_xgb,
#     param_distributions=param_dist_xgb,
#     cv=tscv,
#     scoring='neg_mean_squared_error',
#     n_iter=40,
#     n_jobs=1,
#     random_state=42
# )
# random_search_xgb.fit(X_train, y_train)
# y_pred_xgb = random_search_xgb.predict(X_test)
# rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
# r2_xgb = r2_score(y_test, y_pred_xgb)
# print("XGBoost 测试集 RMSE (log尺度):", rmse_xgb)
# print("XGBoost 测试集 R² (log尺度):", r2_xgb)

# ###########################################
# # 将 log 预测结果反变换回原始尺度
# ###########################################
# y_pred_rf_orig = np.exp(y_pred_rf) - 1
# y_pred_xgb_orig = np.exp(y_pred_xgb) - 1
# y_test_orig = np.exp(y_test) - 1

# ###########################################
# # 计算原始尺度下的评估指标
# ###########################################
# rmse_rf_orig = np.sqrt(mean_squared_error(y_test_orig, y_pred_rf_orig))
# r2_rf_orig = r2_score(y_test_orig, y_pred_rf_orig)
# print("RandomForest 原始尺度 测试集 RMSE:", rmse_rf_orig)
# print("RandomForest 原始尺度 测试集 R²:", r2_rf_orig)

# rmse_xgb_orig = np.sqrt(mean_squared_error(y_test_orig, y_pred_xgb_orig))
# r2_xgb_orig = r2_score(y_test_orig, y_pred_xgb_orig)
# print("XGBoost 原始尺度 测试集 RMSE:", rmse_xgb_orig)
# print("XGBoost 原始尺度 测试集 R²:", r2_xgb_orig)

# ###########################################
# # 绘制预测值与真实值的散点图（原始尺度）
# ###########################################
# plt.figure(figsize=(8, 6))
# plt.scatter(y_test_orig, y_pred_rf_orig, alpha=0.6, label='RandomForest')
# plt.scatter(y_test_orig, y_pred_xgb_orig, alpha=0.6, label='XGBoost')
# plt.plot([min(y_test_orig), max(y_test_orig)], [
#          min(y_test_orig), max(y_test_orig)], 'k--', lw=2)
# plt.xlabel("真实 Benchmark")
# plt.ylabel("预测 Benchmark")
# plt.title("预测值 vs 真实值（原始尺度）")
# plt.legend()
# plt.show()

# ###########################################
# # 绘制残差分布图（原始尺度）
# ###########################################
# rf_residuals = y_test_orig - y_pred_rf_orig
# xgb_residuals = y_test_orig - y_pred_xgb_orig

# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.hist(rf_residuals, bins=30, alpha=0.7)
# plt.title("RandomForest 残差分布")
# plt.xlabel("残差")
# plt.ylabel("频数")

# plt.subplot(1, 2, 2)
# plt.hist(xgb_residuals, bins=30, alpha=0.7)
# plt.title("XGBoost 残差分布")
# plt.xlabel("残差")
# plt.ylabel("频数")
# plt.tight_layout()
# plt.show()


import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings
import xgboost as xgb
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------
# 1. 数据预处理及特征构造
# ---------------------------
df = pd.read_csv('../cleaned_data/mergeData.csv')
df.sort_values("Date", inplace=True)
df.reset_index(drop=True, inplace=True)

# 创建滞后、差分特征
for col in ['Benchmark', 'Rate', 'CPI_Value', 'AvgMonthlyEarnings', 'Unemployment_rate']:
    df[f'{col}_lag1'] = df[col].shift(1)
    df[f'{col}_lag2'] = df[col].shift(2)
    df[f'{col}_lag3'] = df[col].shift(3)
    df[f'{col}_lag6'] = df[col].shift(6)
    df[f'{col}_lag12'] = df[col].shift(12)
    df[f'{col}_diff'] = df[col].diff()

# 时间特征
df['Date'] = pd.to_datetime(df['Date'])
df['month'] = df['Date'].dt.month
df['quarter'] = df['Date'].dt.quarter
df['year'] = df['Date'].dt.year

# 滚动特征
df['Benchmark_roll3'] = df['Benchmark'].rolling(window=3).mean()
df['Rate_roll3'] = df['Rate'].rolling(window=3).mean()
df['Benchmark_roll_std3'] = df['Benchmark'].rolling(window=3).std()
df['Rate_roll_std3'] = df['Rate'].rolling(window=3).std()

# 删除因特征构造产生的 NaN 行
df = df.dropna().copy()

# 构造目标变量：预测下个月的 Benchmark（直接使用原始值，不做对数变换）
df['Next_Month_Benchmark'] = df['Benchmark'].shift(-1)
df_model = df.dropna(subset=['Next_Month_Benchmark']).copy()

# ---------------------------
# 2. 定义特征列表
# ---------------------------
feature_cols_all = [
    'Next_Month_Benchmark', 'Price Index', 'Rate', 'CPI_Value', 'AvgMonthlyEarnings', 'Unemployment_rate',
    'Benchmark_lag1', 'Benchmark_lag2', 'Benchmark_lag3', 'Benchmark_lag6',
    'Rate_lag1', 'Rate_lag2', 'Rate_lag3', 'Rate_lag6',
    'Benchmark_roll3', 'Benchmark_roll_std3', 'Rate_roll3', 'Rate_roll_std3',
    'month', 'quarter', 'year',
    'Benchmark_lag12', 'Benchmark_diff', 'Rate_lag12', 'Rate_diff',
    'Number of Commercial Building Permits',
    'Number of Industrial Building Permits',
    'Number of Institutional and Government Building Permits',
    'Number of single dwelling units', 'Number of apartment units',
    'row dwelling units', 'totalNumber'
]

# 如果需要移除贡献较低的 Rate 相关特征，可以设置：
features_to_remove = [
    'Rate', 'Rate_lag1', 'Rate_lag2', 'Rate_lag3', 'Rate_lag6',
    'Rate_roll3', 'Rate_roll_std3', 'Rate_lag12', 'Rate_diff'
]
feature_cols_selected = [
    feat for feat in feature_cols_all if feat not in features_to_remove]

# 将所有特征转换为数值型，并删除缺失值
for col in feature_cols_all:
    df_model[col] = pd.to_numeric(df_model[col], errors='coerce')
df_model = df_model.dropna(
    subset=feature_cols_selected + ['Next_Month_Benchmark']).copy()

# ---------------------------
# 3. 按日期划分训练集和测试集
# ---------------------------
# 假设训练集截止到 2023-01-31，测试集从 2023-02-01 开始
train_df = df_model[df_model['Date'] <= '2023-01-31'].copy()
test_df = df_model[df_model['Date'] >= '2023-02-01'].copy()

print("训练集日期范围:", train_df['Date'].min(), "-", train_df['Date'].max())
print("测试集日期范围:", test_df['Date'].min(), "-", test_df['Date'].max())

X_train = train_df[feature_cols_all]
y_train = train_df['Next_Month_Benchmark']
X_test = test_df[feature_cols_all]
y_test = test_df['Next_Month_Benchmark']


# def add_noise(X, noise_factor):
#     X_noisy = X.copy()
#     for col in X_noisy.columns:
#         std = X_noisy[col].std()
#         X_noisy[col] += np.random.normal(0,
#                                          noise_factor * std, size=X_noisy.shape[0])
#     return X_noisy

# noise_factor = 1.6  # 可以调整这个比例
# X_train_noisy_rf = add_noise(X_train, noise_factor)
# X_train_noisy_xgb = add_noise(X_train, noise_factor)


# ---------------------------
# 4. 模型训练与调参
# ---------------------------
tscv = TimeSeriesSplit(n_splits=5)

# 模型1：随机森林（降低复杂度）
pipeline_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(random_state=42))
])
param_dist_rf = {
    'rf__n_estimators': [100, 200, 300],
    'rf__max_depth': [5, 7, 10],
    'rf__min_samples_split': [10, 15, 20],
    'rf__min_samples_leaf': [5, 10],
    'rf__max_features': ['sqrt', 0.5]
}
random_search_rf = RandomizedSearchCV(
    pipeline_rf,
    param_distributions=param_dist_rf,
    cv=tscv,
    scoring='neg_mean_squared_error',
    n_iter=40,
    n_jobs=1,
    random_state=42
)
random_search_rf.fit(X_train, y_train)
y_pred_rf_noisy = random_search_rf.predict(X_test)
rmse_rf_noisy = np.sqrt(mean_squared_error(y_test, y_pred_rf_noisy))
r2_rf_noisy = r2_score(y_test, y_pred_rf_noisy)
print("添加噪声后的随机森林 测试集 RMSE:", rmse_rf_noisy)
print("添加噪声后的随机森林 测试集 R²:", r2_rf_noisy)

# 模型2：XGBoost（不使用早停法）
# 模型2：XGBoost（不使用早停法，添加噪声）
pipeline_xgb = Pipeline([
    ('scaler', StandardScaler()),
    ('xgb', xgb.XGBRegressor(random_state=42))
])
param_dist_xgb = {
    'xgb__n_estimators': [300, 500, 700],
    'xgb__max_depth': [3, 4, 5],
    'xgb__learning_rate': [0.01, 0.05],
    'xgb__subsample': [0.7, 0.8, 0.9],
    'xgb__colsample_bytree': [0.7, 0.8, 0.9],
    'xgb__reg_lambda': [1, 2, 5],
    'xgb__reg_alpha': [0, 0.1, 0.5]
}
random_search_xgb = RandomizedSearchCV(
    pipeline_xgb,
    param_distributions=param_dist_xgb,
    cv=tscv,
    scoring='neg_mean_squared_error',
    n_iter=40,
    n_jobs=1,
    random_state=42
)
random_search_xgb.fit(X_train, y_train)
y_pred_xgb_noisy = random_search_xgb.predict(X_test)
rmse_xgb_noisy = np.sqrt(mean_squared_error(y_test, y_pred_xgb_noisy))
r2_xgb_noisy = r2_score(y_test, y_pred_xgb_noisy)
print("添加噪声后的 XGBoost 测试集 RMSE:", rmse_xgb_noisy)
print("添加噪声后的 XGBoost 测试集 R²:", r2_xgb_noisy)

# ---------------------------
# 5. 验证模型预测效果
# ---------------------------
# 绘制预测值与真实值的散点图（原始尺度）
# plt.figure(figsize=(8, 6))
# plt.scatter(y_test, y_pred_rf_noisy, alpha=0.6, label='RandomForest')
# plt.scatter(y_test, y_pred_xgb_noisy, alpha=0.6, label='XGBoost')
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)
# plt.xlabel("Real Benchmark")
# plt.ylabel("Predicted Benchmark")
# plt.title("Predicted vs Real Benchmark")
# plt.legend()
# plt.show()

# # 绘制残差分布图（原始尺度）
# rf_residuals = y_test - y_pred_rf_noisy
# xgb_residuals = y_test - y_pred_xgb_noisy

# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.hist(rf_residuals, bins=30, alpha=0.7)
# plt.title("RandomForest Residual Distribution")
# plt.xlabel("Residual")
# plt.ylabel("Frequency")

# plt.subplot(1, 2, 2)
# plt.hist(xgb_residuals, bins=30, alpha=0.7)
# plt.title("XGBoost Residual Distribution")
# plt.xlabel("Residual")
# plt.ylabel("Frequency")
# plt.tight_layout()
# plt.show()


# 确保 test_df 的 Date 是 datetime 类型
# test_df['Date'] = pd.to_datetime(test_df['Date'])

# # 筛选 2024 年的测试数据（1月到12月）
# test_2024 = test_df[(test_df['Date'] >= '2024-01-01') &
#                     (test_df['Date'] < '2025-01-01')].copy()

# # 使用训练好的 RandomForest 模型进行预测
# X_2024 = test_2024[feature_cols_all]
# y_true_2024 = test_2024['Next_Month_Benchmark']
# y_pred_2024 = random_search_rf.predict(X_2024)

# # 将预测结果加到 test_2024 中
# test_2024['Predicted_RF'] = y_pred_2024

# # 提取月份
# test_2024['Month'] = test_2024['Date'].dt.to_period('M').dt.to_timestamp()

# # 按月聚合平均
# monthly_avg = test_2024.groupby(
#     'Month')[['Next_Month_Benchmark', 'Predicted_RF']].mean().reset_index()

# # 绘图：每月平均真实值 vs 预测值
# plt.figure(figsize=(12, 6))
# plt.plot(monthly_avg['Month'], monthly_avg['Next_Month_Benchmark'],
#          label='True Benchmark', marker='o')
# plt.plot(monthly_avg['Month'], monthly_avg['Predicted_RF'],
#          label='Predicted Benchmark (RandomForest)', marker='x')
# plt.xlabel("Month")
# plt.ylabel("Benchmark")
# plt.title("2024 Monthly Benchmark: True vs Predicted (RandomForest)")
# plt.xticks(rotation=45)
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # 构建对比 DataFrame（按月）
# comparison_df = monthly_avg.rename(columns={
#     'Month': 'Date',
#     'Next_Month_Benchmark': 'True_Benchmark',
#     'Predicted_RF': 'Predicted_Benchmark'
# })

# # 打印前几行
# print(comparison_df.head())
# 确保 test_df 的 Date 是 datetime 类型
test_df['Date'] = pd.to_datetime(test_df['Date'])

# 筛选 2024 年的测试数据（1月到12月）
test_2024 = test_df[(test_df['Date'] >= '2024-01-01') &
                    (test_df['Date'] < '2025-01-01')].copy()

# 使用训练好的 XGBoost 模型进行预测
X_2024 = test_2024[feature_cols_all]
y_true_2024 = test_2024['Next_Month_Benchmark']
y_pred_2024 = random_search_xgb.predict(X_2024)

# 将预测结果加入 test_2024
test_2024['Predicted_XGB'] = y_pred_2024

# 提取月份
test_2024['Month'] = test_2024['Date'].dt.to_period('M').dt.to_timestamp()

# 按月聚合平均值
monthly_avg = test_2024.groupby(
    'Month')[['Next_Month_Benchmark', 'Predicted_XGB']].mean().reset_index()

# 绘图：每月平均真实值 vs XGBoost 预测值
plt.figure(figsize=(12, 6))
plt.plot(monthly_avg['Month'], monthly_avg['Next_Month_Benchmark'],
         label='True Benchmark', marker='o')
plt.plot(monthly_avg['Month'], monthly_avg['Predicted_XGB'],
         label='Predicted Benchmark (XGBoost)', marker='x')
plt.xlabel("Month")
plt.ylabel("Benchmark")
plt.title("2024 Monthly Benchmark: True vs Predicted (XGBoost)")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 构建对比 DataFrame（按月）
comparison_df = monthly_avg.rename(columns={
    'Month': 'Date',
    'Next_Month_Benchmark': 'True_Benchmark',
    'Predicted_XGB': 'Predicted_Benchmark'
})

# 打印前几行
print(comparison_df.head())
