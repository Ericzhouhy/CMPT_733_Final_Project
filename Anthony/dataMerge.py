import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取房地产数据和利率数据（假设文件名分别为 real_estate.csv 和 interest_rate.csv）
real_estate_df = pd.read_csv('../cleaned_data/Cleaned_property_HPI_data.csv')
interest_df = pd.read_csv('../cleaned_data/monthly_interest_rates.csv')

# 如果房地产数据的日期列名为 Date，可以重命名为 YearMonth
real_estate_df.rename(columns={'Date': 'YearMonth'}, inplace=True)

# 处理 Benchmark 列：去除美元符号和逗号，然后转换为数值
real_estate_df['Benchmark'] = real_estate_df['Benchmark'].str.replace(
    '[$,]', '', regex=True).astype(float)

# 合并数据，根据 YearMonth 这个字段进行左连接（保留房地产数据的所有记录）
merged_df = pd.merge(real_estate_df, interest_df, on='YearMonth', how='left')

# # 查看合并后的数据
print(merged_df.head())

# desc_stats = merged_df.describe()
# print("描述性统计信息：\n", desc_stats)

# 单独计算 Price Index 和 Rate 的均值
# print("Price Index均值：", merged_df["Price Index"].mean())
# print("Rate均值：", merged_df["Rate"].mean())
# merged_df['YearMonth_dt'] = pd.to_datetime(
#     merged_df['YearMonth'], format='%Y-%m')

# # 按日期排序
# merged_df.sort_values('YearMonth_dt', inplace=True)
# fig, ax1 = plt.subplots(figsize=(10, 6))

# # 在 ax1 上绘制 Price Index
# color1 = 'tab:blue'
# ax1.set_xlabel('YearMonth')
# ax1.set_ylabel('Price Index', color=color1)
# ax1.plot(merged_df['YearMonth_dt'], merged_df['Price Index'],
#          color=color1, label='Price Index')
# ax1.tick_params(axis='y', labelcolor=color1)

# # 创建第二个 y 轴，与 ax1 共享 x 轴
# ax2 = ax1.twinx()
# color2 = 'tab:orange'
# ax2.set_ylabel('Rate', color=color2)
# ax2.plot(merged_df['YearMonth_dt'], merged_df['Rate'],
#          color=color2, label='Rate')
# ax2.tick_params(axis='y', labelcolor=color2)

# plt.title('Price Index vs. Rate (Two Y-axes)')
# fig.tight_layout()
# plt.show()

# 按 Residential Type 分组，计算 Price Index 和 Benchmark 的均值和标准差
# grouped_stats = merged_df.groupby('Residential Type').agg({
#     'Price Index': ['mean', 'std'],
#     'Benchmark': ['mean', 'std']
# })
# print("按住宅类型分组的统计信息：\n", grouped_stats)


# # 计算相关系数
# corr_price = merged_df['Rate'].corr(merged_df['Price Index'])
# corr_bench = merged_df['Rate'].corr(merged_df['Benchmark'])
# print("Rate 与 Price Index 的相关系数：", corr_price)
# print("Rate 与 Benchmark 的相关系数：", corr_bench)

# # 构建相关矩阵，并绘制热力图
# corr_matrix = merged_df[['Price Index', 'Benchmark', 'Rate']].corr()

# plt.figure(figsize=(6, 4))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
# plt.title('相关性矩阵')
# plt.tight_layout()
# plt.show()
