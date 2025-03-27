import pandas as pd

# 假设文件名为 'rates.csv'
df = pd.read_csv('../Raw_data/10yearsinterestrate.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
# 确保 'Date' 列正确识别为日期类型
df_monthly_avg = df.resample('M').mean()
df_monthly_avg['Rate'] = df_monthly_avg['Rate'].round(2)

df_monthly_avg['YearMonth'] = df_monthly_avg.index.strftime('%Y-%m')
# 重置索引方便导出
df_monthly_avg.reset_index(drop=True, inplace=True)
df_monthly_avg.to_csv('../cleaned_data/monthly_interest_rates.csv', index=False)
