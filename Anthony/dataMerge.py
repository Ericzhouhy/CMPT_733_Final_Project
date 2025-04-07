import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取房地产数据和利率数据（假设文件名分别为 real_estate.csv 和 interest_rate.csv）
real_estate_df = pd.read_csv('../cleaned_data/Cleaned_property_HPI_data.csv')
interest_df = pd.read_csv('../cleaned_data/monthly_interest_rates.csv')
housing_cpi = pd.read_csv('../cleaned_data/new_cpi_data.csv')
average_income = pd.read_csv('../cleaned_data/industrial_aggregate_timeseries.csv')
unEmpRate = pd.read_csv('../cleaned_data/cleaned_unemployment.csv')

commercial_bp = pd.read_csv('../cleaned_data/Residential Building Permits Commercial Building Permits.csv')
industrial_bp = pd.read_csv('../cleaned_data/Residential Building Permits Industrial Building Permits.csv')
instAndGovers_bp = pd.read_csv('../cleaned_data/Residential Building Permits Institutional and Government Building Permits.csv')
singleUnits_bp = pd.read_csv('../cleaned_data/Residential Building Permits Number of single dwelling units.csv')
apartsUnits_bp = pd.read_csv('../cleaned_data/Residential Building Permits Number of apartment units.csv')
rowsDewll_bp = pd.read_csv('../cleaned_data/Residential Building Permits row dwelling.csv')
totalNumbers_bp = pd.read_csv('../cleaned_data/Residential Building Permits totalNumber.csv')

# 如果房地产数据的日期列名为 Date，可以重命名为 YearMonth
real_estate_df.rename(columns={'Date': 'YearMonth'}, inplace=True)

# 处理 Benchmark 列：去除美元符号和逗号，然后转换为数值
real_estate_df['Benchmark'] = real_estate_df['Benchmark'].str.replace(
    '[$,]', '', regex=True).astype(float)

# 合并数据，根据 YearMonth 这个字段进行左连接（保留房地产数据的所有记录）
merged_df = pd.merge(real_estate_df, interest_df, on='YearMonth', how='left')
merged_df.rename(columns={'YearMonth': 'Date'}, inplace=True)
# print(merged_df.columns.tolist())
# print(housing_cpi.columns.tolist())

df_merged = pd.merge(
    merged_df,    # 包含 Date, Benchmark, Price Index, 等
    housing_cpi,   # 包含 Date, Shelter, All-items, 等
    on="Date",      # 根据 Date 列进行合并
    how="inner"     # 只保留两边都有的日期
)

df_temp = pd.merge(
    df_merged,
    average_income,
    on="Date",      # 根据 Date 列进行合并
    how="inner"     # 只保留两边都有的日期
)

df_ump = pd.merge(
    df_temp,
    unEmpRate,
    on="Date",      # 根据 Date 列进行合并
    how="inner"     # 只保留两边都有的日期
)

df_commercial_bp = pd.merge(
    df_ump,
    commercial_bp,
    on="Date",      # 根据 Date 列进行合并
    how="inner"     # 只保留两边都有的日期
)
df_commercial_bp = df_commercial_bp.drop(columns=['Month'])


df_industrial_bp = pd.merge(
    df_commercial_bp,
    industrial_bp,
    on="Date",      # 根据 Date 列进行合并
    how="inner"     # 只保留两边都有的日期
)
df_industrial_bp = df_industrial_bp.drop(columns=['Month'])
df_instAndGovers_bp = pd.merge(
    df_industrial_bp,
    instAndGovers_bp,
    on="Date",      # 根据 Date 列进行合并
    how="inner"     # 只保留两边都有的日期
)
df_instAndGovers_bp = df_instAndGovers_bp.drop(columns=['Month'])
df_singleUnits_bp = pd.merge(
    df_instAndGovers_bp,
    singleUnits_bp,
    on="Date",      # 根据 Date 列进行合并
    how="inner"     # 只保留两边都有的日期
)
df_singleUnits_bp = df_singleUnits_bp.drop(columns=['Month'])
df_apartsUnits_bp = pd.merge(
    df_singleUnits_bp,
    apartsUnits_bp,
    on="Date",      # 根据 Date 列进行合并
    how="inner"     # 只保留两边都有的日期
)
df_apartsUnits_bp = df_apartsUnits_bp.drop(columns=['Month'])
df_rowsDewll_bp = pd.merge(
    df_apartsUnits_bp,
    rowsDewll_bp,
    on="Date",      # 根据 Date 列进行合并
    how="inner"     # 只保留两边都有的日期
)
df_rowsDewll_bp = df_rowsDewll_bp.drop(columns=['Month'])
df_totalNumbers_bp = pd.merge(
    df_rowsDewll_bp,
    totalNumbers_bp,
    on="Date",      # 根据 Date 列进行合并
    how="inner"     # 只保留两边都有的日期
)
df_totalNumbers_bp = df_totalNumbers_bp.drop(columns=['Month'])

# # 查看合并后的数据
df_totalNumbers_bp.to_csv("../cleaned_data/mergeData.csv",
               index=False, encoding="utf-8-sig")






