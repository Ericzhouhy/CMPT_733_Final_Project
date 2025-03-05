import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据
df_start = pd.read_excel('cleaned_data/filtered_housing_start_2006_2023.xlsx')
df_completion = pd.read_excel(
    'cleaned_data/filtered_housing_completions_2006_2023.xlsx')

# 如果Excel中数字带有逗号，先去除逗号再转换为数值（如果需要）
cols = ['Singles', 'Semis', 'Row', 'Apartment\nand Other']
for col in cols:
    if df_start[col].dtype == object:
        df_start[col] = df_start[col].str.replace(',', '').astype(float)
    if df_completion[col].dtype == object:
        df_completion[col] = df_completion[col].str.replace(
            ',', '').astype(float)

# 将住宅类型列转换为长格式数据
df_start_long = df_start.melt(id_vars=['Centre', 'Year'],
                              value_vars=cols,
                              var_name='Residential_Type',
                              value_name='Start_Units')

df_completion_long = df_completion.melt(id_vars=['Centre', 'Year'],
                                        value_vars=cols,
                                        var_name='Residential_Type',
                                        value_name='Completion_Units')

# 合并开工和竣工数据
df_long = pd.merge(df_start_long, df_completion_long,
                   on=['Centre', 'Year', 'Residential_Type'],
                   how='outer')

# 如果只关注多伦多、蒙特利尔和温哥华，则进行过滤
cities = ['Toronto', 'Montréal', 'Vancouver']
df_filtered = df_long[df_long['Centre'].isin(cities)]

years = sorted(df_long['Year'].unique())

# 示例：绘制不同住宅类型的开工情况
plt.figure(figsize=(12, 8))
sns.lineplot(data=df_filtered,
             x='Year',
             y='Start_Units',
             hue='Residential_Type',
             style='Centre',
             markers=True)
plt.title('Housing Starts by Residential Type in Toronto, Montréal, and Vancouver')
plt.xticks(years, [str(y) for y in years], rotation=0)
plt.ylabel('Number of Units')
plt.tight_layout()
plt.show()

# 示例：绘制不同住宅类型的竣工情况
plt.figure(figsize=(12, 8))
sns.lineplot(data=df_filtered,
             x='Year',
             y='Completion_Units',
             hue='Residential_Type',
             style='Centre',
             markers=True)
plt.title(
    'Housing Completions by Residential Type in Toronto, Montréal, and Vancouver')
plt.xticks(years, [str(y) for y in years], rotation=0)
plt.ylabel('Number of Units')
plt.tight_layout()
plt.show()
