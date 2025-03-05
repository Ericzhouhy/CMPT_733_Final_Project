# Real Estate Market Health Analysis
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def to_thousands(x, pos):
    return f'{round(x/1000, 1)}K'
# 读取
df_start = pd.read_excel('cleaned_data/filtered_housing_start_2006_2023.xlsx')
print(df_start.columns)
df_completion = pd.read_excel(
    'cleaned_data/filtered_housing_completions_2006_2023.xlsx')

df_start['Total'] = df_start['Total'].replace(',', '', regex=True).astype(int)
df_completion['Total'] = df_completion['Total'].replace(
    ',', '', regex=True).astype(int)

# 按城市(Centre)和年份(Year)分组汇总“Total”
grouped_start = df_start.groupby(
    ['Centre', 'Year'], as_index=False)['Total'].sum()
grouped_start.rename(columns={'Total': 'Start_Total'}, inplace=True)

grouped_completion = df_completion.groupby(
    ['Centre', 'Year'], as_index=False)['Total'].sum()
grouped_completion.rename(columns={'Total': 'Completion_Total'}, inplace=True)

# 合并到一个DataFrame
df_city_year = pd.merge(grouped_start, grouped_completion, on=[
                        'Centre', 'Year'], how='outer')
# 如果某些城市-年份组合没有开工或没有竣工，则出现NaN，使用0填充
df_city_year.fillna(0, inplace=True)

plot_cities = ['Toronto', 'Montréal', 'Vancouver', 'Ottawa', 'Calgary']
plot_data = df_city_year[df_city_year['Centre'].isin(plot_cities)]

df_melted = plot_data.melt(
    id_vars=['Centre', 'Year'],
    value_vars=['Start_Total', 'Completion_Total'],
    var_name='Type',       # 会是 'Start_Total' 或 'Completion_Total'
    value_name='Units'
)
plt.figure(figsize=(10, 6))
# 使用 hue 区分不同城市，用 style 区分 Start/Completion
sns.lineplot(
    data=df_melted,
    x='Year',
    y='Units',
    hue='Centre',       # 不同城市用不同颜色
    style='Type',       # 不同类型(开工/竣工)用不同线型
    markers=True
)

plt.title('Housing Starts and Completions by City and Year')
plt.xlabel('Year')
plt.ylabel('Number of Units')

years = sorted(df_melted['Year'].unique())
plt.xticks(years, [str(y) for y in years], rotation=0)

ax = plt.gca()
# 限制刻度数量，这里设定最多显示6个刻度
ax.yaxis.set_major_locator(ticker.MaxNLocator(6))
# 使用自定义函数，将数值以千为单位显示
ax.yaxis.set_major_formatter(ticker.FuncFormatter(to_thousands))

plt.tight_layout()
plt.show()









# apartment starts completions D3-1/4
# 比较分析的几个方向：
# 分析2023house的
# 开工与竣工的对比：
# 对比每个城市的开工和竣工数据，可以分析哪些城市的建筑活动非常活跃，哪些城市可能出现竣工滞后的情况。
# 如果某个城市开工量较高，但竣工量较低，可能反映该城市的建筑项目尚未完成，可能存在供应链问题或市场吸纳问题。
# 按规模的对比：
# 特定城市的对比：
# 可以选择一些关键城市（如Vancouver、Toronto、Montreal等）进行开工和竣工数据的对比，帮助评估这些城市的房地产市场健康状况。
# 总计数据对比：
# 比较总的开工数量和竣工数量（在表格的底部有总计数据），可以查看全国范围内的建筑活动情况，了解总体的供应情况。
