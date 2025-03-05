# 按都会区（CMA）整体人口排名(我们收集并清理前五大CMA的house的开工竣工情况)
import pandas as pd
import numpy as np
years = list(range(2006, 2024))

all_data = []
for year in years:
    # 构造当前年份对应的文件路径
    file_path = f'Raw_data/housing-completions-dwelling-type-{year}.xlsx'
    # 读取该年度的 Excel
    df = pd.read_excel(file_path, sheet_name='CSD', header=3)
    # 筛选目标城市
    target_cities = ['Toronto', 'Montréal',
                     'Vancouver', 'Ottawa', 'Calgary', 'Edmonton', 'Québec', 'Winnipeg', 'Hamilton', 'Kitchener']
    df = df[df['Centre'].isin(target_cities)]

    # 筛选 Census Subdivision 中包含 "Total" 的行
    df = df[df['Census Subdivision'].str.contains(
        'Total', case=False, na=False)]

    # 添加年份列 & 状态列
    df['Year'] = year
    df['Status'] = 'housing completion'

    # 将当前年份的数据追加到列表
    all_data.append(df)

final_df = pd.concat(all_data, ignore_index=True)

final_df.to_excel(
    'cleaned_data/filtered_housing_completions_2006_2023.xlsx', index=False)
print("已合并 2006-2023 年的数据，并成功导出！")
