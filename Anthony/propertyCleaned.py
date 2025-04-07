import pandas as pd

# cities = ["Vancouver", "Burnaby", "Richmond",
#           "Surrey", "Coquitlam", "West Vancouver", "North Vancouver"]

# df = pd.read_csv("../cleaned_data/propertyInVancouver.csv")


# def extract_city(address):
#     parts = [p.strip() for p in address.split(',')]
#     return parts[-1] if parts else ""


# # Filter rows by checking if the last comma-separated part of 'Address' is in the city list
# filtered_df = df[df["Address"].apply(
#     lambda addr: extract_city(addr) in cities)]

# filtered_df.to_csv(
#     "../cleaned_data/filteredPropertyInVancouver.csv", index=False)

# import pandas as pd

# df_raw = pd.read_csv(
#     "../Raw_data/unemp.csv",
#     skiprows=11,    
#     header=0       
# )

# df_long = df_raw.melt(
#     var_name="Date",       # 后续列（月份）转换为新的 "Date" 列
#     value_name="Value"     # 各月份对应的数值存入 "Value" 列
# )

# df_long.rename(columns={
#     "Statistics Estimate": "Data type"
# }, inplace=True)

# print(df_long.head())


# import pandas as pd

# df = pd.read_csv("../Raw_data/unemp.csv", header=0)

# df.columns = df.columns.str.strip()

# print("原始数据：")
# print(df.head())

# df_melted = df.melt(
#     id_vars=["Labour force characteristics"],  # 保留该列不变
#     var_name="Date",                           # 将原来的列名(如 "Mar-15")变成新的列 "Date"
#     # 将对应的数值(如 5.9)放到 "Unemployment_rate"
#     value_name="Unemployment_rate"
# )

# df_melted["Date"] = pd.to_datetime(
#     df_melted["Date"], format="%b-%y", errors="coerce").dt.strftime("%Y-%m")

# # 如果不需要 "Labour force characteristics" 列了，可以删除
# df_melted.drop(columns=["Labour force characteristics"], inplace=True)

# print("转换后的数据：")
# print(df_melted)

# # 导出成新的 CSV
# df_melted.to_csv("../cleaned_data/cleaned_unemployment.csv", index=False, encoding="utf-8-sig")

# import pandas as pd

# df_raw = pd.read_csv("../Raw_data/average_income.csv", header=0)

# industry_col = df_raw.columns[0]

# # 筛选出目标行业行，注意匹配时确保字符串完全一致
# df_filtered = df_raw[df_raw[industry_col] ==
#                      "Industrial aggregate excluding unclassified businesses  [11-91N] 7 8"].copy()

# print("筛选后的数据：")
# print(df_filtered)

# # 删除表示行业名称的这一列，只保留月份列
# df_filtered.drop(columns=[industry_col], inplace=True)

# # 将宽表转换成长表
# df_long = df_filtered.melt(var_name="Date", value_name="AvgMonthlyEarnings")

# # 如果月份列是类似 "Mar-15" 的字符串，转换成 "YYYY-MM" 格式
# df_long["Date"] = pd.to_datetime(
#     df_long["Date"], format="%b-%y", errors="coerce").dt.strftime("%Y-%m")

# # 将平均收入乘以 4.33（假设原数据为平均周薪，此处转换成平均月收入）
# # 去除数值中的逗号并转换为 float，再乘以 4.33
# df_long["AvgMonthlyEarnings"] = (
#     df_long["AvgMonthlyEarnings"].str.replace(',', '')
#     .astype(float) * 4.33
# ).round(2)



# # 查看最终结果
# print("转换后的竖表数据：")
# print(df_long.head())

# # 导出成 CSV 文件
# df_long.to_csv("../cleaned_data/industrial_aggregate_timeseries.csv",
#                index=False, encoding="utf-8-sig")

import pandas as pd

# 定义月份缩写到数字的映射
month_map = {
    'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
    'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
    'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
}


def convert_month_format(s):
    # 假设 s 格式为 "Jan-03"
    parts = s.split('-')
    if len(parts) == 2:
        mon, yr = parts
        # 将年份 "03" 变为 "2003"
        year = "20" + yr if len(yr) == 2 else yr
        month = month_map.get(mon, mon)
        return f"{year}-{month}"
    else:
        return s


rbp_total = pd.read_csv('../Raw_data/resunitstotal.csv',
                        skiprows=1, thousands=',')
rbp_total['SGC Code'] = rbp_total['SGC Code'].astype(str).str.strip()

row = rbp_total[rbp_total['SGC Code'] == "5915"]


melted = row.melt(
    var_name='Month',
    value_name='totalNumber'
)
melted = melted[melted['Month'] != 'Unnamed: 1']
melted = melted[melted['Month'] != 'SGC Code']

# 应用函数到 'Month' 列
melted['Date'] = melted['Month'].apply(convert_month_format)
melted.to_csv(
    '../cleaned_data/Residential Building Permits totalNumber.csv', index=False)


# rbp_urows = pd.read_csv('../Raw_data/resunitsrow.csv', skiprows = 1)
# rbp_urows['SGC Code'] = rbp_urows['SGC Code'].astype(str).str.strip()

# row = rbp_urows[rbp_urows['SGC Code'] == "5915"]


# melted = row.melt(
#     var_name='Month',
#     value_name='row dwelling units'
# )
# melted = melted[melted['Month'] != 'Unnamed: 1']
# melted = melted[melted['Month'] != 'SGC Code']

# # 应用函数到 'Month' 列
# melted['Date'] = melted['Month'].apply(convert_month_format)

# melted.to_csv(
#     '../cleaned_data/Residential Building Permits row dwelling.csv', index=False)

# rbp_uapart = pd.read_csv(
#     '../Raw_data/resunitsapartments.csv', skiprows=1, thousands=',')

# rbp_uapart['SGC Code'] = rbp_uapart['SGC Code'].astype(str).str.strip()

# row = rbp_uapart[rbp_uapart['SGC Code'] == "5915"]


# melted = row.melt(
#     var_name='Month',
#     value_name='Number of apartment units'
# )
# melted = melted[melted['Month'] != 'Unnamed: 1']
# melted = melted[melted['Month'] != 'SGC Code']

# # # 应用函数到 'Month' 列
# melted['Date'] = melted['Month'].apply(convert_month_format)

# melted.to_csv(
#     '../cleaned_data/Residential Building Permits Number of apartment units.csv', index=False)

# rbp_usingles = pd.read_csv(
#     '../Raw_data/resunitssingles.csv', skiprows=1, thousands=',')

# rbp_usingles['SGC Code'] = rbp_usingles['SGC Code'].astype(str).str.strip()

# row = rbp_usingles[rbp_usingles['SGC Code'] == "5915"]


# melted = row.melt(
#     var_name='Month',
#     value_name='Number of single dwelling units'
# )
# melted = melted[melted['Month'] != 'Unnamed: 1']
# melted = melted[melted['Month'] != 'SGC Code']

# # # 应用函数到 'Month' 列
# melted['Date'] = melted['Month'].apply(convert_month_format)

# melted.to_csv(
#     '../cleaned_data/Residential Building Permits Number of single dwelling units.csv', index=False)


# rbp_govers = pd.read_csv('../Raw_data/instigov.csv', skiprows=1, thousands=',')
# rbp_govers['SGC Code'] = rbp_govers['SGC Code'].astype(str).str.strip()

# row = rbp_govers[rbp_govers['SGC Code'] == "5915"]


# melted = row.melt(
#     var_name='Month',
#     value_name='Number of Institutional and Government Building Permits'
# )
# melted = melted[melted['Month'] != 'Unnamed: 1']
# melted = melted[melted['Month'] != 'SGC Code']

# # # 应用函数到 'Month' 列
# melted['Date'] = melted['Month'].apply(convert_month_format)

# melted.to_csv(
#     '../cleaned_data/Residential Building Permits Institutional and Government Building Permits.csv', index=False)

# rbp_industrials = pd.read_csv(
#     '../Raw_data/industrial.csv', skiprows=1, thousands=',')
# rbp_industrials['SGC Code'] = rbp_industrials['SGC Code'].astype(
#     str).str.strip()

# row = rbp_industrials[rbp_industrials['SGC Code'] == "5915"]


# melted = row.melt(
#     var_name='Month',
#     value_name='Number of Industrial Building Permits'
# )
# melted = melted[melted['Month'] != 'Unnamed: 1']
# melted = melted[melted['Month'] != 'SGC Code']

# # # 应用函数到 'Month' 列
# melted['Date'] = melted['Month'].apply(convert_month_format)

# melted.to_csv(
#     '../cleaned_data/Residential Building Permits Industrial Building Permits.csv', index=False)



# rbp_commercials = pd.read_csv(
#     '../Raw_data/commercial.csv', skiprows=1, thousands=',')
# rbp_commercials['SGC Code'] = rbp_commercials['SGC Code'].astype(
#     str).str.strip()

# row = rbp_commercials[rbp_commercials['SGC Code'] == "5915"]


# melted = row.melt(
#     var_name='Month',
#     value_name='Number of Commercial Building Permits'
# )
# melted = melted[melted['Month'] != 'Unnamed: 1']
# melted = melted[melted['Month'] != 'SGC Code']

# # # 应用函数到 'Month' 列
# melted['Date'] = melted['Month'].apply(convert_month_format)

# melted.to_csv(
#     '../cleaned_data/Residential Building Permits Commercial Building Permits.csv', index=False)






