import pandas as pd

df = pd.read_csv(
    "../Raw_data/housing_cpi.csv",
    skiprows=9,        # 跳过前 9 行
    header=0           # 使用当前行（第10行）作为列名
)

df.rename(columns={
    "Products and product groups 3 4": "Category"
}, inplace=True)


df_long = df.melt(
    id_vars="Category",
    var_name="Date",
    value_name="CPI_Value"
)

df_long["Date"] = pd.to_datetime(
    df_long["Date"], format="%B %Y").dt.strftime("%Y-%m")


categories_of_interest = [
    "Shelter 6",
    "All-items",
    "All-items excluding food and energy 7"
]
df_cpi_filtered = df_long[df_long["Category"].isin(
    categories_of_interest)].copy()
df_cpi_filtered["Category"] = df_cpi_filtered["Category"].str.replace(
    r"\s*\d+$",  # 匹配末尾空格及数字
    "",          # 替换为空字符串
    regex=True
)


df_cpi_filtered.to_csv("../cleaned_data/new_cpi_data.csv", index=False, encoding="utf-8-sig")
