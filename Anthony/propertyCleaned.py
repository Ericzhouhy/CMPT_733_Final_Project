import pandas as pd

cities = ["Vancouver", "Burnaby", "Richmond",
          "Surrey", "Coquitlam", "West Vancouver", "North Vancouver"]

df = pd.read_csv("../cleaned_data/propertyInVancouver.csv")


def extract_city(address):
    parts = [p.strip() for p in address.split(',')]
    return parts[-1] if parts else ""


# Filter rows by checking if the last comma-separated part of 'Address' is in the city list
filtered_df = df[df["Address"].apply(
    lambda addr: extract_city(addr) in cities)]

filtered_df.to_csv(
    "../cleaned_data/filteredPropertyInVancouver.csv", index=False)
