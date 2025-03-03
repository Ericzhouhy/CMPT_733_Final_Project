from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd

# Set up Selenium WebDriver
driver = webdriver.Chrome()

# Base URL pattern
base_url = "https://www.gvrealtors.ca/market-watch/MLS-HPI-home-price-comparison.hpi.lower_mainland.all.all.{}.html"

# List to store all data
all_data = []

# Loop over each month from January 2024 to January 2025
for year in range(2005, 2026):
    for month in range(1, 13):
        if year == 2025 and month > 1:  # Stop after January 2025
            break

        # Format the month to ensure two digits
        month_str = f"{month:02d}"
        # Construct the URL for the specific month
        url = base_url.format(f"{year}-{month_str}-1")

        # Visit the URL
        driver.get(url)

        # Allow time for the page to load
        wait = WebDriverWait(driver, 20)

        try:
            # Extract data from the table
            table = wait.until(EC.presence_of_element_located((By.TAG_NAME, 'table')))
            rows = table.find_elements(By.TAG_NAME, 'tr')

            # Extract headers
            headers = [header.text for header in rows[0].find_elements(By.TAG_NAME, 'th')]

            # Extract data for each row
            for row in rows[1:]:
                cols = row.find_elements(By.TAG_NAME, 'td')
                cols = [ele.text.strip() for ele in cols]
                if cols:
                    all_data.append([f"{year}-{month_str}"] + cols)  # Include date in each row

            print(f"Data for {year}-{month_str} extracted.")

        except Exception as e:
            print(f"An error occurred for {year}-{month_str}: {e}")

# Close the browser
driver.quit()

# Convert data to a DataFrame and save to a CSV
if all_data:
    df = pd.DataFrame(all_data, columns=['Date'] + headers)
    df.to_csv('lower_mainland_property_data.csv', index=False)
    print("All data has been saved to lower_mainland_property_data.csv")
else:
    print("No data extracted.")
