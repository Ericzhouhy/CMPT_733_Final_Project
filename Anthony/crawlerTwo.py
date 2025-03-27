import PyPDF2
import re
import requests
import csv

api_key = ""

# List of valid cities (only these addresses will trigger geocoding and be output)
valid_cities = ["Vancouver", "Burnaby", "Richmond",
                "Surrey", "Coquitlam", "West Vancouver", "North Vancouver", "Delta"]


def extract_city(address):
    address_part = address.split('|')[0].strip()  # 取 '|' 前面的部分
    parts = [p.strip() for p in address_part.split(',')]
    if len(parts) > 1:
        return parts[-1]  # 返回最后一部分，即城市名
    return ''


def is_valid_city(address):
    return extract_city(address) in valid_cities


def get_lat_long_google(address):
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        'address': address,
        'key': api_key,
        'components': 'country:CA'
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['status'] == 'OK':
            lat = data['results'][0]['geometry']['location']['lat']
            lng = data['results'][0]['geometry']['location']['lng']
            return lat, lng
        else:
            print("Error:", data['status'])
            return None, None
    else:
        print("Request failed with status code:", response.status_code)
        return None, None


def get_existing_addresses(csv_file):
    existing_addresses = set()
    try:
        with open(csv_file, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                existing_addresses.add(row['Address'].strip())
    except FileNotFoundError:
        print(f"{csv_file} not found. No existing data to check against.")
    return existing_addresses


def save_to_csv(property_list, csv_file):
    existing_addresses = get_existing_addresses(csv_file)
    with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=[
            'Address', 'Price', 'Price Type', 'Latitude', 'Longitude', 'Beds', 'Baths', 'Date'
        ])
        if file.tell() == 0:
            writer.writeheader()
        for property_data in property_list:
            address = property_data['Address'].strip()
            if address in existing_addresses:
                continue
            # Geocode for valid addresses (this is optional if already done during extraction)
            lat, lng = get_lat_long_google(address)
            property_data['Latitude'] = lat
            property_data['Longitude'] = lng

            writer.writerow(property_data)
            existing_addresses.add(address)
    print(f"New data has been written to {csv_file}")


# Lists to store extracted information
addresses = []
prices = []
price_types = []
beds = []    # Store number of beds
baths = []   # Store number of baths
dates = []
latitudes = []
longitudes = []

property_list = []
start_page = 0
csv_file = '/Users/anthony/Desktop/cmpt733 final project/cleaned_data/filteredPropertyInVancouver.csv'

# Open the PDF file and extract data from pages
with open('/Users/anthony/Desktop/cmpt733 final project/Raw_data/data15.pdf', 'rb') as file:
    reader = PyPDF2.PdfReader(file)
    # Loop through pages (adjust the range as needed)
    for page_num in range(start_page, 1505):
        # Clear previous page data
        addresses.clear()
        prices.clear()
        price_types.clear()
        latitudes.clear()
        longitudes.clear()
        beds.clear()
        baths.clear()
        dates.clear()

        page = reader.pages[page_num]
        text = page.extract_text()

        # Extract titles, descriptions, and publication dates using regex
        titles = re.findall(r'<title>(.*?)</title>', text, re.DOTALL)
        descriptions = re.findall(
            r'<description\s*>\s*(.*?)\s*</description\s*>', text, re.DOTALL)
        pubDate = re.findall(r'<pubDate\s*>\s*(.*?)\s*</pubDate\s*>', text)

        # Process titles (skip the first entry)
        for title in titles[1:]:
            addr = re.findall(r'^(.*?)\s*\|', title)
            price_type_match = re.findall(r'\b(For Sale|For Rent)\b', title)
            price = re.findall(r'@ \$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', title)
            if addr:
                addresses.append(addr[0].strip())
            if price:
                prices.append(price[0].strip())
            if price_type_match:
                price_types.append(price_type_match[0])

        # Get latitude and longitude only for addresses with valid cities
        for address in addresses:
            if is_valid_city(address):
                lat, lng = get_lat_long_google(address)
                latitudes.append(lat)
                longitudes.append(lng)

        # Extract bed and bath information from descriptions (skip the first description)
        for description in descriptions[1:]:
            bed_bath = re.findall(
                r'(\d+)\s*bed\s*-\s*(\d+)\s*bath', description)
            if bed_bath:
                beds.append(bed_bath[0][0])
                baths.append(bed_bath[0][1])
            else:
                beds.append(None)
                baths.append(None)

        # Extract publication dates
        for pub_date in pubDate:
            match = re.findall(r'(\d{2})\s([A-Za-z]{3})\s(\d{2})', pub_date)
            if match:
                day, month, year = match[0]
                dates.append(f"{day} {month} {year}")
            else:
                dates.append(None)

        # Combine data into property_list (only for valid addresses)
        for address, price, price_type, lat, lng, bed, bath, date in zip(
                addresses, prices, price_types, latitudes, longitudes, beds, baths, dates):
            if not is_valid_city(address):
                continue  # Skip properties not in valid cities
            property_data = {
                'Address': address,
                'Price': price,
                'Price Type': price_type,
                'Latitude': lat,
                'Longitude': lng,
                'Beds': bed,
                'Baths': bath,
                'Date': date
            }
            property_list.append(property_data)

# --- Additional Filtering Step ---
# Filter the property_list using a condition similar to a DataFrame filter:
filtered_property_list = [prop for prop in property_list
                          if extract_city(prop['Address']) in valid_cities]

# Print all filtered properties
# for prop in filtered_property_list:
#     print(prop)

# Save only the filtered properties to CSV
save_to_csv(filtered_property_list, csv_file)
