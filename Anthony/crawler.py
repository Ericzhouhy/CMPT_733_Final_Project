import PyPDF2
import re
import requests
import csv

# api_key = "AIzaSyCXWOG_sUAT5W9VE0lvH6jpSLMg7klaU9s"

api_key = "AIzaSyDUZ_NCrIIkur3xPFNfn-jyW_zn3WaIEmE"


def get_lat_long_google(address):
    # Google Geocoding API URL
    url = "https://maps.googleapis.com/maps/api/geocode/json"

    # 请求参数
    params = {
        'address': address,
        'key': api_key,  # 用你自己的 API 密钥
        'components': 'country:CA'  # 限定为加拿大的地址
    }

    # 发送 GET 请求
    response = requests.get(url, params=params)

    # 解析返回的 JSON 数据
    if response.status_code == 200:
        data = response.json()

        if data['status'] == 'OK':
            # 获取经纬度
            lat = data['results'][0]['geometry']['location']['lat']
            lng = data['results'][0]['geometry']['location']['lng']
            return lat, lng
        else:
            print("Error:", data['status'])
            return None, None
    else:
        print("Request failed with status code:", response.status_code)
        return None, None


addresses = []
prices = []
price_types = []
beds = []  # 存储床数量
baths = []  # 存储浴室数量
dates = []
latitudes = []
longitudes = []

# 存储所有的房产信息
property_list = []
start_page = 700
# 打开 PDF 文件
with open('/Users/anthony/Downloads/data.pdf', 'rb') as file:
    reader = PyPDF2.PdfReader(file)
    # max_page = min(20, len(reader.pages))
    # total_pages = len(reader.pages)

    # for page_num in range(max_page):
    for page_num in range(start_page, 702):
        addresses.clear()
        prices.clear()
        price_types.clear()
        latitudes.clear()
        longitudes.clear()
        beds.clear()
        baths.clear()
        dates.clear()
        # 获取第一页
        page = reader.pages[page_num]

        # 提取第一页的文本
        text = page.extract_text()

        # 使用正则表达式提取 title, description, pubDate
        titles = re.findall(r'<title>(.*?)</title>', text, re.DOTALL)
        descriptions = re.findall(
            r'<description\s*>\s*(.*?)\s*</description\s*>', text, re.DOTALL)
        pubDate = re.findall(r'<pubDate\s*>\s*(.*?)\s*</pubDate\s*>', text)

        # 从第二条数据开始处理
        for title in titles[1:]:  # 跳过第一个条目
            # 提取地址：从开头到 "For Sale @" 前的部分
            address = re.findall(r'^(.*?)\s*\|', title)

            price_type = re.findall(r'\b(For Sale|For Rent)\b', title)

            # 提取售价："$"后面的数字部分
            price = re.findall(r'@ \$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', title)

            if address:
                addresses.append(address[0].strip())  # 去掉前后的空格
            if price:
                prices.append(price[0].strip())  # 去掉前后的空格
            if price_type:
                price_types.append(price_type[0])

        # 获取经纬度
        for address in addresses:
            # 加入“Canada”来增强地址匹配
            lat, lng = get_lat_long_google(address)
            latitudes.append(lat)
            longitudes.append(lng)

        # 提取 bed 和 bath 信息，分别存储
        for description in descriptions[1:]:  # 跳过第一个描述
            bed_bath = re.findall(r'(\d+)\s*bed\s*-\s*(\d+)\s*bath', description)
            if bed_bath:
                beds.append(bed_bath[0][0])  # 存储床的数量
                baths.append(bed_bath[0][1])  # 存储浴室的数量
            else:
                beds.append(None)
                baths.append(None)

        # 提取 publication date 中的日期部分
        for pub_date in pubDate: 
            # 使用正则表达式提取年、月、日
            match = re.findall(r'(\d{2})\s([A-Za-z]{3})\s(\d{2})', pub_date)
            if match:
                # 年、月、日分别存入
                day, month, year = match[0]
                dates.append(f"{day} {month} {year}")
            else:
                dates.append(None)

        # 将所有提取的数据放入字典，并添加到列表中
        for address, price, price_type, lat, lng, bed, bath, date in zip(
                addresses, prices, price_types, latitudes, longitudes, beds, baths, dates):
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


# for property in property_list:
#     print(property)

def save_to_csv(property_list):
    # 指定 CSV 文件的路径
    output_file = '/Users/anthony/Downloads/propertyInVancouver.csv'

    # CSV 字段名
    fieldnames = ['Address', 'Price', 'Price Type',
                  'Latitude', 'Longitude', 'Beds', 'Baths', 'Date']

    # 写入 CSV 文件
    with open(output_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # 如果文件为空（即没有表头），则写入表头
        if file.tell() == 0:
            writer.writeheader()

        # 写入每一条数据
        for property_data in property_list:
            writer.writerow(property_data)

    print(f"Data has been written to {output_file}")


save_to_csv(property_list)

