# 🏠 Real Estate Property Investment Recommendation System  
**CMPT 733 2025 Spring Final Project – Group Gongxifacai**  
Huanyu Zhou, Lingjie Li, Xia Meng, Enze Jiang

---

## 📌 Overview
Greater Vancouver's real estate market is complex and expensive. This project uses **machine learning**, **geospatial analysis**, and **economic indicators** to help:
- 🏡 **Homebuyers & Sellers**: Understand price trends for better timing  
- 💼 **Investors**: Discover under/overvalued properties  
- 🏙️ **Urban Planners**: Explore effects of unemployment, income, and development  
- 📊 **Researchers**: Reuse our data pipeline for similar tasks

We combine historical trends, real-time listings, and predictive modeling — all presented via a **dynamic Tableau dashboard**.

🔗 **[Live Tableau Dashboard](https://public.tableau.com/views/CMPT733VancouverRealEstate/Dashboard1)**  
🔗 **[GitHub Repo Link](https://github.com/Ericzhouhy/CMPT_733_Final_Project)**

---

## 💡 Key Features
- 📈 **Trend Analysis** of housing prices (2005–2025)
- 🧠 **Price Prediction** with ML models (XGBoost & Random Forest)
- 🏷️ **Valuation Tags**: Underpriced / Fairly Priced / Overpriced
- 🗺️ **Interactive Dashboard**: Filter by neighborhood, price, type
- 🧾 **Economic Insight Integration**: Income, unemployment, CPI, permits

---

## 🧪 Data Sources
| Source | Description |
|--------|-------------|
| MLS® HPI | Long-term benchmark housing prices |
| BC Stats | Monthly economic indicators |
| Zolo.ca | Real-time property listings |
| GeoJSON | Neighborhood boundary shapefiles |

---

## ⚙️ Tools and Technologies
- **Scraping & Processing**: `Selenium`, `BeautifulSoup`, `Pandas`, `NumPy`
- **Geospatial**: `GeoPandas`, `Shapely`, Google Maps API
- **ML Modeling**: `scikit-learn`, `XGBoost`, `RandomizedSearchCV`
- **Visualization**: `Tableau`, `Matplotlib`, `Seaborn`

---

## 🔮 Machine Learning Pipeline
1. **Data Collection**: Scraped Zolo listings, HPI data, BC Stats
2. **Preprocessing**: Standardization, cleaning, imputation
3. **Geo-mapping**: Spatial joins to tag listings with neighborhood
4. **Modeling**:  
   - `RandomForest` & `XGBoost`  
   - `TimeSeriesSplit` for time-aware validation  
   - Evaluation with RMSE, MAE, R²
5. **Valuation**:  
   - Compare listing price to model-predicted price  
   - Tag as Underpriced / Fairly Priced / Overpriced  
6. **Visualization**: Live dashboard in Tableau

---

## 🧪 Model Performance
| Model | RMSE | R² Score |
|-------|------|----------|
| XGBoost | ~$26,000 | > 0.90 |

---

## 🖥️ Dashboard Highlights
👉 **[Explore the Dashboard](https://public.tableau.com/views/CMPT733VancouverRealEstate/Dashboard1?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)**  
Features include:
- 🗺️ **Interactive Map** with color-coded valuation
- 🔍 **Filters** for price, bedrooms, type, and neighborhood
- 📊 **Live Stats Panel** showing averages, medians, percentiles
- 🏘️ **Tooltip Popups** with predicted vs listed prices

---

## 🔧 How to Run the Code

This project includes scripts for scraping, preprocessing, modeling, and exporting data for Tableau.

### 📁 Main Files

| File | Purpose |
|------|---------|
| `house_price_cleaner.py` | Clean house price(HPI) dataset |
| `house_price.py` | train model to predict next month housing benchmark |
| `house_price_cleaner.py` | Clean house price(HPI) dataset |
| `cleand_data_visulization.ipynb` | Cleaned data visulization |
| `property_listings_map.html` | Property list visulization|


## ▶️ Run Instructions

```bash
# Step 1: Data scraping


# Step 2: Data cleaning and geospatial mapping
python3 house_price_cleaner.py
Run cleand_data_visulization.ipynb

# Step 3: Model training & prediction
python3 house_price.py

# Step 4: Generate Tableau-ready output
```


## ▶️ Run Instructions

```bash
# Step 1: Data scraping


# Step 2: Data cleaning and geospatial mapping
python3 house_price_cleaner.py
Run cleand_data_visulization.ipynb

# Step 3: Model training & prediction


# Step 4: Generate Tableau-ready output
```

## 📚 Lessons Learned
- ⚙️ **Data Integration is hard** — especially across sources and resolutions
- 🕒 **Time-aware modeling** is essential to avoid leakage
- 🌐 **Geospatial alignment** is more complex than expected
- 📊 **Visual accessibility** matters — Tableau made insights usable for everyone
- 🔁 **Iterative development** improves outcomes at every stage

---

## 🖼️ Project Poster
![Project Poster](poster.png)

---

## 📦 References
- [CMHC Housing Data](https://www.cmhc-schl.gc.ca/professionals/housing-markets-data-and-research/housing-data)  
- [CREA National Stats](https://stats.crea.ca/en-CA/)  
- [Greater Vancouver MLS® HPI](https://www.gvrealtors.ca/market-watch/MLS-HPI-home-price-comparison.hpi.greater_vancouver.all.all.2021-12-1.html)  
- [Kaggle Vancouver Price Data](https://www.kaggle.com/datasets/jennyzzhu/vancouver-house-prices-for-past-20-years)

---

## 🙌 Acknowledgements
Thanks to the CMPT 733 teaching team for their support, and to all contributors in the Greater Vancouver real estate data ecosystem.


