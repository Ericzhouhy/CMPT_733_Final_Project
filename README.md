# 🏠 Real Estate Property Investment Recommendation System  
**CMPT 733 Final Project – Group Gongxifacai**  
Huanyu Zhou, Lingjie Li, Xia Meng, Enze Zhou  

## 📌 Project Overview
This project aims to assist **homebuyers, real estate investors, policymakers, and researchers** in navigating the complex Greater Vancouver housing market by integrating **machine learning**, **geospatial analytics**, and **economic indicators**. Our system analyzes historical price trends, predicts future benchmark housing prices, and identifies **overvalued/undervalued properties** using a real-time, interactive **Tableau dashboard**.

---

## 📈 Key Features
- 🔍 **Housing Price Trend Analysis** across Greater Vancouver (2005–2025)
- 🧠 **Price Prediction Models** using Random Forest and XGBoost
- 🏷️ **Property Valuation Tags** (Underpriced, Fairly Priced, Overpriced)
- 🗺️ **Interactive Map Dashboard** with real-time filtering and exploration
- 📊 **Economic Context Integration**: Unemployment, income, CPI, building permits

---

## 🧪 Data Sources
| Source | Description |
|--------|-------------|
| MLS® Home Price Index (HPI) | Historical benchmark prices (2005–2025) |
| BC Stats | Monthly economic indicators: income, unemployment, CPI |
| Zolo.ca | Real-time property listings for Greater Vancouver |
| Neighborhood Shapefiles | For geospatial mapping of property data |

---

## 🧰 Tech Stack
- **Data Collection**: `Requests`, `BeautifulSoup`, `Selenium`
- **Data Processing**: `Pandas`, `NumPy`
- **Geospatial Analysis**: `GeoPandas`, `Shapely`, `Google Maps API`
- **Machine Learning**: `scikit-learn`, `XGBoost`, `StandardScaler`, `TimeSeriesSplit`
- **Visualization**: `Tableau`, `Matplotlib`, `Seaborn`

---

## 🧠 Methodology Summary
1. **Data Acquisition**: Scraped historical and real-time data from HPI, Zolo, and BC Stats.
2. **Preprocessing**: Cleaned, merged, and standardized data across formats and timestamps.
3. **Geocoding**: Translated addresses to lat/lon using Google Maps API and joined with neighborhood shapefiles.
4. **Modeling**: Applied RandomForest and XGBoost with time-aware validation (TimeSeriesSplit).
5. **Valuation Assessment**: Compared listing prices to predicted benchmarks to assign valuation tags.
6. **Visualization**: Built a fully interactive Tableau dashboard to deliver insights dynamically.

---

## 📊 Model Performance
| Model | RMSE | R² Score | MAE |
|-------|------|----------|-----|
| XGBoost | ~$26,000 | > 0.90 | - |

*Model trained on cleaned, time-aware features including economic indicators and historical price trends.*

---

## 🖥️ How to Use the Dashboard
Visit the Tableau dashboard (link shared internally) and:
- Filter properties by **neighborhood**, **price range**, or **number of bedrooms**
- Hover over listings to see price, size, and predicted benchmark
- Explore **interactive maps**, **valuation tags**, and **summary statistics**

---

## 📚 Lessons Learned
- Integrating multi-source, time-series, and spatial data is complex but rewarding.
- Time-aware modeling (e.g., `TimeSeriesSplit`) is crucial for avoiding data leakage.
- Geospatial joins require precision to avoid misclassification of properties.
- Interactive dashboards significantly improve accessibility for non-technical users.

---

## 📎 References
- [CMHC Housing Data](https://www.cmhc-schl.gc.ca/professionals/housing-markets-data-and-research/housing-data)  
- [CREA National Statistics](https://stats.crea.ca/en-CA/)  
- [Greater Vancouver REALTORS® – MLS® HPI](https://www.gvrealtors.ca/market-watch/MLS-HPI-home-price-comparison.hpi.greater_vancouver.all.all.2021-12-1.html)  
- [Vancouver House Prices – Kaggle](https://www.kaggle.com/datasets/jennyzzhu/vancouver-house-prices-for-past-20-years)  

---

## 🙌 Acknowledgments
Special thanks to **CMPT 733 instructors and TAs** for their guidance throughout this project. We hope this system contributes to better housing decisions in one of Canada’s most complex real estate markets.


https://public.tableau.com/views/CMPT733VancouverRealEstate/Dashboard1?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link
