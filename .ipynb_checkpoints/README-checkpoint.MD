# 🌾 Nepal Crop Price & Climate Analysis App

This Streamlit web app analyzes the impact of climate variables on **rice** and **wheat flour** prices across different districts in Nepal. It includes data visualization, statistical insights, and machine learning-based price prediction models.

---

## 📊 Features

- 📈 Visualize rice and wheat flour prices over time.
- 🗺️ Static map of Nepal showing average crop prices by district.
- 🔥 Correlation heatmap between climate and crop price data.
- 🤖 ML model (Random Forest) to predict rice prices using top climate and spatial features.
- 📌 Feature importance visualization for price prediction.

---

## 📁 Project Structure

├── app.py # Main Streamlit app
├── rice_model.py # Model training and visualization functions for Rice 
├── wheat_flour_model # Model training and visualization functions for Wheat flour
├── merged_climate_food.csv # Cleaned Climate + price + geo-merged dataset
├── Data Pre-processing, and Cleaning
├── data/
│    ├── districts
│    │   ├── districts.shp # Nepal map with districts
│ ├── monthly_climate.csv # Processed monthly climate data
│ └── wfp_food_prices_npl.csv # Food prices of Rice and Wheat Flour
├── requirements.txt # Required Python libraries
└── README.md # You're here!