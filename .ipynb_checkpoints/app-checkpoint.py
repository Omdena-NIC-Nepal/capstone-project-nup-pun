import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from rice_model import run_rice_price_prediction
from wheat_flour_model import run_wheat_flour_price_prediction

# Title and Intro
st.title("Climate Variability and Food Price Dynamics in Nepal")
st.markdown("""
This app presents an analysis of how climate factors influence the prices of rice and wheat flour across districts of Nepal.
The analysis uses historical climate and crop price data along with machine learning modeling.
""")

# Load Data
@st.cache_data
def load_data():
    merged_df = pd.read_csv("merged_climate_food.csv")  
    gdf = gpd.read_file("data/districts/districts.shp")   
    return merged_df, gdf

merged_df, gdf = load_data()

# Data Overview
st.subheader("Data Overview")
st.dataframe(merged_df.head())
st.download_button("Download Sample Data", data=merged_df.to_csv(index=False), file_name="data_sample.csv")

# EDA Section
st.subheader("Exploratory Data Analysis")

# Price trends over time
col1, col2 = st.columns(2)
if 'year' not in merged_df.columns and 'date' in merged_df.columns:
    merged_df['year'] = pd.to_datetime(merged_df['date']).dt.year

with col1:
    st.markdown("### Rice Price Over Time")
    fig_rice, ax_rice = plt.subplots()
    sns.lineplot(data=merged_df, x='year', y='Rice(coarse)_price', ci=None, ax=ax_rice)
    st.pyplot(fig_rice)

with col2:
    st.markdown("### Wheat Flour Price Over Time")
    fig_flour, ax_flour = plt.subplots()
    sns.lineplot(data=merged_df, x='year', y='Wheat_flour_price', ci=None, ax=ax_flour)
    st.pyplot(fig_flour)

# Ensure mapping columns are consistent
gdf['DISTRICT'] = gdf['DISTRICT'].str.upper()
merged_df['DISTRICT'] = merged_df['DISTRICT'].str.upper()
map_data = gdf.merge(merged_df, how='left', on='DISTRICT')

# Merge shapefile GeoDataFrame with crop price data
nepal_map = gdf.merge(merged_df, how='left', left_on='DISTRICT', right_on='DISTRICT')
nepal_map['rice_price_display'] = nepal_map['Rice(coarse)_price'].fillna(-1)

# Show summary statistics
st.subheader("Summary Statistics")
st.dataframe(merged_df.describe())

# Correlation Heatmap
st.subheader("Correlation Between Climate Variables and Crop Prices")

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(
    merged_df.corr(numeric_only=True),
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    ax=ax
)
ax.set_title("Correlation Heatmap")
st.pyplot(fig)

# Grouped bar plot of average rice and wheat flour prices by district
st.subheader("Average Crop Prices by District")

# Compute average prices per district
avg_prices = merged_df.groupby('DISTRICT')[['Rice(coarse)_price', 'Wheat_flour_price']].mean().sort_values(by='Rice(coarse)_price', ascending=False)

# Plot using matplotlib
fig, ax = plt.subplots(figsize=(12, 6))
avg_prices.plot(kind='bar', ax=ax)
ax.set_ylabel("Price (NPR)")
ax.set_title("Average Rice and Wheat Flour Prices by District")
plt.xticks(rotation=90)
st.pyplot(fig)

# Plot function for district-wise heatmap
def plot_price_map(data, column_name, title):
    fig, ax = plt.subplots(figsize=(12, 10))
    data.plot(column= column_name,
                cmap='YlGn',
                linewidth=0.8,
                edgecolor='0.8',
                legend=True,
                ax=ax,
                missing_kwds={
                    "color": "lightgrey",
                    "label": "No data"
                })
    plt.title(title)
    plt.axis('off')
    return fig

# Show Rice Price Map
st.subheader("District-wise Average Rice Price")
fig_rice_map = plot_price_map(nepal_map, 'Rice(coarse)_price', 'Rice Price per District')
st.pyplot(fig_rice_map)

# Show Wheat Flour Price Map
st.subheader("District-wise Average Wheat Flour Price")
fig_wheat_map = plot_price_map(nepal_map, 'Wheat_flour_price', 'Wheat Flour Price per District')
st.pyplot(fig_wheat_map)

# Rice prediction
st.header("Rice Price Prediction")
run_rice_price_prediction(merged_df)

# Wheat Flour prediction
st.header("Wheat Flour Price Prediction")
run_wheat_flour_price_prediction(merged_df)

# Conclusion
st.subheader("Conclusion")
st.markdown("""
- Climate variables such as year and pressure significantly influence rice and wheat flour prices.
- Price variability can be mapped and monitored across districts.
- This tool can be used for policy planning and forecasting.
""")

# Footer
st.markdown("---")
st.markdown("Created with ❤️ by Anup Pun")
