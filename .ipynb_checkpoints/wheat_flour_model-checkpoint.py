import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def run_rice_price_prediction(merged_df):
    st.subheader("Wheat Flour Price Prediction with Random Forest")

    # Scaling and preprocessing
    numeric_cols = merged_df.select_dtypes(include='number').columns.drop(['Rice(coarse)_price', 'Wheat_flour_price'])
    scaler = StandardScaler()
    merged_df[numeric_cols] = scaler.fit_transform(merged_df[numeric_cols])

    merged_df.dropna(inplace=True)
    merged_df = pd.get_dummies(merged_df, columns=['DISTRICT'], drop_first=True)
    wheat_df = merged_df.dropna(subset=['Wheat_flour_price'])

    # Feature-target split
    X_wheat = wheat_df.drop(columns=['Rice(coarse)_price', 'Wheat_flour_price', 'date'])
    y_wheat = wheat_df['Wheat_flour_price']

    # Train-test split
    X_train_wheat, X_test_wheat, y_train_wheat, y_test_wheat = train_test_split(
        X_wheat, y_wheat, test_size=0.2, random_state=42
    )

    # Model training
    rf_wheat = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_wheat.fit(X_train_wheat, y_train_wheat)

    # Prediction
    y_pred_wheat = rf_wheat.predict(X_test_wheat)

    # Evaluation metrics
    st.write("**Evaluation Metrics:**")
    st.write(f"MAE: {mean_absolute_error(y_test_wheat, y_pred_wheat):.2f}")
    st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test_wheat, y_pred_wheat)):.2f}")
    st.write(f"R² Score: {r2_score(y_test_wheat, y_pred_wheat):.2f}")

    # Plot Actual vs Predicted
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.scatter(y_test_wheat, y_pred_wheat, alpha=0.7, color='teal')
    ax1.plot([y_test_wheat.min(), y_test_wheat.max()],
             [y_test_wheat.min(), y_test_wheat.max()],
             'r--', lw=2)
    ax1.set_xlabel('Actual Wheat Flour Price')
    ax1.set_ylabel('Predicted Wheat Flour Price')
    ax1.set_title('Actual vs. Predicted Wheat Flour Price')
    ax1.grid(True)
    st.pyplot(fig1)

    # Feature importances
    importances = rf_wheat.feature_importances_
    features = X_train_wheat.columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Top 15 features
    st.write("**Top 15 Features Influencing Wheat Flour Price:**")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(15), palette='viridis', ax=ax2)
    ax2.set_title('Top 15 Feature Importances for Wheat Flour Price Prediction')
    plt.tight_layout()
    st.pyplot(fig2)
