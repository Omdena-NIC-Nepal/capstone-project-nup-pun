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
    st.subheader("Rice Price Prediction with Random Forest")

    # Scaling and preprocessing
    numeric_cols = merged_df.select_dtypes(include='number').columns.drop(['Rice(coarse)_price', 'Wheat_flour_price'])
    scaler = StandardScaler()
    merged_df[numeric_cols] = scaler.fit_transform(merged_df[numeric_cols])

    merged_df.dropna(inplace=True)
    merged_df = pd.get_dummies(merged_df, columns=['DISTRICT'], drop_first=True)
    rice_df = merged_df.dropna(subset=['Rice(coarse)_price'])

    # Feature-target split
    X_rice = rice_df.drop(columns=['Rice(coarse)_price', 'Wheat_flour_price', 'date'])
    y_rice = rice_df['Rice(coarse)_price']

    # Train-test split
    X_train_rice, X_test_rice, y_train_rice, y_test_rice = train_test_split(
        X_rice, y_rice, test_size=0.2, random_state=42
    )

    # Model training
    rf_rice = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_rice.fit(X_train_rice, y_train_rice)

    # Prediction
    y_pred_rice = rf_rice.predict(X_test_rice)

    # Evaluation metrics
    st.write("**Evaluation Metrics:**")
    st.write(f"MAE: {mean_absolute_error(y_test_rice, y_pred_rice):.2f}")
    st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test_rice, y_pred_rice)):.2f}")
    st.write(f"R² Score: {r2_score(y_test_rice, y_pred_rice):.2f}")

    # Plot Actual vs Predicted
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.scatter(y_test_rice, y_pred_rice, alpha=0.7, color='teal')
    ax1.plot([y_test_rice.min(), y_test_rice.max()],
             [y_test_rice.min(), y_test_rice.max()],
             'r--', lw=2)
    ax1.set_xlabel('Actual Rice Price')
    ax1.set_ylabel('Predicted Rice Price')
    ax1.set_title('Actual vs. Predicted Rice Price')
    ax1.grid(True)
    st.pyplot(fig1)

    # Feature importances
    importances = rf_rice.feature_importances_
    features = X_train_rice.columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Top 15 features
    st.write("**Top 15 Features Influencing Rice Price:**")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(15), palette='viridis', ax=ax2)
    ax2.set_title('Top 15 Feature Importances for Rice Price Prediction')
    plt.tight_layout()
    st.pyplot(fig2)
