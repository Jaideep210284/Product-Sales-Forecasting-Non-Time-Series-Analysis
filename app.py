import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load your data with the correct file paths
customer_demo_path = '/Users/jai/Desktop/Soothsayer/Usecase_DemadForecasting1/CustomerDemographics.csv'
product_info_path = '/Users/jai/Desktop/Soothsayer/Usecase_DemadForecasting1/ProductInfo.csv'
transaction_01_path = '/Users/jai/Desktop/Soothsayer/Usecase_DemadForecasting1/Transactional_data_retail_01.csv'
transaction_02_path = '/Users/jai/Desktop/Soothsayer/Usecase_DemadForecasting1/Transactional_data_retail_02.csv'

# Load the datasets
customer_demo_df = pd.read_csv(customer_demo_path)
product_info_df = pd.read_csv(product_info_path)
transaction_01_df = pd.read_csv(transaction_01_path)
transaction_02_df = pd.read_csv(transaction_02_path)

# Combine the two transactional datasets into one
df = pd.concat([transaction_01_df, transaction_02_df], ignore_index=True)

# Merge customer demographics to include Country information
df = df.merge(customer_demo_df[['Customer ID', 'Country']], on='Customer ID', how='left')

# Convert 'InvoiceDate' to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

# Define top products
top_products = ['85123A', '71053', '84406B', '84029G', '85048', '79323P', '79323W', '22041', '21232', '21756']

# Streamlit App Title
st.title("Product Sales Forecasting & Non-Time Series Analysis")

# Select a product from top 10
product_code = st.selectbox("Select a Product", top_products)

# Time Series Analysis with Prophet
st.subheader(f"Time Series Evaluation: Forecast the next 15 weeks for Product {product_code}")

# Prepare data for Prophet for the selected product
weekly_sales_df = df.set_index('InvoiceDate').groupby(['StockCode', pd.Grouper(freq='W')])['Quantity'].sum().reset_index()

# Filter data for the selected product
product_sales = weekly_sales_df[weekly_sales_df['StockCode'] == product_code]
product_sales = product_sales.rename(columns={'InvoiceDate': 'ds', 'Quantity': 'y'})

# Initialize and fit Prophet model
model = Prophet()
model.fit(product_sales)

# Forecast the next 15 weeks
future = model.make_future_dataframe(periods=15, freq='W')
forecast = model.predict(future)

# Plot historical and forecasted data
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(product_sales['ds'], product_sales['y'], 'ko', label='Historical Demand')
ax.plot(forecast['ds'], forecast['yhat'], 'b-', label='Forecasted Demand')
ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='blue', alpha=0.2, label='Uncertainty Interval')
ax.set_xlabel("Date")
ax.set_ylabel("Quantity Sold")
ax.legend()
st.pyplot(fig)

# Non-Time Series Analysis with Decision Tree
st.subheader(f"Non-Time Series Analysis: Decision Tree for Product {product_code}")

# Prepare features and target
features = ['Price', 'Country']  # Replace with relevant features
X = df[df['StockCode'] == product_code][features]
y = df[df['StockCode'] == product_code]['Quantity']

# One-hot encode the 'Country' feature to make it numeric
onehot_encoder = OneHotEncoder(drop='first', sparse=False)
X_encoded = onehot_encoder.fit_transform(X[['Country']])
X_encoded_df = pd.DataFrame(X_encoded, columns=onehot_encoder.get_feature_names(['Country']))

# Combine with numerical features like 'Price'
X_numerical = X[['Price']].reset_index(drop=True)
X_combined = pd.concat([X_numerical, X_encoded_df], axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Initialize and fit Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# Predictions using Decision Tree
y_pred_train_dt = dt_model.predict(X_train)
y_pred_test_dt = dt_model.predict(X_test)

# Calculate RMSE and MAE for Decision Tree
rmse_dt_train = mean_squared_error(y_train, y_pred_train_dt, squared=False)
rmse_dt_test = mean_squared_error(y_test, y_pred_test_dt, squared=False)
mae_dt_train = mean_absolute_error(y_train, y_pred_train_dt)
mae_dt_test = mean_absolute_error(y_test, y_pred_test_dt)

st.write(f"Decision Tree - Training RMSE: {rmse_dt_train}, Test RMSE: {rmse_dt_test}")
st.write(f"Decision Tree - Training MAE: {mae_dt_train}, Test MAE: {mae_dt_test}")

# Plot Actual vs Predicted for Decision Tree
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(range(len(y_train)), y_train, color='blue', label='Train Actual Demand', alpha=0.5)
ax.scatter(range(len(y_train)), y_pred_train_dt, color='red', label='Train Predicted Demand (DT)', alpha=0.5)
ax.scatter(range(len(y_test)), y_test, color='green', label='Test Actual Demand', alpha=0.5)
ax.scatter(range(len(y_test)), y_pred_test_dt, color='orange', label='Test Predicted Demand (DT)', alpha=0.5)
ax.set_xlabel("Index")
ax.set_ylabel("Quantity Sold")
ax.legend()
st.pyplot(fig)

# Error distribution plots for Decision Tree
st.subheader("Training and Testing Error Distributions (Decision Tree)")

# Training error distribution
fig, ax = plt.subplots(figsize=(10, 6))
train_errors_dt = y_train - y_pred_train_dt
sns.histplot(train_errors_dt, kde=True, color='blue')
ax.set_title(f'Training Error Distribution (DT) - {product_code}')
st.pyplot(fig)

# Testing error distribution
fig, ax = plt.subplots(figsize=(10, 6))
test_errors_dt = y_test - y_pred_test_dt
sns.histplot(test_errors_dt, kde=True, color='orange')
ax.set_title(f'Testing Error Distribution (DT) - {product_code}')
st.pyplot(fig)
