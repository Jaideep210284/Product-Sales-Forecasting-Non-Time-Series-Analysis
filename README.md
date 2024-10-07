# Product-Sales-Forecasting-Non-Time-Series-Analysis

This repository contains a **Streamlit**-based web application that predicts product sales using **time series** and **non-time series** methods. The app provides a forecasting solution for the next 15 weeks and analyzes product sales data using **Prophet** for time series forecasting and **Decision Trees** for non-time series predictions.

## Features:

1. **Time Series Evaluation (Prophet)**:
   - Forecast the next 15 weeks of product sales using the Prophet library.
   - Visualize historical demand and future predictions.
   - Display error distribution for time series forecasting.

2. **Non-Time Series Analysis (Decision Tree)**:
   - Predict actual vs. predicted demand for products using Decision Tree Regressor.
   - Visualize **training and testing error distributions** for evaluation.

3. **User Interaction**:
   - Users can select a product from the top 10 selling products.
   - Users can input the number of weeks for forecasting.

---

## Demo Video

Watch the demo of the Product Sales Forecasting App below:

[![Watch the video](https://img.youtube.com/vi/VIDEO_ID/maxresdefault.jpg)](https://www.youtube.com/watch?v=VIDEO_ID)


---

## Installation

Follow these steps to set up the project on your local machine:

1. **Clone the repository**:
   ```bash
   [git clone https://github.com/<your-username>/product-sales-forecasting-app.git](https://github.com/Jaideep210284/Product-Sales-Forecasting-Non-Time-Series-Analysis.git)
   cd product-sales-forecasting-app
   ```

2. **Install required dependencies**:
   You need Python 3.x installed on your machine. To install the required Python libraries, run the following command:
   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` file contains all the necessary dependencies, including:
   - Streamlit
   - Prophet
   - Pandas
   - XGBoost
   - Scikit-learn
   - Seaborn
   - Matplotlib

3. **Set up the dataset**:
   Ensure that you have the following CSV files:
   - `CustomerDemographics.csv`
   - `ProductInfo.csv`
   - `Transactional_data_retail_01.csv`
   - `Transactional_data_retail_02.csv`

   Place these files in the project directory or specify their paths in the code.

---

## How to Run the App

After setting up the dependencies and datasets, you can run the app using Streamlit:

1. **Run the app**:
   In the project directory, run the following command:
   ```bash
   streamlit run app.py
   ```

2. **Access the app**:
   After running the command, you’ll be provided with a local URL (usually `http://localhost:8501` or similar). Open this URL in your web browser.

---

## Usage

Once the app is running, follow these steps:

1. **Select a Product**:
   - From the dropdown menu, select a product from the top 10 selling products.
   
2. **Choose the Forecasting Duration**:
   - Use the slider to choose how many weeks (up to 15 weeks) you want the forecast for.

3. **View Forecast and Error Distribution**:
   - The app will generate a forecast for the selected product using the **Prophet** time series model.
   - The forecast includes both historical data and predictions for the selected duration.
   - Error distribution for the time series predictions is also displayed.

4. **Non-Time Series Analysis**:
   - The app also provides predictions for the selected product using the **Decision Tree Regressor** for non-time series analysis.
   - You can view the actual vs. predicted results for the training and test sets, as well as training and testing error distributions.

---

## App Workflow

### 1. Time Series Forecasting (Prophet):
   - The app uses **Prophet** to predict future sales based on historical data.
   - Users can input the number of weeks for forecasting.
   - A graph is generated, showing historical data, forecasted data, and uncertainty intervals.

### 2. Non-Time Series Forecasting (Decision Tree):
   - For non-time series analysis, the app uses a **Decision Tree Regressor** to predict product demand.
   - The app plots **Actual vs. Predicted** results for both training and test sets.
   - Error distribution plots for both training and test predictions are displayed to analyze model performance.

---

## Project Structure

Here’s a breakdown of the project files and structure:

```
├── app.py                     # Main Streamlit app script
├── CustomerDemographics.csv    # Customer demographics dataset
├── ProductInfo.csv             # Product information dataset
├── Transactional_data_retail_01.csv # Retail transactional data (part 1)
├── Transactional_data_retail_02.csv # Retail transactional data (part 2)
├── README.md                   # This README file
├── requirements.txt            # Python dependencies file
```

---

## Future Improvements

1. Add more product options or allow users to upload custom datasets.
2. Integrate advanced models like **XGBoost** for more complex non-time series forecasting.
3. Improve user experience by allowing users to adjust model parameters directly in the app.


