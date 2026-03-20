# 📈💻 Stock Price Prediction using LSTM & Streamlit

---

## 🧠 Project Overview
- This project leverages Deep Learning (LSTM) to predict future stock prices based on historical data. It features a real-time web dashboard built with Streamlit that allows users to input any stock ticker and visualize trends alongside 100-day and 200-day Moving Averages.
- This tool aims to provide a data-driven perspective on stock momentum and potential price direction.

---

## ❓ Analytical Questions
- How do 100-day and 200-day Moving Averages signal price trends?
- Can a Long Short-Term Memory (LSTM) network accurately capture the volatility of the stock market?
- How does the model's prediction compare to the actual closing prices over the last 100 days?

---

## 📁 Dataset & Features
- Data Source: Real-time data fetched via yfinance (Yahoo Finance).
- Timeframe: 14 years of historical data.
- Features Used: - Close Price (Primary target)
- MA100 (100-day Moving Average)
- MA200 (200-day Moving Average)

---

## 📊 Dashboard Features
- Real-time Data Fetching: Users can enter any ticker (e.g., AAPL, TSLA, GOOGL).
- Interactive Visualizations: High-quality charts comparing Closing Prices with Moving Averages.
- Live Prediction: Generates a forecast plot comparing "Original" vs "Predicted" prices using the pre-trained LSTM model.

---

## 🖼️ Dashboard Preview
(Replace these with your actual screenshots after uploading)

---

## 🔍 Key Insights
- Momentum Indicators: The crossover between the 100-day and 200-day MA provides a visual "Golden Cross" or "Death Cross" for investors.
- LSTM Performance: The model effectively tracks the general trend and direction of the stock, though it acknowledges the inherent "lag" and noise in financial markets.

---

## 🤖 Machine Learning Model
- Architecture: Sequential LSTM (Long Short-Term Memory).
- Layers: Multiple LSTM layers with Dropout to prevent overfitting.
- Optimizer: Adam
- Loss Function: Mean Squared Error (MSE)

---

## 🛠️ Tools & Technologies Used
- Python (Core Logic)
- TensorFlow / Keras (Deep Learning Model)
- Streamlit (Web Dashboard)
- YFinance (Financial Data API)
- Matplotlib & Pandas (Data Processing & Viz)

## 🚀 How to Run the Project
1. Clone the repository:

git clone https://github.com/sanchit-2511/Stock-Price-Prediction-LSTM.git

2. Install the required dependencies:

pip install -r requirements.txt

3. Launch the app:

streamlit run App/app.py

---

## 🙌 Author

Sanchit G. Barne

---
