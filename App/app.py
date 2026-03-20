import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.models import load_model
import streamlit as st 

st.title('Stock Price Prediction')
ticker = st.text_input('Enter Stock Ticker', 'AAPL')
data = yf.download(ticker, period='14y',interval='1d')
print(data.head())


# Describing Data:
st.subheader('Data from 2011-2025')
st.write(data.describe())


# Visualizations:
st.subheader('Graph 1:- Closing Price vs Time')
fig = plt.figure(figsize = (12,6))
plt.plot(data.Close)
st.pyplot(fig)

st.subheader('Graph 2:- Closing Price vs Time with 100 Days Moving Averages')
ma100 = data.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r')
plt.plot(data.Close, 'b')
st.pyplot(fig)

st.subheader('Graph 3:- Closing Price vs Time with 100 and 200 Days Moving Averages')
ma100 = data.Close.rolling(100).mean()
ma200 = data.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(data.Close, 'b')
st.pyplot(fig)

# Splitting data into Training and Testing:-
data_training = pd.DataFrame(data['Close'][0:int(len(data)*0.70)]) 
data_testing = pd.DataFrame(data['Close'][int(len(data)*0.70):int(len(data))])
print('\nPrinting shapes of both Splitted Data:-\n')
print(f'For Data Training :- {data_training.shape}')
print(f'For Data Tesing :- {data_testing.shape}')

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

# Load my model:
model = load_model('keras_model.keras')

# Testing data:
past_100_days = data_training.tail(100)
final_df = pd.concat([data_testing, past_100_days], ignore_index = True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test =[]
for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i,0])
x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)  
factor = scaler.scale_
scale_factor =  1/factor[0] # 1/copied value
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Finally Ploting the data:-
st.subheader('Graph 4:-Predictions vs Originals')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
#plt.show()
st.pyplot(fig2)
