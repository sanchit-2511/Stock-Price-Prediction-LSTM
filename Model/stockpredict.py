import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



# Fetch 14 years of aaple stock data:-
ticker = 'AAPL'
data = yf.download(ticker, period='14y',interval='1d')
print('\nPrinting the starting few lines of data:-\n')
print(data.head())

# Resetting the index from date to integers:-
data = data.reset_index()
print('\nAfter resetting the indexes:-\n')
print(data.head())
'''
print('\nPrinting the whole data:-\n') #[3521 rows x 6 columns]
print(data)
'''

#------As we are only working here on the closing prices of the stocks-------#


# Plotting the graph of closing price:-

print('\nGraph representing the closing price of AAPL stock:-\n')
plt.plot(data.Close) 
plt.title('Closing price Graph')
plt.show()

# Creating two Moving Averages columns:- 
'''
(Note): o. Moving Average is used when you want to perform specific operation(like here, mean) on all of the data except specific range of the data(like here, 100 days MA). 
        o. It will not give Average of first 100 rows and will continue from 101th row.
        o. To find Moving Average : rolling functon is used.
'''
data['MA100'] = data['Close'].rolling(100).mean()
data['MA200'] = data['Close'].rolling(200).mean()
print('\nAfter Creating columns of Moving Averages of 100 and 200 days:-')
print(data) #[3521 x 8]

# Drop the missing values:-
#data.dropna(inplace=True)


# Plotting the Moving Averages on the Closing Graph:-
print('\nPlotting the 100 Day Moving Averages on the Closing Graph:-\n')
plt.figure(figsize = (12,6))
plt.plot(data.Close)
plt.plot(data.MA100, 'r')
plt.title('100Days MA v/s Closing Price')

print('\nPlotting the 200 Day Moving Averages on the Closing Graph and 100 Day Moving Average:-\n')
plt.figure(figsize = (12,6))
plt.plot(data.Close)
plt.plot(data.MA100, 'r')
plt.plot(data.MA200, 'g')
plt.title('200Days MA v/s Closing Price & 100Days MA')
plt.show()

# Splitting data into Training and Testing:-
data_training = pd.DataFrame(data['Close'][0:int(len(data)*0.70)]) #------Hence Training is done on 70% of data.
data_testing = pd.DataFrame(data['Close'][int(len(data)*0.70):int(len(data))]) #------Testing will be done on rest 30% of data.
print('\nPrinting shapes of both Splitted Data:-\n')
print(f'For Data Training :- {data_training.shape}') #------[2326x1]
print(f'For Data Tesing :- {data_testing.shape}') #------[997x1]

print('\nPrinting starting few rows of Splitted data:-')
print('\nData Training:-')
print(data_training.head())
print('\nData Testing:-')
print(data_testing.head())


# Scaling the data:-
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
# Converting Training Data into an array:-
data_training_array = scaler.fit_transform(data_training)
print('\nAfter Scaling TRAINING DATA from 0 to 1 and converting them into an array:-')
print(data_training_array,'\n')
# Now we have to divide our data into xtrain and ytrain:-
'''
(Note): o. x_train is the data which is already present; y_train is the data we predict.
        o. We have to declare a range of interval(here, no. of days) in x_train to predict y_train. [e.g.;10 days]
           In other words: [1,2,3,4,5,6,7,8,9,10] = 11
        o. Once we predicted the 11th day data and wanted to predict 12th day data,
           The starting day from x_train will replaced by the seconnd day till the 11th day data becomes 10th day data.
           In other words: 1,[2,3,4,5,6,7,8,9,10,11] = 12
        o. And this happens so on.
'''
x_train = []
y_train = []

# We have to Insert Values in them:-
for i in range(100,data_training_array.shape[0]): # Here, Interval is of 100 days starting from 0th index of data_training_array.
    x_train.append(data_training_array[i-100:i]) # Appending 0-100th value of data_training_array into xtrain.
    y_train.append(data_training_array[i,0]) # Appending only one 100th value of data_traiing_array ino ytrain.

# Printing the lists after appendig values in it:-
'''Its Too LONG, better we dont print it'''
'''
print('\nPrinting the lists after appendig values in it:-\n')
print(f'X_Train:-\n{x_train}\n')
print(f'Y_train:-\n{y_train}\n')
'''

# We have to convert this into numpy arrays So we can input them into LSTM Model:-
x_train, y_train = np.array(x_train), np.array(y_train)


# Machine learning Model:-

import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import tensorflow as tf
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

model = Sequential()  # Creating a sequential model.

# 1st Layer
model.add(LSTM(units = 50, activation = 'relu', return_sequences = True, input_shape = (x_train.shape[1],1))) # Adding layers to the sequential model.
model.add(Dropout(0.2)) # Adding a dropout layer.

# 2nd Layer
model.add(LSTM(units = 60, activation = 'relu', return_sequences = True)) # Adding layers to the sequential model.
model.add(Dropout(0.3)) # Adding a dropout layer.

# 3rd Layer
model.add(LSTM(units = 80, activation = 'relu', return_sequences = True)) # Adding layers to the sequential model.
model.add(Dropout(0.4)) # Adding a dropout layer.

# 4th Layer
model.add(LSTM(units = 120, activation = 'relu')) # Adding layers to the sequential model.
model.add(Dropout(0.5)) # Adding a dropout layer.

# Dense Layer to connect all the above layers
model.add(Dense(units = 1)) #For last layer we have to predictonly closing price hence 1 unit.


# We can just see the summary of model:-
model.summary()

# We will now compile our model:-
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(x_train, y_train, epochs = 50)



# Saving our model
model.save('keras_model.keras')


# Testing our data:-
'''
(Note): o. We want to predict our data; we also know, 70% is training and 30% is testing data.
        o. But we need past 100 days data to test/predict our data viz; actually tail part of training data.
        o. hence we will append past 100 days data to testing data for prediction.
'''
past_100_days = data_training.tail(100)

final_df = pd.concat([data_testing, past_100_days], ignore_index = True)

input_data = scaler.fit_transform(final_df) # We have to scale it between 0 to 1 as it isn't scaled.
print(f'\nScaled final Input data:-\n{input_data}\n')
print(f'Shape of our Final Input Data:-\n{input_data.shape}\n')

x_test = []
y_test =[]
for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i,0])
        
x_test, y_test = np.array(x_test), np.array(y_test)
print('\nPrinting shapes of our two testing data:-\n')
print(f'Shape of xtest data:-\n{x_test.shape}')
print(f'Shape of ytest data:-\n{y_test.shape}')


# Making Predictions:- 

y_predicted = model.predict(x_test)
#print(y_predicted.shape)
y_test
y_predicted

factor = scaler.scale_ # It will give us the factor by which all the predicted values are scaled out.
factor
#--copy the output value--#
scale_factor =  1/factor # 1/copied value
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Finally Ploting the data:-
plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()



'''
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Define the LSTM Model
class StockPriceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StockPriceLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)  # Hidden state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)  # Cell state
        out, _ = self.lstm(x, (h0, c0))  # LSTM output
        out = self.fc(out[:, -1, :])  # Fully connected layer (last time step)
        return out

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Convert data to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

# Hyperparameters
input_size = 1
hidden_size = 50
num_layers = 2
output_size = 1
learning_rate = 0.001
num_epochs = 50

# Initialize the model, loss function, and optimizer
model = StockPriceLSTM(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    outputs = model(x_train_tensor)
    optimizer.zero_grad()
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

past_100_days = data_training.tail(100)

final_df = past_100_days.append(data_testing, ignore_index = True)

input_data = scaler.fit_transform(final_df) # We have to scale it between 0 to 1 as it isn't scaled.
x_test = []
y_test =[]
for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i,0])
        
x_test, y_test = np.array(x_test), np.array(y_test)

# Testing the model
model.eval()
x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

y_predicted = model(x_test_tensor).detach().cpu().numpy()
y_test = y_test_tensor.cpu().numpy()

# Rescale the predictions and actual values
scale_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
'''

















'''
from sklearn.processing import MinMaxScalar
from keras.models import Sequential
from keras.layers import LSTM, Dense
'''
