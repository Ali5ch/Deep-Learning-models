#   Recurrent Neural Network
"""
Created on Fri Jun 16 23:46:48 2017

@author: Ali Abbas
"""
# Part 1 - Data Preprocessing 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

# Importing the libraries 
training_set = pd.read_csv('Google_Stock_Price_Train.csv') 
training_set = training_set.iloc[:, 1:2].values

#Feature Scaling 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler() # normalization 
training_set = sc.fit_transform(training_set)

# Getting the inputs and the outputs
X_train = training_set[0:1257]
y_train = training_set[1:1258]

# Reshaping 
X_train = np.reshape(X_train, (1257, 1, 1))

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Initialising the RNN
regressor = Sequential()

# Adding the input layer and LSTM layer
regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None,1)))

# Adding the Output layer 
regressor.add(Dense(units = 1))

# compining the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the training set 
regressor.fit(X_train, y_train, batch_size = 32, epochs = 200)

# Part3 - making the prediction and visualizing the results

# Getting the real stock price 2017
test_set = pd.read_csv('Google_Stock_Price_Test.csv') 
real_stock_price = test_set.iloc[:, 1:2].values

# Getting the real stock price of 2017
inputs = real_stock_price
inputs = sc.transform(inputs) # normalizing the test set values as the machine was trained on skaled vales 
inputs = np.reshape(inputs, (20, 1, 1)) # same here have to reshape the inputs, to make 3d array 
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualiting the results 
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()