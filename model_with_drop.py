import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

# Assuming your CSV loading and initial data cleaning remain the same
df = pd.read_csv("Nifty50stocks/AXISBANK.NS.csv")
df = df.dropna()

# Filtering to 'Close' column and converting to numpy array
data = df.filter(['Close'])
dataset = data.values

# Calculating the training data length
training_data_len = int(np.ceil(len(dataset) * .80 ))

# Scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Splitting the scaled data into training and test sets
train_data = scaled_data[0:training_data_len, :]
test_data = scaled_data[training_data_len - 100:, :]  # Keep the overlap for context

# Function to create the sequences
def create_dataset(dataset, step):
    X, Y = [], []
    for i in range(len(dataset) - step - 1):
        a = dataset[i:(i + step), 0]
        X.append(a)
        Y.append(dataset[i + step, 0])
    return np.array(X), np.array(Y)

# Preparing the training and test sequences
step = 90
x_train, y_train = create_dataset(train_data, step)
x_test, y_test = create_dataset(test_data, step)

# Reshaping input to be [samples, time steps, features]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Building the LSTM model with Dropout
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

# Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Adding Early Stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10)

# Training the model with early stopping
model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=15, epochs=10, callbacks=[early_stop])

# Making predictions
train_predict = model.predict(x_train)
test_predict = model.predict(x_test)

# Inverting predictions back to original scale
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])

# Plotting the training data results
plt.figure(figsize=(14, 7))
plt.plot(y_train[0], 'b-', label="Actual (Training)")
plt.plot(train_predict[:,0], 'r-', label="Predicted (Training)")
plt.title('Stock Price Prediction (Training)')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Plotting the testing data results
plt.figure(figsize=(14, 7))
plt.plot(y_test[0], 'b-', label="Actual (Testing)")
plt.plot(test_predict[:,0], 'r-', label="Predicted (Testing)")
plt.title('Stock Price Prediction (Testing)')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()