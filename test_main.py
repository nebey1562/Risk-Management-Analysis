import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from create import create_dataset
from strategy1 import combine


class Bot:

    def __init__(self,symbol,start,end):
        self.symbol=symbol
        self.start=start
        self.end=end
        self.df=None
        self.df_full=None
        self.close=None

        #correcting path of csv file
        
        self.path= 'Nifty50stocks/' + self.symbol + '.NS.csv'

    def read_full(self):

        #accessing the whole data
        self.df_full=pd.read_csv(self.path)
        return self.df_full

    def read(self):
    
        # Reading the CSV file
        self.df = pd.read_csv(self.path)

        # Converting the date to datetime format
        self.df['Date'] = pd.to_datetime(self.df['Date'])

        # Setting index as date for parsing
        self.df.set_index('Date', inplace=True)
    
        # Filtering data based on start and end dates
        self.df = self.df.loc[self.start:self.end]

        return self.df

    
    def bh_return(self):

        self.read()

        #caluclating the cumilative sum(pnl each day)
        self.close=self.df['Close'].to_frame()
        self.close['lag1']=self.close.shift(periods=1)
        self.close['pnl']=self.close.Close.sub(self.close.lag1).cumsum()
        final_pnl=self.close['pnl'].iloc[-1]

        return 'final pnl:{:.2f}'.format(final_pnl)
    

    def bh_plot(self):

        self.bh_return()
        
        plt.figure(figsize=(15, 8))
        plt.plot(self.close['pnl'], label='Cumulative Returns', linestyle='-', linewidth=1.5, color='blue')

        # Adding labels and title
        plt.title("Buy and Hold Equity Curve", fontsize=16)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Cumulative Returns", fontsize=12)

        # Adding grid lines
        plt.grid(True, linestyle='--', alpha=0.7)

        # Adding legend
        plt.legend(loc='upper left', fontsize=12)

        # Adding a horizontal line at y=0 for reference
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)

        # Adding a background color for positive and negative returns
        plt.fill_between(self.close.index, 0, self.close['pnl'], where=(self.close['pnl'] >= 0), color='green', alpha=0.3)
        plt.fill_between(self.close.index, 0, self.close['pnl'], where=(self.close['pnl'] < 0), color='red', alpha=0.3)

        # Rotating x-axis labels for better readability
        plt.xticks(rotation=45)

        # Displaying the plot
        plt.show()

    def model(self):
        df = self.read_full()
        df = df.dropna()

        # Filtering to 'Close' column and converting to numpy array
        data = df.filter(['Close'])
        dataset = data.values

        # Calculating the training data length
        training_data_len = int(np.ceil(len(dataset) * .75 ))

        # Scaling the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        # Splitting the scaled data into training and test sets
        train_data = scaled_data[0:training_data_len, :]
        test_data = scaled_data[training_data_len - 100:, :]  # Keep the overlap for context


        # Preparing the training and test sequences
        step = 90
        x_train, y_train = create_dataset(train_data, step)
        x_test, y_test = create_dataset(test_data, step)

        # Reshaping input to be [samples, time steps, features]
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        # Adding noise to training data
        x_train_noisy = x_train + np.random.normal(0, 0.1, x_train.shape)
        y_train_noisy = y_train + np.random.normal(0, 0.1, y_train.shape)

        # Building the LSTM model with Dropout
        model = Sequential([
            LSTM(45, return_sequences=True, input_shape=(x_train.shape[1], 1)),
            Dropout(0.4),
            LSTM(20, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])

        # Compiling the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Adding Early Stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=5)

        # Training the model with early stopping
        model.fit(x_train_noisy, y_train_noisy, validation_data=(x_test, y_test), batch_size=15, epochs=8, callbacks=[early_stop])

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

        return test_predict,data

        

    


