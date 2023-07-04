import streamlit as st
import pandas as pd
import numpy as np
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Function to create LSTM model
def create_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(1, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Load and preprocess the data
data = pd.read_csv('RELIANCE.NS .csv')  # Replace with your own dataset
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Split the data into training and test sets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Prepare the training data
X_train = []
y_train = []
for i in range(1, len(train_data)):
    X_train.append(train_data[i-1:i])
    y_train.append(train_data[i])
X_train = np.array(X_train)
y_train = np.array(y_train)

# Train the LSTM model
model = create_model()
model.fit(X_train, y_train, epochs=30, batch_size=25)

# Prepare the test data
X_test = []
y_test = []
for i in range(1, len(test_data)):
    X_test.append(test_data[i-1:i])
    y_test.append(test_data[i])
X_test = np.array(X_test)
y_test = np.array(y_test)

# Streamlit app
st.title('Stock Market Price Forecasting')
st.write('Predicting stock prices using an LSTM model')
# Convert the 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Extract the month and year from the 'Date' column
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year

# Group the data by month and calculate the average closing price
#monthly_avg = df.groupby(['Year', 'Month'])['Close'].mean().reset_index()

# Train a linear regression model
X = ['Year', 'Month']
y = ['Close']

# Slider to select the number of days to forecast
#days = st.slider('Select the number of days to forecast', 1, 365, 1)

# Create a calendar date picker for start date
start_date = st.date_input("Select start date")

# Create a calendar date picker for end date
end_date = st.date_input("Select end date")




if st.button('Forecast'):
    # Prepare the data for forecasting
    last_data = test_data[-1]
    forecast = []

# Generate predictions for the selected date range
if start_date and end_date:
    # Filter the data based on the selected date range
    filtered_data = monthly_avg[(monthly_avg['Date'] >= start_date) & (monthly_avg['Date'] <= end_date)]

    # Extract the year and month from the filtered data
    X_pred = filtered_data[['Year', 'Month']]

    # Generate predictions for the selected date range
    predictions = model.predict(X_pred)

    # Create a DataFrame with the predicted values
    predicted_data = pd.DataFrame({'Date': filtered_data['Date'], 'Prediction': predictions})

    # Display the predicted data
    st.write(predicted_data)
    
    for _ in range(start_date,end_date):
        input_data = np.reshape(last_data, (1, 1, 1))
        prediction = model.predict(input_data)
        forecast.append(prediction[0][0])
        last_data = np.append(last_data[1:], prediction[0])
        
    # Inverse transform the forecasted prices
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

    # Create the forecast dataframe
    forecast_dates = pd.date_range(start=data['Date'].iloc[-1], periods=days+1)[1:].strftime('%Y-%m-%d')
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast.flatten()})


    
    # Display the forecasted prices
    st.subheader(f'Forecasted Prices for the next {days} days')
    st.dataframe(forecast_df)

    # Plot the forecasted prices
    fig, ax = plt.subplots()
    ax.plot(data['Date'], data['Close'], label='Actual')
    ax.plot(forecast_dates, forecast, label='Forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title('Forecasted Stock Prices')
    ax.legend()
    st.pyplot(fig)
