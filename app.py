import streamlit as st
import pandas as pd
import numpy as np
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
data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
# Treat outliers using winsorization
q1 = data['Close'].quantile(0.25)
q3 = data['Close'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
data['Close'] = data['Close'].clip(lower=lower_bound, upper=upper_bound)
# Treat outliers using IQR method
q1 = data['Close'].quantile(0.25)
q3 = data['Close'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
data['Treated_Price'] = data['Close'].clip(lower=lower_bound, upper=upper_bound)

# Treat outliers using winsorization
q1 = data['Volume'].quantile(0.25)
q3 = data['Volume'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
data['Volume'] = data['Volume'].clip(lower=lower_bound, upper=upper_bound)
# Treat outliers using IQR method
q1_v = data['Volume'].quantile(0.25)
q3_v = data['Volume'].quantile(0.75)
iqr = q3_v - q1_v
lower_bound = q1_v - 1.5 * iqr
upper_bound = q3_v + 1.5 * iqr
data['Treated_Volume'] = data['Volume'].clip(lower=lower_bound, upper=upper_bound)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Treated_Price'].values.reshape(-1, 1))

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

# Slider to select the number of days to forecast
days = st.slider('Select the number of days to forecast', 1, 365, 1)

if st.button('Forecast'):
    # Prepare the data for forecasting
    last_data = test_data[-1]
    forecast = []
    for _ in range(days):
        input_data = np.reshape(last_data, (1, 1, 1))
        prediction = model.predict(input_data)
        forecast.append(prediction[0][0])
        last_data = np.append(last_data[1:], prediction[0])


    # Inverse transform the forecasted prices
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
    
    # Create the forecast dataframe
    forecast_dates = pd.date_range(start=data['Date'].iloc[-1], periods=days+1)[1:].strftime('%Y-%m-%d')
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Close': forecast.flatten()})

    # Display the forecasted prices
    st.subheader(f'Forecasted Prices for the next {days} days')
    st.dataframe(forecast_df)

    
