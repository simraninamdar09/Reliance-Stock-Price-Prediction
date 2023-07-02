

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense


# Load the data
data = pd.read_csv('RELIANCE.NS .csv')
data['Date'] = pd.to_datetime(data['Date'])

data['Year'] = data['Date'].dt.year

# Treat outliers using winsorization
q1 = data['Cloe'].quantile(0.25)
q3 = data['Close'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

data['Close] = data['Close'].clip(lower=lower_bound, upper=upper_bound)

# In[332]:

# Treat outliers using IQR method
plt.figure(figsize=(12,8))
q1 = data['Close'].quantile(0.25)
q3 = data['Close'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

data['Treated_Price'] = data['Close'].clip(lower=lower_bound, upper=upper_bound)




#data['Year'] = pd.to_datetime(data['Date']).dt.strftime("%Y")
data['Month'] = pd.to_datetime(data['Date']).dt.strftime('%b')
data['Day'] = pd.to_datetime(data['Date']).dt.strftime("%d")


#OHE
month_dummies = pd.DataFrame(pd.get_dummies(data['Month']))
data = pd.concat([data,month_dummies],axis = 1)

data=data.drop(['Month','Price'],axis=1)
data=data.dropna()
















prices = data['Treated_Price'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# Define the window size for input sequences
window_size = 60

# Prepare the training data
X_train, y_train = [], []
for i in range(window_size, len(scaled_prices)):
    X_train.append(scaled_prices[i - window_size:i, 0])
    y_train.append(scaled_prices[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape the input data for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=32)


# Streamlit app
def predict_price(closing_prices):
    scaled_inputs = scaler.transform(closing_prices)
    inputs = []
    for i in range(window_size, len(scaled_inputs)):
        inputs.append(scaled_inputs[i - window_size:i, 0])
    inputs = np.array(inputs)
    inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], 1))
    predicted_prices = model.predict(inputs)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    return predicted_prices.flatten()


st.title("Reliance Stock Price Prediction")

# Sidebar input
selected_dates = st.sidebar.date_input("Select Dates", [pd.to_datetime("2023-01-01"), pd.to_datetime("2023-02-01")])

start_date, end_date = selected_dates[0], selected_dates[1]

start_index = data[data['Date'] == str(start_date)].index[0]
end_index = data[data['Date'] == str(end_date)].index[0]

selected_prices = prices[start_index - window_size:end_index]

# Prediction
predicted_prices = predict_price(selected_prices)

# Display actual and predicted prices
st.subheader("Actual and Predicted Prices")
df = pd.DataFrame({'Actual Prices': selected_prices.flatten(), 'Predicted Prices': predicted_prices})
st.line_chart(df)

# Display predicted prices table
st.subheader("Predicted Prices Table")
st.dataframe(df)
