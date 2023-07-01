
!pip install tensorflow
# In[321]:
import streamlit as st
import pandas as pd
from tensorflow.keras.layers import LSTM, Dense
# importing libraries
import pandas as pd
import numpy as np
# to Visualize
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import Grouper
from pandas import DataFrame
from pandas.plotting import lag_plot
import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings

warnings.filterwarnings('ignore')
# For stationarity check
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import pickle

# In[150]:
st.markdown('''
<style>
.stApp {

    background-color:#8DC8ED;
    align:center;\
    display:fill;\
    border-radius: false;\
    border-style: solid;\
    border-color:#000000;\
    border-style: false;\
    border-width: 2px;\
    color:Black;\
    font-size:15px;\
    font-family: Source Sans Pro;\
    background-color:#8DC8ED;\
    text-align:center;\
    letter-spacing:0.1px;\
    padding: 0.1em;">\
}
.sidebar {
    background-color: black;
}

.st-b7 {
    color: #8DC8ED;
}
.css-nlntq9 {
    font-family: Source Sans Pro;
}
</style>
''', unsafe_allow_html=True)

data = pd.read_excel('RELIANCE.NS .csv', parse_dates=True, squeeze=True)

data['Date'] = pd.to_datetime(data['Date'])

data['Year'] = data['Date'].dt.year

# Treat outliers using winsorization (close column)
q1 = data['Close'].quantile(0.25)
q3 = data['Close'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

data['Close'] = data['Close'].clip(lower=lower_bound, upper=upper_bound)

# In[332]:

# Treat outliers using IQR method
plt.figure(figsize=(12, 8))
q1 = data['Close'].quantile(0.25)
q3 = data['Close'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

data['Treated_Price'] = data['Close'].clip(lower=lower_bound, upper=upper_bound)

# Treat outliers using winsorization (Volume column)
q1 = data['Volume'].quantile(0.25)
q3 = data['Volume'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

data['Volume'] = data['Volume'].clip(lower=lower_bound, upper=upper_bound)

# In[332]:

# Treat outliers using IQR method
plt.figure(figsize=(12, 8))
q1 = data['Volume'].quantile(0.25)
q3 = data['Volume'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

data['Treated_Volume'] = data['Volume'].clip(lower=lower_bound, upper=upper_bound)

# data['Year'] = pd.to_datetime(data['Date']).dt.strftime("%Y")
data['Month'] = pd.to_datetime(data['Date']).dt.strftime('%b')
data['Day'] = pd.to_datetime(data['Date']).dt.strftime("%d")
#--------------------------------Visual-----------------------------
plt.figure(figsize=(20,10))
# create a density plot
plt.subplot(2,2,1)
sns.lineplot(x='Date',y='Treated_Price',data=data)
plt.title('Price',fontsize=20)

plt.subplot(2,2,2)
sns.lineplot(x='Date',y='Treated_Volume',data=data)
plt.title('Volume',fontsize=20)
plt.show()

plt.figure(figsize=(20,10))
# create a density plot
plt.subplot(2,2,1)
data['Treated_Price'].plot(kind='hist')
plt.title('Price',fontsize=20)

plt.subplot(2,2,2)
data['Treated_Volume'].plot(kind='hist')
plt.title('Volume',fontsize=20)
plt.show()

plt.figure(figsize=(20,10))
# create a density plot
plt.subplot(2,2,1)
data['Treated_Price'].plot(kind='kde')
plt.title('Price',fontsize=20)

plt.subplot(2,2,2)
data['Treated_Volume'].plot(kind='kde')
plt.title('Volume',fontsize=20)
plt.show()
#-----------------Skew-------------
data['Treated_Price']=np.sqrt(data['Treated_Price'])
data['Treated_Volume']=np.sqrt(data['Treated_Volume'])

# OHE
month_dummies = pd.DataFrame(pd.get_dummies(data['Month']))
data = pd.concat([data, month_dummies], axis=1)

data = data.drop(['Month', 'Close'], axis=1)
data = data.dropna()

# Rename the columns to match Prophet's requirements
data = data.rename(columns={'Date': 'ds', 'Treated_Price': 'y'})
# data.head()
# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data['Treated_Price'].values.reshape(-1, 1))

# Split the data into train and test sets
train_size = int(len(scaled_data) * 0.80)  # 80% for training, 20% for testing
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Define the function to create input and output sequences for LSTM
def create_sequences(data, sequence_length):
    X = []
    y = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

# Set the sequence length and create sequences for training
sequence_length = 10
X_train, y_train = create_sequences(train_data, sequence_length)

# Create the LSTM model
model1 = Sequential()
model1.add(LSTM(units=50, return_sequences=True, input_shape=(sequence_length, 1)))
model1.add(LSTM(units=50, return_sequences=False))
model1.add(Dense(units=1))
model1.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Train the model
model1.fit(X_train, y_train, epochs=30, batch_size=25)

# Make predictions on the test set
X_test, y_test = create_sequences(test_data, sequence_length)
y_pred = model1.predict(X_test)

# Inverse scale the predictions and the actual values
y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
print(y_pred_inv)
print(y_test_inv)




    





