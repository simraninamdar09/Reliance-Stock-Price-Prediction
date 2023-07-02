

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
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense
from keras.layers import Dropout

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

data = pd.read_csv('RELIANCE.NS .csv')

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


    





