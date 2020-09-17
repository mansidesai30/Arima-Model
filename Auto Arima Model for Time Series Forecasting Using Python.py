#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pmdarima as pm
import matplotlib.pyplot as plt


# In[13]:


import pandas as pd
# Import
data = pd.read_csv(r"C:\Users\NQE00254\Desktop\Power BI Reports\Data Science Courses\Python\AirPassengers.csv")
data['Date'] = pd.to_datetime(data['Date'])
data.index.freq = 'MS'
data.head(10)
# Plot
fig, axes = plt.subplots(2, 1, figsize=(10,5), dpi=100, sharex=True)

# Usual Differencing
axes[0].plot( data.Passengers ,label='Original Series')
axes[0].plot(data.Passengers.diff(1), label='Usual Differencing')
axes[0].set_title('Usual Differencing')
axes[0].legend(loc='upper left', fontsize=10)


# Seasinal Dei
axes[1].plot( data.Passengers, label='Original Series')
axes[1].plot( data.Passengers.diff(12), label='Seasonal Differencing', color='green')
axes[1].set_title('Seasonal Differencing')
plt.legend(loc='upper left', fontsize=10)
plt.suptitle('a10 - Airline Passenger Details', fontsize=16)
plt.show()


# In[14]:


# !pip3 install pyramid-arima
import pmdarima as pm

# Seasonal - fit stepwise auto-ARIMA
smodel = pm.auto_arima(data.Passengers, start_p=1, start_q=1,
                         test='adf',
                         max_p=3, max_q=3, m=12,
                         start_P=0, seasonal=True,
                         d=None, D=1, trace=True,
                         error_action='ignore',  
                         suppress_warnings=True, 
                         stepwise=True)

smodel.summary()


# In[15]:


#import
from pandas.tseries.offsets import DateOffset


# In[16]:


# Forecast
n_periods = 24
fitted, confint = smodel.predict(n_periods=n_periods, return_conf_int=True)

#index_of_fc = pd.date_range(data.DATE.index[1], periods = n_periods, freq='MS')
index_of_fc = pd.date_range('1960-12-01', periods = n_periods, freq='MS')

#index_of_fc=[data.index[-1]+ DateOffset(months=x)for x in range(0,2)]

# make series for plotting purpose
fitted_series = pd.Series(fitted, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# Plot
plt.plot(data.Passengers)


# In[17]:


import chart_studio.plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go
import plotly.graph_objs as go
import chart_studio.plotly as py
#plot monthly sales
plot_data = [
    go.Scatter(
        x = data.Passengers,
        y = data.Date
    )
]
plot_layout = go.Layout(
        title='Test Prediction'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[18]:


lower_series.plot()


# In[19]:


upper_series.plot()


# In[20]:


fitted_series.plot()


# In[21]:


#data.plot(legend=True,label='data')
fitted_series.plot(legend=True,label='Fitted Series',figsize=(12,8))
upper_series.plot(legend=True,label='Upper Series',figsize=(12,8))
lower_series.plot(legend=True,label='Lower Series',figsize=(12,8))


# In[ ]:




