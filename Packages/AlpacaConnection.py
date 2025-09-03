#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas as pd
from datetime import datetime
from datetime import date
from dateutil.relativedelta import relativedelta




# Load .env into environment variables
load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")

client = StockHistoricalDataClient(API_KEY, API_SECRET)
# Define request
symbol = 'mmm'
request = StockBarsRequest(
    symbol_or_symbols=[symbol],
    timeframe=TimeFrame.Day,
    start=datetime.combine(date.today()- relativedelta(years=1), datetime.min.time()),
    end=datetime.combine(date.today(), datetime.min.time())
)

# Fetch data
bars = client.get_stock_bars(request)
df = bars.df  # Convert to pandas DataFrame
df = df.reset_index()


# In[ ]:




