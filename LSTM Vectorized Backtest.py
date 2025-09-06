#!/usr/bin/env python
# coding: utf-8

# 

# In[25]:

import pandas as pd
import numpy as np
from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import LSTM 
#from keras.layers.embeddings import Embedding 
from keras.layers import Embedding
from keras.preprocessing import sequence
from keras.layers import Dropout


# ### Get Data from alpaca

# In[ ]:


from Packages.Alpaca_Connection import get_data_today
bars = get_data_today('mmm')
df= get_data_today('mmm')


# In[21]:



#df = bars.df  # Convert to pandas DataFrame
df = df.reset_index()
# As you can see the simple import process allows for us to get an overall view of the daily market trade.  The date of the market instance is indexed making it easy to carry these variables without loosing the time coefficient.  For the aformentioned emphesis on simplicity we are going to focus on the market close price.  
df = df.rename(columns = {'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume', 'trade_count':'Trade Count'})


# It would appear that this may or may not be a great opportunity to buy.  It appears to have experianced a remarkable correction in February.  I hope to create a model that represents the best course of action.  

# ## Prepping the test data
# We are going to predict 38 days of open market into the furture giving us about 15% of the data.  We create a new variable to the dataframe called 'Prediction' where we look at the next 38 days close value and will use it as a response. 

df = df.reset_index()
future_days = 38
df['Prediction'] = df[['Close']].shift(-future_days)


# In[23]:



X = np.array(df.drop(['Prediction','index', 'symbol', 'timestamp'], axis = 1))[:-future_days]
y = np.array(df['Prediction'])[:-future_days]
print("True Values")
print(y)


# Finally, we use the prediction column to create an array of the the future predictor days.  

# Get the last 'x' rows of the feature data set
x_future = df.drop(['index','Prediction', 'symbol', 'timestamp'],axis = 1)[:-future_days]
x_future = x_future.tail(future_days)
x_future = np.array(x_future)
# ## Splitting the data for performance testing
# In addition to the visual analysis, we are going to want to use our conventional Test/train split in order to train the data. We can use our conventional train_test_split dividing our 'live' close date with our prediction close date.   
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y)


# In[26]:


# ## LSTM network
# From background research I have done with predictive stock market modeling it appears that LSTM is the most common method used and I would not be a fool to explore it.  It manages to hold onto important information and 'forget' about useless information hence Long Term Short Memory.  
# ### Reprep the data
# It appeares that the LSTM Model is much more sensitive to the data structure than the other methods.  After running the same data the model was set on deliving a constant.  It appears that the data has to me scaled and properly wrangled before usage.  Hence we will have to reprepare the data.  

# As seen before we are going to look at the market close data and prepare all but 38 values to generate predictions.  

lstmData = df.filter(['Close'])
dataset = lstmData.values
training_data_len = len(dataset)-38


# Now as mentioned above, we have to scale the data.  Using the min max scaler we can turn all of our data into decimals (except for the min =0 and max =1).  We will save this data into our scaled_data datset.    

#Scale the data 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)


# Below you will see that we need to split the data again into x and y train and test.  Using a 60 day offset we can append them to an empty list using the prior 60 days to predict the next 60 days.  

#Re create Training Set 
lstm_train_data = scaled_data[0:training_data_len,:]
test_data = scaled_data[training_data_len - 60: ,:]

# Create the datasets
lstm_x_train = []
lstm_y_train = []
lstm_x_test = []
lstm_y_test = dataset[training_data_len:, :]

for i in range(60, len(lstm_train_data)):
    lstm_x_train.append(lstm_train_data[i-60:i])
    lstm_y_train.append(lstm_train_data[i,0])
    

for i in range(60, len(test_data)): 
    lstm_x_test.append(test_data[i-60:i, 0])    


# Now we have to convert the lists into numpy arrays so the data can be read.  Also, we have to reshape them into a format that the LSTM will understand.  

#Conver to numpy
lstm_x_train, lstm_y_train =   np.array(lstm_x_train), np.array(lstm_y_train)
lstm_x_test = np.array(lstm_x_test)
#Reshape the data
lstm_x_train = np.reshape(lstm_x_train, (lstm_x_train.shape[0], lstm_x_train.shape[1], 1))
lstm_x_test = np.reshape(lstm_x_test, (lstm_x_test.shape[0], lstm_x_test.shape[1], 1))


# Finally, we are back to building the models.  We are using 4 layers: The first two training layers then a dense layer and finally an output layer.  
model = Sequential() 
model.add(LSTM(50, return_sequences = True, input_shape=(lstm_x_train.shape[1],1)))
model.add(LSTM(50, return_sequences = False))
model.add(Dense(25))
model.add(Dense(1))
# Now that we have created the network, we can run the data through using the MSE method.  We will use 10 epocs because the loss is low, but futher iterations are not necessary.  Finally we have to undo the scaling to our predictions.  

#Compile the model 
model.compile(optimizer='adam', loss='mean_squared_error')
#Train the model 
model.fit(lstm_x_train, lstm_y_train, batch_size = 1, epochs = 10)  
LSTM_predictions = model.predict(lstm_x_test)
LSTM_predictions = scaler.inverse_transform(LSTM_predictions)
LSTM_predictions


# In[27]:


# Because of the necessary transformations that were done to the model, we cannot use the same function as above. However we can poach  a massive chunk from above and paste it into our model.  

t = lstmData[:training_data_len]
v = lstmData[training_data_len:]
v['Predictions'] = LSTM_predictions



#mean_squared_error(X_test[:38],LSTM_predictions)

print("Actual Values:")
X_test[0][2]
#df.head()
X_test[0]

    
[row[2] for row in X_test]


# ## Buy/Sell Strategy: 
# If the next prediction is less than current actual: Sell
# If the next prediction is more than the current actual: Buy
# Create a dataframe of the values with the buy/sell signal.  
# Then create look into backtesting the signal.  

# In[37]:


buy_sell = pd.DataFrame({"Actuals": [row[2] for row in X_test][15:]})#, "Predictions":LSTM_predictions })
buy_sell["Predictions"] = LSTM_predictions
buy_sell["Predictions Offset"] = pd.DataFrame(LSTM_predictions).shift(1)
buy_sell["signal"] = np.where(buy_sell["Actuals"]> buy_sell["Predictions Offset"], 1, -1 )
buy_sell["returns"] = np.log(buy_sell["Actuals"]/ buy_sell["Actuals"].shift(1))
buy_sell["strategy"] = buy_sell["returns"]*buy_sell["signal"]
buy_sell


# In[40]:


buy_sell[["returns", "strategy"]].sum()


# In[42]:


buy_sell[["returns", "strategy"]].sum().apply(np.exp)


# In[44]:


buy_sell[["returns", "strategy"]].cumsum().apply(np.exp).plot()


# ## Average period return risk statistics for both the stock and strategy. 
# (32 = # of cycles; 252 = annual period)
# 
# Calculates the period mean return in both log and regular price
#  
# Conclusion(8/30): Equal risk, much higher reward

# In[52]:


buy_sell[['returns', 'strategy']].mean()*32


# In[51]:


np.exp(buy_sell[['returns', 'strategy']].mean()*32)-1


# Calculates the period standard deviation of both log and regular spaces

# In[53]:


buy_sell[['returns', 'strategy']].std()*32**.5


# In[54]:


(buy_sell[['returns', 'strategy']].apply(np.exp)-1).std()*32**.5


# ## Evaluate the drawdaown with the cummax/cumret
# 
# Defines a new column, cumret, with a gross performance over time

# In[55]:


## Evaluate the drawdaown with the cummax/cumret 
buy_sell['cumret'] =buy_sell['strategy'].cumsum().apply(np.exp)


# Defines yet another column with a running maximum value of the gross performance

# In[56]:


buy_sell['cummax'] = buy_sell['cumret'].cummax()


# In[58]:


buy_sell[['cumret', 'cummax']].dropna().plot(figsize= (10,6))


# The max drawdown is calcualated as the difference between the two columns

# In[60]:


drawdown = buy_sell['cummax'] - buy_sell['cumret']
drawdown.max()

