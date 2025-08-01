#!/usr/bin/env python
# coding: utf-8

# # Final Project
# The purpose of this project is to develop an elemantary understanding of algorytheomic trading. In the furure I hope to build a system that can make serious cash.  One project will not likely be sufficient - I suspect that it will require an effort equivilant to earning a masters degree.  
# 
# My expectations is to develop a model that can read ANY stock and successfully predict a few weeks of 3M stock readings.  The models being produced in this project will be based on time series algorytheoms, which allows for a 'one size fits all' approach, which allows for future scanning for grand opportunities.  One major advantage to this approach is that after the project, I can review the project and scan for the best opportunities to either invest, liquidate or continue building the model (ephasis on the last).  Because the sample size for predictions is so small, the hyperparameters must be vauge and hardly used to prevent from overfitting the model.  Most of the interpetation of the model will be based on the RMSE and the visualiztion of the model.  
# 
# Because of my desire to keep this project as simple as possible- allowing for forward complexity.  I choosed 3M because I have experiance with the company giving me insights to how to further continue the model.  The stock is also considered a high cap company making it more stable and less likely to be a victim of overfitting.  Finally - but not so importantly, this is my favorite market symbol; 3M = MMM is funny to me!

# # Load the data
# 
# We are looking for a buy/sell signal.  We hope to have these signals on a daily basis, which gives us a cost effective(data thrity) way of collecting data and seeing long term trends.  No matter the time, we will always look at one year of data and will have 252 to 253 trading days worth of data.  
# 
# Using the pandas_datareader you get an amazing tool that allows you to read live market values up to 10000 readings a second!  However, what I must celebrate what this does for this project is it's easy to read!  I have programmed the parameters to capture the entire year based on the date - making my findings time sensitive.  

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


# As you can see the simple import process allows for us to get an overall view of the daily market trade.  The date of the market instance is indexed making it easy to carry these variables without loosing the time coefficient.  For the aformentioned emphesis on simplicity we are going to focus on the market close price.  

df = df.rename(columns = {'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume', 'trade_count':'Trade Count'})
df


# ## Visualize the data 
# Becuase we are going to use our eyes to determine the quality of the model, paired with RMSE, we are going to want to start by getting an overview of the stock in addition to buiding a market strategy to the project.  

import matplotlib.pyplot as plt 
plt.figure(figsize=(16,8))
plt.title(symbol)
plt.xlabel('Days')
plt.ylabel('Close Price')
plt.plot(df['timestamp'], df['Close'])
plt.show()


# It would appear that this may or may not be a great opportunity to buy.  It appears to have experianced a remarkable correction in February.  I hope to create a model that represents the best course of action.  

# ## Prepping the test data
# We are going to predict 38 days of open market into the furture giving us about 15% of the data.  We create a new variable to the dataframe called 'Prediction' where we look at the next 38 days close value and will use it as a response. 

df = df.reset_index()
future_days = 38
df['Prediction'] = df[['Close']].shift(-future_days)
import numpy as np
X = np.array(df.drop(['Prediction','index', 'symbol', 'timestamp'], axis = 1))[:-future_days]
y = np.array(df['Prediction'])[:-future_days]
print(y)


# Finally, we use the prediction column to create an array of the the future predictor days.  

# Get the last 'x' rows of the feature data set
x_future = df.drop(['index','Prediction', 'symbol', 'timestamp'],axis = 1)[:-future_days]
x_future = x_future.tail(future_days)
x_future = np.array(x_future)
x_future


# ## Splitting the data for performance testing
# In addition to the visual analysis, we are going to want to use our conventional Test/train split in order to train the data. We can use our conventional train_test_split dividing our 'live' close date with our prediction close date.   

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y)


# # Building the models 
# Below you will see that I have build five models that can be evaluated and determined if the stock price is predictiable using time series analysis.  Because I have condidtioned the model to train against future values, we are going to want to use regressive models instead of classification models(using such for a continued analysis may be a good practice for building buy/sell signals). The models include a Decision Tree Regressor, a Linear Regression, Random Forest Regression, and a LSTM Model.  The three former models have been used successfully in the class and I hope to apply them successfully to this challenge.  The fourth came up when looking into the subject.  It appears to be the most popular method at the moment and I think this is a great opportunity to learn something new and work with an industry standard.  

# ## Producing the visualization 
# Below we will be visualizing the data with a redundant process - hence we will be using a function that plots our findings.  The only input required is the predictions array.  
# 

import matplotlib.pyplot as plt 
def plot_model(pre):
    valid = df[X.shape[0]:]
    valid['Predictions'] =pre
    plt.figure(figsize = (16,8))
    plt.title('Model')
    plt.xlabel('Days')
    plt.ylabel('Close Price USD ($)')
    plt.plot(df['Close'])
    plt.plot( valid[['Close', 'Predictions']])
    plt.legend(['Orig', 'Val', 'Pred'])
    plt.show()


# ## Linear Model

from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X_train, y_train)
lr_prediction = lr.predict(x_future)
print(lr_prediction)


# I have to admit, this model acted different than I expected.  I would assume that this model would have a consistant tragectory as the data without any cycles or harmonics.  It appears that the model anticipate the correction, but split the difference between the begining of the prediction cycle and the end of the prediction cycle.   

plot_model(lr_prediction)


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test[:38],lr_prediction)


lr_prediction


# ## Tree Regression

# Create the models: Tree
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor().fit(X_train, y_train)
tree_prediction = tree.predict(x_future) 
print(tree_prediction)


# As you can see this model mirrors the entire correction period.  You will also notice that the model is a tad more dramatic day to day, representing more agressive buy and sell signals.  You will also notice that our final prediction is an agressive buy. It appears that the model has learned for every correction, there appears to be a bounceback and it is timing that correction for right now.  I do appreciate the model to recognize this cycle, but I am convinced that the underlying cause of the correction is due to current tragic global events and the rising price of oil.  There was not any of this information fed into the model, and better judgement would dictate that this model is to be ignored until a less eventful time.   

plot_model(tree_prediction)


mean_squared_error(y_test[:38],tree_prediction)


# ## Random forest regression 
# Given our luck with the Tree Regression, intuition tells us to use a random forest (Q: Whats better than a decision tree? A: A bunch of decision trees!).  

# Build a Random Forest 
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
regr = RandomForestRegressor(max_depth=2, random_state=0)
rf = regr.fit(X_train, y_train)
rf_prediction = rf.predict(x_future)
print(rf_prediction)


# It appears the the Random Forest has a very similar approach as the linear model.  That is to embrace the correction and split the difference between the early February values and the bottom of the value.  This is likely that the model learned a few corrections and thinks that this current correction is dramatic and money can be made by assuming that the value is overweight.  Again, there are factors on wall street that have given most stocks very agressive selloffs that have not been seen in the past year that the model is not aware of.  

plot_model(rf_prediction)


# ## SVM Regression

# SVM  
from sklearn.svm import SVR 
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler
regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
svm = regr.fit(X_train, y_train)
svm_prediction = svm.predict(x_future)
print(svm_prediction)


# This is my worst model yet! According to this model, the value of 3M stock appears to be business as usual - if only!  It must be that none of the prediction values trigger a correction that indicates a dramatic decrease in the stock price.    

plot_model(svm_prediction)


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




from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import LSTM 
#from keras.layers.embeddings import Embedding 
from keras.layers import Embedding
from keras.preprocessing import sequence
from keras.layers import Dropout
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


# Because of the necessary transformations that were done to the model, we cannot use the same function as above. However we can poach  a massive chunk from above and paste it into our model.  

t = lstmData[:training_data_len]
v = lstmData[training_data_len:]
v['Predictions'] = LSTM_predictions

plt.figure(figsize=(16,8))
# Include the title 
# icldue xlabel 
#include y label 
plt.plot(t['Close'])
plt.plot(v[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc = 'lower right')
plt.show()


#mean_squared_error(X_test[:38],LSTM_predictions)


X_test[0][2]
#df.head()


len(X_test[:38])== len(LSTM_predictions)


X_test


len(y)


# If you agree this appears to be the best model yet.  It recognizes the bear market we are approaching and finds a similar bottom.  The data is not too "bouncy", which is an indication that there is not a lot of overfit happening.  

# # Conclusion
# As you can see with some ML methods we can predict the market value of 3M stock among other symobls. It appears the LSTM is the best.  Some of the better models built look promising, but is definately not ready for me to bet my savings or 401K on it(maybe I need to invest in more education).  I am quite satisfied with this proof of concept and am looking forward to reading additional literature on the subject.  
# 
# ## Continuing the project  
# Again, this project was robust, but still a proof of concept.  There are dozens of strategies in the industry, and I simply threw ML models at stock ticking data.  When it comes to applying additional information to the model, I would like to add additional inputs like the cost of oil or the stock of a compeditor like GE. Finally, I would like to study the metrics that best predict sucess (MSE, RMSE, ect) so I can run 500 iterations of the program to look for the best promising model and make some serious cash.  
# 
