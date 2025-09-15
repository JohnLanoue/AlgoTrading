import numpy as np
from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import LSTM 
from keras.layers import Embedding
from keras.preprocessing import sequence
from keras.layers import Dropout


def lstm_prep(df, future_days = 38):
#.  PRE: Pandas is already installed and we recieve a data frame
#.  POST: We have split data frames that is ready for train/testing.  
#.  Intent: Take the incoming data frame and break it out to train/test
#.  Improvement: Convert the output to df.  

#.  THis is the target preictor: Conside
    df['Prediction'] = df[['Close']].shift(-future_days)
    
#.  We identify the predictor/response.  
    X = np.array(df.drop(['Prediction','index', 'symbol', 'timestamp'], axis = 1))[:-future_days]
    y = np.array(df['Prediction'])[:-future_days]


    # Finally, we use the prediction column to create an array of the the future predictor days.  
    # Get the last 'x' rows of the feature data set
    x_future = df.drop(['index','Prediction', 'symbol', 'timestamp'],axis = 1)[:-future_days]
    x_future = x_future.tail(future_days)
    x_future = np.array(x_future)
    # ## Splitting the data for performance testing
    # In addition to the visual analysis, we are going to want to use our conventional Test/train split in order to train the data. We can use our conventional train_test_split dividing our 'live' close date with our prediction close date.   
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    return X_train, X_test, y_train, y_test
    
    
    
def lstm(df, future_days = 38):
#.   PRE: We recieve a data frame that is fit for LSTM training.  
#.   Post: We return our predicted values.  
#.   Intent: TO be able to generate the prdeictions from the lSTM 
#.   Improvements: Return the test results: In case we get a low accuracy.  More parameters for fiddlingwith the layers and hyperparameters.  

    # As seen before we are going to look at the market close data and prepare all but 38 values to generate predictions.  
    lstmData = df.filter(['Close'])
    dataset = lstmData.values
    training_data_len = len(dataset)-future_days


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


    # Because of the necessary transformations that were done to the model, we cannot use the same function as above. However we can poach  a massive chunk from above and paste it into our model.  

    t = lstmData[:training_data_len]
    v = lstmData[training_data_len:]
    v['Predictions'] = LSTM_predictions



#    print(mean_squared_error(X_test[:38],LSTM_predictions))
#    print("Actual Values:")
#    print([row[2] for row in X_test])
    return  LSTM_predictions# [row[2] for row in X_test]



def vecorized_buy_sell(ls_pred, arr_actuals):
    import pandas as pd
    buy_sell = pd.DataFrame({"Actuals": [row[2] for row in arr_actuals][:]})#, "Predictions":LSTM_predictions })
    buy_sell["Predictions"] = ls_pred
    buy_sell["Predictions Offset"] = pd.DataFrame(ls_pred).shift(1)
    buy_sell["signal"] = np.where(buy_sell["Actuals"]> buy_sell["Predictions Offset"], 1, -1 )
    buy_sell["returns"] = np.log(buy_sell["Actuals"]/ buy_sell["Actuals"].shift(1))
    buy_sell["strategy"] = buy_sell["returns"]*buy_sell["signal"]
    return buy_sell

def df_finalize(df ):
#   PRE: Recieves a data frame from the Alpaca_connection.get_data_today package.  Pandas is already installed.  
#   POST: Finalizes a data frame that is ready for next steps. 
#   Intent: Band-aid for data handling.  Consider closing the gap in get_data_today
#   Improvements: Replace function ALL together
    df = df.reset_index()
    df = df.rename(columns = {'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume', 'trade_count':'Trade Count'})
    return df