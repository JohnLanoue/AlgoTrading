#!/usr/bin/env python
# coding: utf-8

# ## Dependant libraries: 
# Below, we import all of the libraries in scope.  This project will require getting data from yahoo finance articles and ticker data.  These will require seperate apis for pulling the data independatly.  Further into the project, we will need to be leveraging the NLP functions as well as machine learning fitting and evaulation packages.  

# In[1]:


#Given Data science
import pandas as pd
import numpy as np
# Web Scraping Support
import requests
from bs4 import BeautifulSoup
from datetime import timedelta, date, datetime
#NLP Essentials
from nltk.tokenize import word_tokenize
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
#NLP Additions
nltk.download('punkt')
nltk.download('punkt')
nltk.download('stopwords')

#Basic ML Packages
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM

#Finance
import yfinance as yf
#MISC
import asyncio
#import config
#from alpaca.trading.client import TradingClient
#import alpaca_trade_api as api
#from alpaca.trading.requests import MarketOrderRequest
#from alpaca.trading.enums import OrderSide, TimeInForce
#import praw


# ## Aquiring Data
# Below we be looking for articles that are included for a specific symbol.  These articles will be scraped using beautiful soup then inserted into tokens.  A date will be extracted in order to look up the the ticker data. Unfortuatly, the number of articles that may be requested without risking a dropped conection is limited. We will be targeting following Medicare giants: United Health Group, Humana, CVS, Cigna, Evolent, Molina, Clover Health.  The intent of limiting the project to a single topic is that vocabulary can vary from industry to industry.  An example is that an article that includes "Medicare Advantage" should be neutral, while most indutries would recognnize ".... Advantage" as positive.  Therefore, I would like to train them independantly.  

# In[20]:


def scrape_yahoo_finance_news(symbol = 'UNH'):
    # URL of Yahoo Finance's news page
#    url = 'https://finance.yahoo.com/news/'
#    url = 'https://finance.yahoo.com/quote/UNH?.tsrc=fin-srch'

    # Send a GET request to the URL
    url = 'https://finance.yahoo.com/quote/'+symbol+'/?.tsrc=fin-srch'
    response = requests.get(url)

    # Parse the HTML content of the page using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all news article headlines and links
    headlines = soup.find_all('h3', class_='Mb(5px)')

    # Extract headlines and links
    news_data = []
    for headline in headlines:
        title = headline.get_text()
        link = headline.find('a')['href']
        news_data.append({'title': title, 'link': link, 'Symbol': symbol})

    return news_data


# In[3]:


def tokenize_article(url):
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Parse the HTML content of the page using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the main article content
        article_content = soup.find('div', class_='caas-body')
        if article_content is None:
            # If the article content is not found under 'caas-body', try 'cass-body-content'
            article_content = soup.find('div', class_='caas-body-content')

        if article_content:
            # Extract text from the article content
            article_text = article_content.get_text()

            # Tokenize the words
            tokens = word_tokenize(article_text)

            return tokens
        else:
            return "Article content not found."
    except Exception as e:
        return f"Error occurred: {e}"


# In[4]:


def medicare_validation(tokens): 
    for i in tokens: 
        if i  == "Medicare":
            return True 
    return False


# In[5]:


def add_articles(chunk): 
    #PRE: Adding a list object that includes the news title (technically optional) and the link[suffix] to the article)
    #POST: Each element of the list includes the article in the dictionary object
    for i in chunk: 
        i['Article'] = tokenize_article('https://finance.yahoo.com'+ i['link'])
    return chunk


# ## Gathering the data
# These steps below will be leveraging the functions above that will create an list of dictionaries object, based on a single stock symbol (or keyword technically).
# ### Below we are showing the  titles of the articles and the path beyond the yahoo finance root
# A small sample of title, link, symbols are included below.  

# In[21]:


yahoo_finance_news=[]
#topics = ['UNH','HUM', 'CVS', 'CI', 'ELV', 'MOH', 'CLOV']
topics = ['AAPL',	'GOOGL',	'MSFT',	'AMZN',	'FB',	'TSLA',	'JPM',	'JNJ',	'NVDA',	'V',	'BAC',	'WMT',	'PG',	'MA',	'DIS',	'HD',	'INTC',	'UNH',	'VZ',	'CRM',	'NFLX',	'PYPL',	'CMCSA',	'ORCL',	'NKE',	'CSCO',	'PFE',	'XOM',	'T',	'ADBE',	'CVX',	'MRK',	'KO',	'WFC',	'PEP',	'ABT',	'IBM',	'BA',	'COST',	'MCD',	'SBUX',	'BMY',	'LLY',	'MMM']#,	'AAP',	'HON']#,	'TXN',	'AVGO',	'GM',	'C',	'LMT',	'TMO',	'MDLZ',	'UPS',	'CHTR',	'DHR',	'SBAC',	'NOW',	'ACN',	'AMGN',	'CAT',	'ABBV',	'CME',	'NEE',	'AXP',	'SPGI',	'MDT',	'GS',	'LIN',	'FIS',	'GE',	'KMB',	'GILD',	'FDX',	'CVS',	'DUK',	'WBA',	'SO',	'ISRG',	'BKNG',	'CRM',	'ECL',	'NEE',	'DD',	'NSC',	'TGT',	'DUK',	'LOW',	'MS',	'RTX']
for i in topics: 
    yahoo_finance_news.append(scrape_yahoo_finance_news(i))
yahoo_finance_news = [item for sublist in yahoo_finance_news for item in sublist]    
yahoo_finance_news[:4]


# ### Now we are creating a new object, like above including the article content
# This object will temporarily hold all of the articles.  The articles will be formatted in tokens.  

# In[9]:


pred = add_articles(yahoo_finance_news)
predtmp = pred
#pred[2] 


# ## Handle the text
# As you can see the incoming data from the article is still pretty raw.  We should preprocess the text for sentiment analysis: 
#     Handling the case, so that puncuation patterns do not generate their own noisy indicators
#     Eliminating stopowords, in order to be ignore the evaluation of meaninless words
#     Stemmer: We will be using porter.  This will unite a lot of different words by eliminating their participals.
#     

# In[10]:


# Download NLTK resources (if not already downloaded)
stemmer = PorterStemmer()

def pre_process_text(tokens): 
    #Case Handle
    tokens = [token.lower() for token in tokens]
    #Stopword handle
    stop_words = set(stopwords.words('english'))
    stop_words = [token for token in tokens if token not in stop_words]
    #Remove Punctuation
    tokens = [token for token in tokens if token.isalnum()]
    #Stem words: Porter
#    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens] 
    return tokens


# Below we implement all of the incoming text into our pre process.  

# In[11]:


for i in range(0,len(pred)):
    pred[i]['Article'] = pre_process_text(pred[i]['Article'])
    pred[i]
    pred[i]['title'] = pre_process_text(pred[i]['title'].split(' '))
#pred[0]['title']    


# In[13]:





# ## Getting ticker/ signal data
# Below we will be getting the data to predict the data model.  First, as mentioned, we need to scrape for the article date.  I want to be able to tell in a 5 day span whether or not an article is good or bad.  Then based on the date, we can get a few signals based on the change in bar.  We will be looking for the variance in the market value that is:  the first market value of opening day minus the last market value of the closing day, minus the initial value.  
# 
# Additionally, we can extrapolate the values instead of a magnitude, a simple buy/sell signal.  The former is much more precisise and could be highly valueable in production if results are acceptable.  The latter is much more common for sentiment analysis, and would be more likely to produce reliable results.  

# In[ ]:


def get_datetime(url):
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Parse the HTML content of the page using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the article timestamp
        timestamp_element = soup.find('div', class_='caas-attr-time-style').find('time')

        if timestamp_element is not None:
            # Extract the timestamp text
            timestamp = timestamp_element['datetime'].split('T')[0]
            return timestamp
        else:
            return "Timestamp not found."
    except Exception as e:
        return f"Error occurred: {e}"


# In[18]:


# Gets a signal if the stock went up or down in 1 or 7 days
def get_signal(symbol, article_date, offset):
    ad = datetime.strptime(article_date, '%Y-%m-%d')
    barset = yf.download(symbol, start=ad, end=ad+ timedelta(days= offset), progress=False)
    return (barset['Open'][0]-barset['Close'][-1]) / barset['Open'][0]


# In[ ]:


def buy_sell(val):
    #Just need to see if a number is positive or negative: 
    if(val> 0):
#        return "Buy" 
        return 1
    if(val< 0):
#        return "Sell" 
        return 0 
    if(val == 0):
#        return "Hold"
        return 0


# Below we sample some time frames as responses to the article.  The origin date is when the article was published, the window length is the window that the market had to respond ot it. 
# 
# For this project we are lookinng at a 5 day timeframe.  If an article was dropped on a weekday, the prediction will be skipped.  

# In[16]:


root = 'https://finance.yahoo.com'
resp = []
for i in yahoo_finance_news:
    date = get_datetime(root + i['link'])
    if len(date) == 10: #Sloppy, because python does not lazy evaluate, and the second condition could cause an error.  
        if datetime.strptime(get_datetime(root + i['link']), '%Y-%m-%d').weekday()  <5:  #Skipping weekdays.  Could look to next day.  
            resp.append({'link' : i['link'], "1 day": get_signal(i['Symbol'], date, 1),
                         "1D Call":buy_sell(get_signal(i['Symbol'], date, 1)),  
                         "2 day": get_signal(i['Symbol'], date, 2),"2D Call":buy_sell(get_signal(i['Symbol'], date, 2)),  
                         "3 day": get_signal(i['Symbol'], date, 3),"3D Call":buy_sell(get_signal(i['Symbol'], date, 3)),  
                         "4 day": get_signal(i['Symbol'], date, 4),"4D Call":buy_sell(get_signal(i['Symbol'], date, 4)), 
                         "5 day": get_signal(i['Symbol'], date, 5), "5D Call":buy_sell(get_signal(i['Symbol'], date, 5))})


# * Sample of the response list

# In[17]:


resp[:1]


# ## Assemble the data in a format that can run the data
# Below we join and convert the data into a single dataframe that is easy to read for both the eyes and the upcoming model.  

# In[ ]:


print("TST") 
print(pd.DataFrame(pred).columns)
print(pd.DataFrame(resp).columns)

df= pd.merge(pd.DataFrame(pred), pd.DataFrame(resp), on='link', how = 'outer')
df = df.dropna(subset = "1 day")
df


# In[ ]:


raw = []
for i in df.index:
    tk = df['Article'][i] 
    rawtext = ""
    for j in tk:
        rawtext += j + " "
    raw.append(rawtext)    


# # Building the models
# As illuded to above, we want to experiment with different timeframes for when news affectuates into the market.  Each will reqire a different test/train set.  These are easy to set in scale, or manually.  For simplicity we will be adding 5 manually (less processing to build inline functions than recurring objects).  Sticking with the convention of splitting train and test data sets with an 80/20.  

# In[ ]:


y1 = df["1 day"]
y2 = df["2 day"]
y3 = df["3 day"]
y4 = df["4 day"]
y5 = df["5 day"]
y1_train, y1_test = train_test_split(y1, test_size=0.2, random_state=42)
y2_train, y2_test = train_test_split(y2, test_size=0.2, random_state=42)
y3_train, y3_test = train_test_split(y3, test_size=0.2, random_state=42)
y4_train, y4_test = train_test_split(y4, test_size=0.2, random_state=42)
y5_train, y5_test = train_test_split(y5, test_size=0.2, random_state=42)


# ## Vectorizing the article 
# Below, we need to condition the predictor values for model evaluation.  For this roject we will be vectorizing the articles using Term Frequency - Inverse Documentation Frequency.  This transformer is prime for variable document lengths (blogs could be one or many paragraphs).  Additionally, they will be able to pick out spicific and important words like "Cyber Attack", "Growth", "Beats", or "Dissapoints".  As you can see the final output is an array of floats ranging from 0 to 1.  A few instances are in fact 0, which are likely 'useless' words.  

# In[ ]:


#X = df['Raw'] 
X = df['Article']
X_train_pre, X_test_pre = train_test_split(X, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(analyzer=lambda x: x)
X_train = vectorizer.fit_transform(X_train_pre)
X_train
X_test = vectorizer.transform(X_test_pre)
X_test.toarray()[0][:50]#Sample


# ## Adding the titles 
# It may be advantagous to include the title, there may be precious words embedded within the title.  

# In[ ]:


Xt = df['title']+ df['Article']
Xt_train_pre, Xt_test_pre = train_test_split(Xt, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(analyzer=lambda x: x)
Xt_train = vectorizer.fit_transform(Xt_train_pre)
Xt_train
Xt_test = vectorizer.transform(Xt_test_pre)


# ## Scale the different regressors
# Running multiple regressions takes up time and space.  It is best practice to build a single function that can run and analyze all of the functions.  This allows for simplicity and experimenation, particulary because I am running a total of 5 different responses.  Below, the model will be trained and evaluated.  

# In[ ]:


def scale_regression(model_type, days = 5,_X_train =X_train,_X_test = X_test,  _y_train_list = [y1_train, y2_train, y3_train, y4_train, y5_train], y_test_list = [y1_test, y2_test, y3_test, y4_test, y5_test]):
#Pre: You are giving the function a model type with .fit
#Post: You get the predictions as well as the accuracy.  
    y_pred_list = [] 
    rmse_list = []
    for i in range(0,days-1): 
        model_type.fit(_X_train, _y_train_list[i])
        y_pred_temp = model_type.predict(_X_test)
        y_pred_list.append(y_pred_temp)
        rmse_temp = np.sqrt(mean_squared_error(y_test_list[i], y_pred_temp))
        rmse_list.append(rmse_temp)
        print(i+1, " Day:")
        print("Pred:  ", y_pred_temp[i])
        print("Actual:", sum(y_test_list[i])/ len(y_test_list[i]))
        print("RMSE:  ", rmse_list[i])
        print("MSE:   ", rmse_list[i] ** 2)
    return rmse_list


# ## Implement the regressors
# As you can see, even though I am using best practices and state of the art methods, I am still a long way from quitting my day job.  Most of the models appear to be unreliable and have more error than response.  It appears that the linear regression did the best thus far.  Linear regression is preferred because it adds an element of simplicity to a highly complex process and by far the least likely to cause any overfitting.  It would be wise to 'tighten up' the data model by fixing the hyper parameters.  

# In[ ]:


# Neural Network preprocessing
nn = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Input shape is determined by the number of features
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
])

# Compile the model
nn.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Convert the sparse matrix to a numpy array for training
X_train_np = X_train.toarray()
X_test_np = X_test.toarray()

vocab_size = 10000  # Example value, adjust according to your data
max_len = 100  # Example value, adjust according to your data

# Define the LSTM model
lstm = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len),
    LSTM(units=64, dropout=0.2, recurrent_dropout=0.2),
    Dense(units=1, activation='sigmoid')
])

# Compile the model
lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


#Linear Regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
print("Linear Regression")
t1  = scale_regression(lr)
print("Best Model: ", "Day ", 1+ t1.index(min(t1)), "RMSE:", min(t1))

#Decision Tree
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
print("\nDecision Tree")
t1 = scale_regression(dt)
print("Best Model: ", "Day ", 1+ t1.index(min(t1)), "RMSE:", min(t1))

#Random forest
from sklearn.ensemble import RandomForestRegressor
rfg = RandomForestRegressor()
print("\nRandom Forest")
t1 = scale_regression(rfg)
print("Best Model: ", "Day ", 1+ t1.index(min(t1)), "RMSE:", min(t1))

#SVR
from sklearn.svm import SVR
svr = SVR()
print("\nSRV")
t1 = scale_regression(svr)
print("Best Model: ", "Day ", 1+ t1.index(min(t1)), "RMSE:", min(t1))

#Neural Network
print("\n Nerual Network")
t1 = scale_regression(nn, _X_train= X_train_np, _X_test= X_test_np, _y_train_list = [y1_train, y2_train, y3_train, y4_train, y5_train], y_test_list = [y1_test,  y2_test, y3_test, y4_test, y5_test])
print("Best Model: ", "Day ", 1+ t1.index(min(t1)), "RMSE:", min(t1))

#LSTM
print("\n LSTM")
t1 = scale_regression(lstm, _X_train= X_train_np, _X_test= X_test_np, _y_train_list = [y1_train, y2_train, y3_train, y4_train, y5_train], y_test_list = [y1_test,  y2_test, y3_test, y4_test, y5_test])


# ## Implement the articles and titles for the possiblity of results.  
# It appears that adding the articles in fact helped!  This will be the ideal lead for looking for the best model.  

# In[ ]:


print("Linear Regression")
t1 = scale_regression(lr, _X_train= Xt_train, _X_test= Xt_test)
print("Best Model: ", "Day ", 1+ t1.index(min(t1)), "RMSE:", min(t1))

print("\nDecision Tree")
t1 = scale_regression(dt,  _X_train= Xt_train, _X_test= Xt_test)
print("Best Model: ", "Day ", 1+ t1.index(min(t1)), "RMSE:", min(t1))

print("\nRandom Forest")
t1 = scale_regression(rfg, _X_train= Xt_train, _X_test= Xt_test)
print("Best Model: ", "Day ", 1+ t1.index(min(t1)), "RMSE:", min(t1))

print("\nSRV")
t1 = scale_regression(svr, _X_train= Xt_train, _X_test= Xt_test)
print("Best Model: ", "Day ", 1+ t1.index(min(t1)), "RMSE:", min(t1))


# ## Tune hyper parameters
# As illuded above, we must look for the best combination of variables to run thought the tree model.  It however appears that the inital model was a better fit, the tuning appears to have overtrained the model

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Define the parameter grid
param_dist = {
    'criterion': ['mse', 'mae'],
    'splitter': ['best', 'random'],
    'max_depth': [None] + list(range(5, 50, 5)),
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2', None],
    'random_state': [42]
}

# Create a Decision Tree Regressor model
dt_model = DecisionTreeRegressor()

# Use RandomizedSearchCV to tune hyperparameters
random_search = RandomizedSearchCV(estimator=dt_model, param_distributions=param_dist, n_iter=100, cv=5, verbose=0, random_state=42, n_jobs=-1)

# Fit the model
random_search.fit(X_train, y3_train)

# Print the best parameters found
print("Best parameters found:", random_search.best_params_)

# Get the best model
best_dt_model = random_search.best_estimator_

# Evaluate the best model
test_score = best_dt_model.score(X_test, y3_test)
print("Test Score:", test_score)


# In[ ]:


y_best_pred = random_search.predict(X_test)
best_mse = mean_squared_error(y3_test, y_best_pred)
print("Best RMSE:", np.sqrt(best_mse))
print((sum(y3_test)/len(y3_test)))



# In[ ]:




