import os
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime
from datetime import date
from dateutil.relativedelta import relativedelta


# Load .env into environment variables
load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
client = StockHistoricalDataClient(API_KEY, API_SECRET)
# Define request
def get_data_today(symbol = 'mmm', gran='day', start =1 , startgran="years"):
    mapping = {
        "month":TimeFrame.Month,
        "week":TimeFrame.Week,
        "minute": TimeFrame.Minute,
        "hour": TimeFrame.Hour,
        "day": TimeFrame.Day
    }
    ##THIS IS ThE HEART and likely what will be maintained
    request = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=mapping.get(gran.lower(), None),#TimeFrame.Day,
        start=datetime.combine(date.today()- relativedelta(**{startgran:start}), datetime.min.time()),
#        start=datetime.combine(date.today()- relativedelta(s=start), datetime.min.time()),
        end=datetime.combine(date.today(), datetime.min.time())
    )

    # Fetch data
    bars = client.get_stock_bars(request)
    df = bars.df  # Convert to pandas DataFrame
    df = df.reset_index()
    # As you can see the simple import process allows for us to get an overall view of the daily market trade.  The date of the market instance is indexed making it easy to carry these variables without loosing the time coefficient.  For the aformentioned emphesis on simplicity we are going to focus on the market close price.  
    df = df.rename(columns = {'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume', 'trade_count':'Trade Count'})
    return df