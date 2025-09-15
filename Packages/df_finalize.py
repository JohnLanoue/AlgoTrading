def df_finalize(df ):
#   PRE: Recieves a data frame from the Alpaca_connection.get_data_today package.  Pandas is already installed.  
#   POST: Finalizes a data frame that is ready for next steps. 
#   Intent: Band-aid for data handling.  Consider closing the gap in get_data_today
#   Improvements: Replace function ALL together
    df = df.reset_index()
    df = df.rename(columns = {'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume', 'trade_count':'Trade Count'})
    return df