from quantlib.database import utils

def run():
    """
    Runs data-injestion, pre-processing, and caching process for a user-specified universe.
    """

    # Get user-specified universe
    universe = input()

    if universe == "R3K":
        
        # Define constants
        R3K_CACHE_PATH = r'C:\Users\marcu\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.7_qbz5n2kfra8p0\LocalCache\local-packages\Python37\site-packages\quantlib\database\cache\russell_3000\russell_3000_cache.pickle'
        CROSS_UNVIERSE_CACHE_PATH = r'C:\Users\marcu\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.7_qbz5n2kfra8p0\LocalCache\local-packages\Python37\site-packages\quantlib\database\cache\russell_3000\\'
        R3K_TICKER_PATH = r'C:\Users\marcu\Documents\Quant\Programming\quant_data\russell_300_returns.pickle'
        
        # Get list of stock tickers
        ticker_list = utils.load_cache(path=R3K_TICKER_PATH).columns
        
        # Get OHLC + returns data
        ohlc_data = utils.get_ohlc_data(ticker_list)
        
        # Ensure data was acquire for each ticker
        try:
            assert list(ohlc_data.keys()) == ticker_list
        except:
            for key in ohlc_data.keys():
                if key not in ticker_list:
                    print(key)
            
        # Cache data to database
        utils.cache(data=ohlc_data, path=R3K_CACHE_PATH)

        # Cache in cross-universe manner
        utils.cache_cross_universe_statistics(ohlc_data=ohlc_data, path=CROSS_UNVIERSE_CACHE_PATH)

    elif universe == 'S&P 500':
        
        # Define constants
        SP_500_CACHE_PATH = r'C:\Users\marcu\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.7_qbz5n2kfra8p0\LocalCache\local-packages\Python37\site-packages\quantlib\database\cache\sp_500\sp_500_cache.pickle'
        CROSS_UNVIERSE_CACHE_PATH = r'C:\Users\marcu\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.7_qbz5n2kfra8p0\LocalCache\local-packages\Python37\site-packages\quantlib\database\cache\sp_500\\'
    
        # Get list of stock tickers
        ticker_list = utils.get_sp500_tickers()
        
        # Get OHLC + returns data
        ohlc_data = utils.get_ohlc_data(ticker_list)
        
        # Ensure data was acquire for each ticker
        try:
            assert list(ohlc_data.keys()) == ticker_list
        except:
            for key in ohlc_data.keys():
                if key not in ticker_list:
                    print(key)
            
        # Cache data to database
        utils.cache(data=ohlc_data, path=SP_500_CACHE_PATH)

        # Cache in cross-universe manner
        utils.cache_cross_universe_statistics(ohlc_data=ohlc_data, path=CROSS_UNVIERSE_CACHE_PATH)

    return

if __name__ == '__main__':

    run()