from quantlib.database import utils as data_utils
from quantlib.alphas import technical_utils 
from quantlib.alphas import alpha_utils 
import pandas as pd
import copy

'''
Formula: 

Hypothesis:
'''

class alpha_001:

    def __init__(self):
        
        # (fast, slow)
        self.ma_pairs = [(20, 60), (50, 200), (10, 20), (50, 80)]
        self.db_path = r'C:\Users\marcu\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.7_qbz5n2kfra8p0\LocalCache\local-packages\Python37\site-packages\quantlib\database\cache\russell_3000\russell_3000_cache.pickle'
        self.start_date = '19991231'
        self.vol_target = 0.20
        self.db_data = self.get_data(self.db_path) 
        self.alpha_data = self.alpha(self.db_data)
        

    def get_data(self, path):
        return data_utils.load_cache(path)
    
    def alpha(self, data):

        # Get alpha for each instrument in universe
        for ticker, tmp_data in data.items():

            print(f"Getting formulaic alpha for {ticker}")
            tmp_data['adx'] = technical_utils.adx_series(high=tmp_data['high'],
                                                         low=tmp_data['low'],
                                                         close=tmp_data['adj_close'],
                                                         n=20).iloc[:, 0]
            
            # Compute fast_ewma - slow_ewma
            for pair in self.ma_pairs:
                tmp_data[f'{str(pair[0])}_ewma'] = technical_utils.ewma_series(tmp_data['adj_close'], n=pair[0])
                tmp_data[f'{str(pair[1])}_ewma'] = technical_utils.ewma_series(tmp_data['adj_close'], n=pair[1])
                tmp_data[f'{str(pair[0])}_{str(pair[1])}_ewma'] = tmp_data[f'{str(pair[0])}_ewma'] - tmp_data[f'{str(pair[1])}_ewma']
            
            # Update instrument's data
            data[ticker] = tmp_data
        
        return data
            

    def run(self):
        
        portfolio = pd.DataFrame(index=self.alpha_data.loc[self.start_date:].index)

        pass