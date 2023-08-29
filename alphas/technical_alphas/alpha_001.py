from quantlib.database import utils as data_utils
from quantlib.alphas import technical_utils 
from quantlib.alphas import alpha_utils 
import quant_tools
import pandas as pd
import numpy as np
import copy

'''
Formula: 

Hypothesis:
'''

class alpha_001:

    def __init__(self, db_data=None, returns_data=None):
        """
        Args:
            db_data (dict, optional): Database data. Defaults to None.
            returns_data (pd.DataFrame, optional): Stock-level returns. Defaults to None.
        """
        
        self.ma_pair = (20, 60)
        self.db_path = r'C:\Users\marcu\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.7_qbz5n2kfra8p0\LocalCache\local-packages\Python37\site-packages\quantlib\database\cache\russell_3000\russell_3000_cache.pickle'
        self.returns_path = r'C:\Users\marcu\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.7_qbz5n2kfra8p0\LocalCache\local-packages\Python37\site-packages\quantlib\database\cache\russell_3000\adj_close_returns.pickle'
        self.start_date = '19991231'
        self.vol_target = 0.20
        
        # If researching alpha in isolation, load cached data. Otherwise pass it via alpha builder abstract class.
        if (db_data == None) or (returns_data is None):
            self.db_data, self.returns_data = self.get_data()
        
        self.run()

    def get_data(self):
        """
        Loads cached database data and returns data.

        Returns:
            tuple: Database data and stock-level returns.
        """
        db_data = data_utils.load_cache(self.db_path)
        returns_data = pd.DataFrame(data_utils.load_cache(self.returns_path))  
        return db_data, returns_data 
    
    def run(self):
        """
        1) Generates intertemporal raw alpha signals for each instrument.
        2) Generates indicator positions (1, 0, -1) for each instrument based on the alpha signal.
        3) Conducts asset-level equal risk allocation volatilty targeting.
        4) Generates unconstrained simulated returns.
        5) Aggregates key alpha model portfolio statistics. 
        """

        # Initialize objects
        self.raw_signal = pd.DataFrame()
        self.ex_ante_vol = pd.DataFrame()

        # Get raw alpha signal for each instrument in universe
        for i, (ticker, tmp_data) in enumerate(self.db_data.items()):

            print(f"Getting formulaic alpha for {ticker}")
            
            # Compute fast_ewma - slow_ewma
            tmp_data[f'ewma({str(self.ma_pair[0])})'] = technical_utils.ewma_series(tmp_data['adj_close'], n=self.ma_pair[0])
            tmp_data[f'ewma({str(self.ma_pair[1])})'] = technical_utils.ewma_series(tmp_data['adj_close'], n=self.ma_pair[1])
            tmp_data[f'ewma({str(self.ma_pair[0])}_{str(self.ma_pair[1])})'] = tmp_data[f'ewma({str(self.ma_pair[0])})'] - tmp_data[f'ewma({str(self.ma_pair[1])})']

            # Get raw alpha signal 
            raw_signal = tmp_data[f'ewma({str(self.ma_pair[0])}_{str(self.ma_pair[1])})'].rename(ticker)

            # Drop signals on untradeable days
            drop_signal_indices = tmp_data['actively_traded'].where(tmp_data['actively_traded'] == False).dropna().index
            raw_signal.loc[drop_signal_indices] = 0

            # Update raw alpha signal
            self.raw_signal = pd.concat([self.raw_signal, raw_signal], axis=1).sort_index()

            # Get ex-ante vol (default to 40% annualized vol) -- preferably import from a pre-computed risk model
            default_vol = 0.40 / np.sqrt(252)
            self.ex_ante_vol = pd.concat([self.ex_ante_vol, tmp_data['adj_close_returns'].rolling(20).std().rename(ticker).fillna(default_vol)], axis=1).sort_index()

            # Update instrument's data
            self.db_data[ticker] = tmp_data

            if i > 20:
                break

        # Get binary votes from alpha singal (here this is long only)
        self.votes = self.raw_signal.mask(self.raw_signal > 0, 1).mask(self.raw_signal <= 0, 0)
        
        # Get signal conviction -- once you have a multi-strategy system, you can create a signal for each instrument that generates dynamic conviction levels in each instrument
        # alpha_data['signal_strength'] = alpha_data['votes'].apply(lambda x: np.sum(x) / len(x.dropna()), axis=1)
        
        # Asset level vol targeting (equal risk allocation)
        daily_vol_target = self.vol_target / np.sqrt(252)
        self.positions_vol_target = self.votes.apply(lambda x: np.abs(x) * daily_vol_target if isinstance(x, float) else np.abs(x) * daily_vol_target / np.sum(np.abs(x)), axis=1)   

        # Asset level vol scalars 
        vol_scalars = self.positions_vol_target / self.ex_ante_vol

        # Nomial positions
        self.positions = self.votes * vol_scalars

        # Proportional weights (not nominal positions)
        self.w = self.positions / np.abs(self.positions).sum(axis=1)
       
        # Gross notional value (leverage)
        self.gnv = np.abs(self.positions).sum(axis=1)

        # Summarize outlier positions
        indices, tickers = np.where(self.positions > self.positions.quantile(.999).quantile(.999))
        outlier_positions = self.positions.values[indices, tickers]
        outlier_tickers = self.positions.columns[tickers]
        outlier_indices = self.positions.index[indices]
        self.outlier_positions = pd.DataFrame({'ticker': outlier_tickers, 'positions': outlier_positions}, index=outlier_indices).sort_index()

        # From here, incorporate alpha/strategy level vol scaling as a function of realized volatilty. This would leverage a vol
        # modeling algorithm to estimate ex-ante vol of the portfolio, then scale all positions based on the proportion to target strategy vol and
        # ex-ante vol. This separates vol targeting between the asset and strategy level. A more refined rendition on this is to 
        # create a risk model that accounts for covariance matrix to capture cross-asset dynamics.

        # Get alpha model returns
        self.instrument_level_alpha_model_returns = (self.positions * self.returns_data.shift(-1)[self.positions.columns]).iloc[:-1]
        self.alpha_model_returns = self.instrument_level_alpha_model_returns.sum(axis=1).rename('alpha_model_returns')

        # Number of alpha model views per instrument
        self.n_views = np.abs(self.votes).sum(axis=0).rename('n_views')

        # Naive volatility decomposition (i.e., this is not intertemporal risk contribution which depends on covariance matrix... just an intrinsic volatility decomposition)
        self.instrument_level_alpha_model_mean_vol = (self.instrument_level_alpha_model_returns.std() * 252 ** .5).rename('instrument_level_alpha_model_mean_vol')
        self.volatility_attribution = (self.instrument_level_alpha_model_mean_vol / self.instrument_level_alpha_model_mean_vol.sum()).rename('volatility_attribution')

        # Naive performance attribution (returns should be distributed evenly if the alpha captures diversifying effects)
        self.instrument_level_alpha_model_mean_return = self.instrument_level_alpha_model_returns.mean().rename('instrument_level_alpha_model_mean_return')
        self.performance_attribution = (self.instrument_level_alpha_model_mean_return / self.instrument_level_alpha_model_mean_return.sum()).rename('performance_attribution')

        # Scale by square root of the NOBS to capture statistical significance of each instruments' returns
        obs_scalars = np.sqrt(self.n_views)
        self.adjusted_performance_attribution = ((self.instrument_level_alpha_model_mean_return * obs_scalars) / (self.instrument_level_alpha_model_mean_return * obs_scalars).sum()).rename('adjusted_performance_attribution')

        # Decompose cumulative gains
        final_cumulative_returns = ((1 + self.instrument_level_alpha_model_returns).cumprod() - 1).iloc[-1]
        self.cumulative_performance_attribution = (final_cumulative_returns / final_cumulative_returns.sum()).rename('cumulative_performance_attribution')

        return