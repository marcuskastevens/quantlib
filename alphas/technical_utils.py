import pandas_ta as talib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

def ewma_series(series, n):
    return series.ewm(span=n).mean()

def sma_series(series, n):
    return series.rolling(n).mean()

def adx_series(high, low, close, n):
    return talib.adx(high, low, close, length=n)
