# Momentum, which scores the stocks based in long term performance (e.g. last year returns) with the expectation that the stocks keep performing accordingly
# Mean reversion, which scores the stocks by the inverse of recent performance (e.g. last 5 days returns) with the expectation that the stocks prices revert to a rolling mean
# Volatility, which scores the stocks by the inverse of standard deviation of returns
# Size, which scores small-cap stocks higher than high-cap stocks
# Fundamental Analysis in general (Value, Quality, etc. )
# Technical Analysis in general (RSI, BollingerBands, Aroon, FastStochasticOscillator, IchimokuKinkoHyo, TrueRange etc.)
# Sentiment factors
# 极窄基波动率、肥尾波动率 （beta）

# alpha:左侧，趋势，投机
# beta:右侧，价值，投资

# Is the factor predictive at all?
# How much is predictive the factor?
# Is the factor consistent across the the full stocks universe?
# How many assets should I trade in the long leg? and on the short one?
# What weighting scheme should I use: factor weighting or equal weighting?
# What holding period should I use? Should I rebalance every day or every week?
# What is the turnover for different holding periods?
# Does the factor performs well across all sectors? Can we trade the factor in a sector neutral strategy?
# How does the factor perform across different level of volatility, market cap, asset price, momentum, book-to-price ratios, Beta exposure etc. ?

class Alphas(object):
    def __init__(self, df_data):

        self.open = df_data['S_DQ_OPEN'] 
        self.high = df_data['S_DQ_HIGH'] 
        self.low = df_data['S_DQ_LOW']   
        self.close = df_data['S_DQ_CLOSE'] 
        self.volume = df_data['S_DQ_VOLUME']*100 
        self.returns = df_data['S_DQ_PCTCHANGE'] 
        self.vwap = (df_data['S_DQ_AMOUNT']*1000)/(df_data['S_DQ_VOLUME']*100+1) 
        
    
    # Alpha#5	 (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))
    def alpha005(self):
        return  (rank((self.open - (sum(self.vwap, 10) / 10))) * (-1 * abs(rank((self.close - self.vwap)))))
    