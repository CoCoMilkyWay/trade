# Momentum, which scores the stocks based in long term performance (e.g. last year returns) with the expectation that the stocks keep performing accordingly
# Mean reversion, which scores the stocks by the inverse of recent performance (e.g. last 5 days returns) with the expectation that the stocks prices revert to a rolling mean
# Volatility, which scores the stocks by the inverse of standard deviation of returns
# Size, which scores small-cap stocks higher than high-cap stocks
# Fundamental Analysis in general (Value, Quality, etc. )
# Technical Analysis in general (RSI, MACD, Bollinger Bands, etc.)
# Sentiment factors
# 极窄基波动率、肥尾波动率 （beta）

# alpha:左侧，趋势，投机
# beta:右侧，价值，投资


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
    