# run 'pip install ~/work/trade/backtrader' to update locally-modified package

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import warnings
from datetime import datetime
import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import math
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
random.seed(datetime.now().timestamp())
warnings.filterwarnings('ignore')

# append module root directory to sys.path
os.sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backtrader as bt
import backtrader.indicators as btind
print(bt.__file__)

# =============================================================================================
# 假装 UTC 就是 Asia/Shanghai (简化计算)，所有datetime默认 tz-naive -> tz-aware
TZ = "UTC"
date_now = datetime.now().strftime('%Y-%m-%d')
START = '2005-01-01'
END = date_now # '2010-08-1'
# start_day = datetime.strptime(START, '%Y-%m-%d')
# end_day = datetime.strptime(END, '%Y-%m-%d')
# days = (end_day-start_day).days

# in/out sample partition
oos = pd.Timestamp(END) - pd.Timedelta('30D') # out-of-sample datetime
CASH = 1000000.0
data_sel = 'SSE' # dummy/SSE/index
if data_sel == 'dummy':
    # fast dummy data
    modpath = os.path.dirname(os.path.abspath(__file__))
    dataspath = './datas'
    datafiles = [
        #'2006-01-02-volume-min-001.txt',
        'nvda-1999-2014.txt',
        #'2006-week-001.txt',
    ]
elif data_sel == 'SSE':
    # real SSE data
    DATAFEED = bt.feeds.PandasData
    assets_list = [
        #'1沪A_不包括科创板', # 1698
        #'2深A_不包括创业板', # 1505
        '3科创板', # 569
        #'4创业板', # 1338
        #'5北A_新老三板', # 244
        #'6上证股指期权',
        #'7深证股指期权',
    ]
    NO_SID = 50
    def stock_pool(): # this should only be init-ed once
        sids = random.sample(range(569+1), NO_SID)
        print(sids)
        return sids
elif data_sel == 'index':
    index = ['sh.000300']
    DATAFEED = bt.feeds.PandasData

plot_observer = True # add a broker / default observers (cash/value; trades; orders)
plot_variable = False
plot_assets = False
plot_volume = False
analysis_factor = False
analysis_portfolio = False
hist_plot = False
cpus = None # None=all cores

# not necessarily cheating, especially for longer period like day bar
cheat_on_open = False
# market open usually has higher slippage due to high external volume
cheat_on_close = True
# Market (default), Close, Limit, Stop, StopLimit
exectype = bt.Order.Market # use Market if set cheat_on_xxxx
enable_log = False
ind_stats = False
# select among 'normal(single strat)', 'multi_strat(customer analyzer)' mode
mode = 'single_strat' # single_strat/multi_strat
if mode=='single_strat': # run single strategy with analysis
    plot = True
    plot_assets = False # plot assets price/buy/sells
    plot_volume = False
    hist_plot = False
    num_bins = 50
    min_edge = 0.5
    max_edge = 1.5
    analysis_factor = False
    analysis_portfolio = False
    analysis_backtrader = True
    optimize = False
    Strats = ['MA=[15]']
elif mode=='multi_strat': # use opt to run multiple strategies
    plot = True
    analysis_backtrader = True
    optimize = True
    optimize_method = bt.analyzers.Portfolio_Value
    optimize_results = 'analyzers.portfolio_value.get_analysis'
    Strats = [
        # MA=[period0]
        # BB=[period0,std]
        # RS=[period0,std,period1,no_fitting_bias]
        'ref=1', # buy at start and hold (for reference)
        #'MA=[15]',
        #'MA=[20]', # :)
        #'MA=[30]',
        #'MA=[60]',
        #'BB=[10, 2]',
        #'BB=[15, 2]', # :)
        #'BB=[20, 2]',

        'RS=[10, 0.7, 200, 0]',
        'RS=[15, 0.7, 200, 0]',
        'RS=[10, 0.7, 200, 1]',
        'RS=[15, 0.7, 200, 1]',
    ]
    cpus = None # BT bug: canonical indicator is not supported in multi-strat mode

class CFG:
    def __init__(self, globals_dict):
        self.__dict__.update(globals_dict)
cfg = CFG(globals())
# =============================================================================================
class variable_observer(bt.observer.Observer):
    alias = ('variables',)
    lines = ('ind0','ind1','ind2','ind3')
    
    plotinfo = dict(plot=True, subplot=True)
    # when accessing other method, make sure they exist as time of access
    def next(self): # executed after 'next' method
        # self._owner = Strategy
        if self._owner.type=='MA':
            self.lines.ind0[0] = self._owner.sma
        elif self._owner.type=='BB':
            self.lines.ind0[0] = self._owner.upper_band
            self.lines.ind1[0] = self._owner.lower_band
        elif self._owner.type=='RS':
            self.lines.ind0[0] = self._owner.rsrs
            self.lines.ind1[0] = self._owner.rsrs_upper
            self.lines.ind2[0] = self._owner.rsrs_lower
            self.lines.ind3[0] = self._owner.rsrs_r2
        pass
    
class Strategy(bt.Strategy):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            #setattr(self, key, value)
            print(key, value)
            if key == 'MA':
                self.type = 'MA'
                self.period0 = value[0]
            elif key == 'BB':
                self.type = 'BB'
                self.period0 = value[0]
                self.std = value[1]
            elif key == 'RS':
                self.type = 'RS'
                self.period0 = value[0]
                self.std = value[1]
                self.period1 = value[2]
                self.no_fitting_bias = value[3]
            elif key == 'ref':
                self.type = 'ref'
                self.period0 = 1
        self.iter = 0
        if cheat_on_open:
            self.cheating = self.cerebro.p.cheat_on_open
        if cheat_on_close:
            self.cheating = self.cerebro.p.cheat_on_close
        self.orderid = None
        self.days = 0
        self.dtlast = 0

        # indicators (canonical)
        # for data in self.datas:
        #     data.sma1 = btind.SimpleMovingAverage(data.close, period=20)
        #     data.buy_sig =  bt.And(data.close < data.sma1)
        #     data.sell_sig = bt.And(data.close > data.sma1)
        # custome indicators (non-canonical)
        
        self.upper_band = 0
        self.lower_band = 0
        self.sma = 0
        self.rsrs = 0
        self.df_rsrs = [0]
        self.rsrs_zs = 0
        self.rsrs_upper = 0
        self.rsrs_lower = 0
        self.rsrs_r2 = 0
        
        self.buy_sig = 0
        self.sell_sig = 0

        # self.datas = [self.data.close,] #(self.data.close+self.data.open)/2
        if(hist_plot):
            self.bin_edges = np.linspace(min_edge, max_edge, num_bins + 1)
            self.bin_histo = [0] * num_bins
            self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
            self.bin_colors = [0] * num_bins
    def start(self):
        self.broker.setcommission(commission=0.000, mult=1.0, margin=0.0)
    def next(self):
        if not self.cheating:
            self.operate()
    def next_open(self): # avaliable only at day-bar level
        if self.cerebro.p.cheat_on_open:
            self.operate()
    def next_close(self): # avaliable only at day-bar level
        if self.cerebro.p.cheat_on_close:
            self.operate()
    def dtime_dt(self, dt):
        return math.trunc(dt)
    def dtime_tm(self, dt):
        return math.modf(dt)[0]
    def _daycount(self):
        dt = self.dtime_dt(self.data.datetime[0])
        if dt > self.dtlast:
            self.days += 1
            self.dtlast = dt
    def OLS_fit(self, y, x): # @day bar lvl, wls performs very well
        # OLS, WLS, GLS work on functionals:
        #       y = a + b*f1 + c*f2 + residue, 
        #       solve for a,b,c that minimize residue
        # (functional)basis of the model
        X = np.column_stack((np.ones(len(x)), x))
        # Fit OLS model
        model = sm.OLS(y, X).fit()
        # endog(y, dependent variable)
        # exog(x, independent variable)
        # Calculate "adjusted R-square" to measure 
        r_squared = model.rsquared # model explains (how much) variability
        adjusted_r_squared = model.rsquared_adj # (better)goodness of fit
        
        # if(r_squared<0.6):# see OLS effect
        #     _, iv_l, iv_u = wls_prediction_std(model)
        #     plt.plot(x, y, 'o', label="data")
        #     plt.plot(x, model.fittedvalues, 'r--.', label="OLS")
        #     plt.plot(x, iv_u, 'r--')
        #     plt.plot(x, iv_l, 'r--')
        #     plt.legend(loc='best')
        #     print(model.params)
        return model.params, adjusted_r_squared
    def operate(self):
        self._daycount()
        
        # portfolio management
        portfolio_value = self.broker.get_value() # last close value (share only) when called
        portfolio_cash = self.broker.get_cash() # cash at last close when called
        total_value_now = portfolio_cash
        def current_price(data):
            return data.close[0]
        for data in self.datas:
            total_value_now += self.getposition(data=data).size*current_price(data)
        
        # for every asset
        size = self.period0
        for data in self.datas:
            df_open = np.array(data.open.get(ago=0, size=size), dtype=float)
            df_high = np.array(data.high.get(ago=0, size=size), dtype=float)
            df_low = np.array(data.low.get(ago=0, size=size), dtype=float)
            df_close = np.array(data.close.get(ago=0, size=size), dtype=float)
            # high[1:]/high[:-1]
            if(self.days<size):# skip N days preparing indicator
                break
            
            # indicators(non-canonical)
            if self.type=='MA':
                self.sma = df_close.mean()
                self.buy_sig =  data.close[0] > self.sma
                self.sell_sig = data.close[0] < self.sma
            if self.type=='BB':
                self.middle_band = df_close.mean()
                self.band_std = df_close.std()*self.std
                self.upper_band = self.middle_band + self.band_std
                self.lower_band = self.middle_band - self.band_std
                self.buy_sig =  data.close[0] > self.upper_band
                self.sell_sig = data.close[0] < self.lower_band
            if self.type=='RS':
                [_, self.rsrs], self.rsrs_r2 = self.OLS_fit(df_high,df_low)
                self.df_rsrs.append(self.rsrs)
                rolling_rsrs = self.df_rsrs[-self.period1:]
                self.rsrs_mean = np.mean(rolling_rsrs)
                self.rsrs_std = np.std(rolling_rsrs)
                # self.rsrs_upper = self.rsrs_mean + self.rsrs_std*self.std
                # self.rsrs_lower = self.rsrs_mean - self.rsrs_std*self.std
                
                # Standard Z-score: (cumulative pdf for normal dist)
                # +-1: 84.1%/15.9%
                # +-2: 97.7%/2.3%
                # +-3: 99.9%/0.1%
                self.rsrs_zs = (self.rsrs - self.rsrs_mean)/self.rsrs_std
                if self.no_fitting_bias==1:
                    self.rsrs_zs = self.rsrs_zs * self.rsrs_r2
                # adjust zs using r2 (filter out less certain signals)
                self.buy_sig =  self.rsrs_zs > self.std
                self.sell_sig = self.rsrs_zs < -self.std
            
            # execution
            if self.type=='ref':
                self.orderid = self.buy(
                data=data,
                size=math.floor(portfolio_cash/data.close),
                exectype=exectype)
            else:
                if self.buy_sig:
                    self.orderid = self.buy(
                        data=data,
                        size=math.floor(portfolio_cash/data.close),
                        exectype=exectype)
                if self.sell_sig:
                    self.orderid = self.close(
                        data=data,
                        size=self.getposition(data=data).size,
                        exectype=exectype)
            
            # analysis
            if(hist_plot):
                for i in range(num_bins):
                    if self.bin_edges[i] <= self.rsrs < self.bin_edges[i + 1]:
                        self.bin_histo[i] += 1
                        self.bin_colors[i] += self.rsrs_r2
                        break
    def stop(self):
        super(Strategy, self).stop()
        if ind_stats:
            from _2_bt_misc import stat_depict
            stat_depict(self.df_rsrs)
        if enable_log:
            from _2_bt_misc import print_data_size
            print(f'{self.days} days simulated')
            print_data_size(self)
        if hist_plot:
            self.hist_plot()

    def hist_plot(self):
        colors = []
        upper_bound = 1
        lower_bound = 0
        bound_thd = 0.8
        for i, center in enumerate(self.bin_centers):
            if(self.bin_histo[i]==0):
                colors.append(mcolors.to_rgba('white'))
                continue
            R_sqr = self.bin_colors[i]/self.bin_histo[i]
            if R_sqr > bound_thd:
                alpha = min(1.0, (R_sqr - bound_thd) / (upper_bound-bound_thd))
                color = mcolors.to_rgba('red', alpha=alpha)
            else:
                alpha = min(1.0, (bound_thd - R_sqr) / (bound_thd-lower_bound))
                color = mcolors.to_rgba('blue', alpha=alpha)
            colors.append(color)
        for i in range(num_bins):
            plt.bar(self.bin_centers[i], self.bin_histo[i], width=self.bin_edges[i+1] - self.bin_edges[i], color=colors[i], edgecolor='black')
        plt.xlabel('Bins(RSRS)')
        plt.ylabel('Frequency')
        plt.title('RSRS(Red=better fit)')
        plt.grid(True)
        plt.show()

    def log(self, txt, dt=None, nodate=False):
        if not enable_log:
            return
        if not nodate:
            dt = dt or self.data.datetime[0]
            dt = bt.num2date(dt)
            print(f'{dt.isoformat()}, {txt}')
        else:
            print(f'---------- {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            # self.log('ORDER ACCEPTED/SUBMITTED', dt=order.created.dt)
            self.order = order
            return
        if order.status in [order.Expired]:
            self.log('BUY EXPIRED')
        elif order.status in [order.Completed]:
            if order.isbuy():
                self.log('BUY EXECUTED, Price: %.2f, Cash-: %.1f(%.0f shares), Comm %.2f' %(
                    order.executed.price,
                    order.executed.value,
                    order.executed.size,
                    order.executed.comm))
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cash+: %.1f( %.0f shares), pnl:%.2f %%, Comm %.2f' %(
                    order.executed.price,
                    -1*order.executed.price * order.executed.size,
                    order.executed.size,
                    order.executed.pnl/order.executed.value*100,
                    order.executed.comm))
        # Sentinel to None: new orders allowed
        self.order = None

import ast
class StFetcher(object):
    _STRATS = []
    for item in Strats:
        _STRATS.append(Strategy)
    def __new__(cls, *args, **kwargs):
        idx = kwargs.pop('idx') if kwargs.pop('multi') else 0
        param_name, value = Strats[idx].split('=')
        value = ast.literal_eval(value)
        param = {param_name: value}
        return cls._STRATS[idx](*args, **kwargs, **param)

def runtest(datas,
            strategy,
            runonce=None,
            preload=None,
            exbar=0,
            plot=False,
            optimize=optimize,
            maxcpus=cpus,
            writer=None,
            analyzer=None,
            **kwargs):

    # not work for ipynb
    # args = parse_args()

    cerebro = bt.Cerebro(
        runonce=runonce,# runonce: indicator in vectorized mode 
        preload=preload,# preload: preload datafeed for strategy(strategy/observer always in event mode)
        maxcpus=maxcpus,
        exactbars=exbar,# exbars:   1: deactivate preload/runonce/plot
        stdstats=plot_observer,
        cheat_on_open=cheat_on_open,
        cheat_on_close=cheat_on_close,
    )
    # if isinstance(datas, bt.LineSeries):
    #     datas = [datas]
    for data in datas:
        data.plotinfo.plot = plot_assets
        cerebro.adddata(data)
        #cerebro.resampledata(data, timeframe=bt.TimeFrame.Weeks)

    # execution order: 'indicator' -> 'next' -> 'observer'
    if plot_variable:
        cerebro.addobserver(variable_observer)
    if optimize: # multi-strat
        cerebro_idx = cerebro.optstrategy(StFetcher, idx=[i for i, _ in enumerate(Strats)], multi=True)
    else:
        cerebro_idx = cerebro.addstrategy(StFetcher, multi=False)

    cerebro.addsizer_byidx(cerebro_idx, bt.sizers.FixedSize)
    cerebro.broker.setcash(CASH)
    # if writer:
    #     wr = writer[0]
    #     wrkwargs = writer[1]
    #     cerebro.addwriter(wr, **wrkwargs)
    # if analyzer:
    #     al = analyzer[0]
    #     alkwargs = analyzer[1]
    #     cerebro.addanalyzer(al, **alkwargs)

    if optimize: # multi-strat
        cerebro.addanalyzer(optimize_method)
    if analysis_portfolio: # single-strat
        cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
    if analysis_backtrader: # single-strat
        cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='AnnualReturn')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='Returns')
        # # cerebro.addanalyzer(bt.analyzers.LogReturnsRolling, _name='LogReturnsRolling')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='SharpeRatio')
        cerebro.addanalyzer(bt.analyzers.VWR, _name='VWR') # https://www.crystalbull.com/sharpe-ratio-better-with-log-returns/
        cerebro.addanalyzer(bt.analyzers.SQN, _name='SQN')
        # # cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='TradeAnalyzer') # show trade details
        # # cerebro.addanalyzer(bt.analyzers.Transactions, _name='Transactions')
        # # cerebro.addanalyzer(bt.analyzers.Calmar, _name='Calmar')
        if mode=='single_strat':
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='DrawDown')
        # # cerebro.addanalyzer(bt.analyzers.GrossLeverage, _name='GrossLeverage')

    results = cerebro.run()

    results = [result[0] for result in results] if optimize else [results[0]] # flatten results
    
    if analysis_portfolio: # single-strat
        from _2_bt_misc import analyze_portfolio
        analyze_portfolio(results, cfg)
    if analysis_backtrader: # single-strat
        from _2_bt_misc import print_autodict_data, print_multi_dict
        print_multi_dict(results, 'AnnualReturn', cfg)
        print_multi_dict(results, 'Returns', cfg)
        # # print_multi_dict(results, 'LogReturnsRolling', cfg)
        print_multi_dict(results, 'SharpeRatio', cfg)
        print_multi_dict(results, 'VWR', cfg)
        print_multi_dict(results, 'SQN', cfg)
        # # print_autodict_data(results[0].analyzers.TradeAnalyzer.get_analysis()) # show trade details
        # # print_multi_dict(results, 'Transactions', cfg)
        # # print_multi_dict(results, 'Calmar', cfg)
        if mode=='single_strat':
            print_autodict_data(results[0].analyzers.DrawDown.get_analysis(), 'DrawDown Analysis: ========= ')
        # # print_multi_dict(results, 'GrossLeverage', cfg)
        
    if optimize: # multi-strat
        # if plot:
        #     fig, axs = plt.subplots(2, figsize=(10, 10))
        for i, strat in enumerate(results):
            # debug: print method/data of a class obj
            # for attribute in dir(strat.analyzers):
            #     if callable(getattr(strat.analyzers, attribute)):
            #         print(attribute)
            #     else:
            #         print(attribute)
            attributes = optimize_results.split('.')
            rets = strat
            for attr in attributes:
                rets = getattr(rets, attr)
            rets = rets() # call on the final method
            print(
                f'Strat {i} Name {strat.__class__.__name__}:\n  - analyzer: {type(rets)}\n'
            )

            if plot:
                x, y = list(rets[0].keys()), list(rets[0].values())
                plt.plot(x, y, label=f'{Strats[i]}')
                plt.text(x[-1], y[-1], f'{Strats[i]}', fontsize=8, in_layout=True)
                #axs[0].plot(list(rets[0].keys()), list(rets[0].values()),label=f'{Strats[i]}')
                #axs[1].plot(list(rets[1].keys()), list(rets[1].values()),label=f'{Strats[i]}')
        if plot:
            plt.title('multi-strat')
            plt.legend()
            # axs[0].set_title('multi-strat')
            # axs[1].set_title('daily-return')
            # for ax in axs.flat:
            #     ax.set_xlabel('keys')
            #     ax.set_ylabel('value')
            #     ax.legend()
            plt.grid()
            plt.show()
    elif plot:
        # plotstyle for OHLC bars: line/bar/candle (on close)
        cerebro.plot(style='bar', volume=plot_volume)
    return [cerebro]

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='This is the helper function')
    parser.add_argument('--plot', required=False, action='store_true',
                        help=('Plot the result'))
    return parser.parse_args()

def main():
    from _2_bt_misc import data_feed_dummy, data_feed_SSE, data_feed_index
    if data_sel == 'dummy':
        datas = data_feed_dummy(cfg)
    elif data_sel == 'SSE':
        datas = data_feed_SSE(cfg)
    elif data_sel == 'index':
        datas = data_feed_index(cfg)
    runtest(datas=datas,strategy=Strategy,plot=plot) # param1=0,...
    
if __name__ == '__main__':
    main()
