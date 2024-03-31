# run 'pip install ~/work/trade/backtrader' to update locally-modified package

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import warnings
from datetime import datetime
import pytz
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# append module root directory to sys.path
os.sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backtrader as bt
import backtrader.indicators as btind
from indicators import *


# =============================================================================================
# 假装 UTC 就是 Asia/Shanghai (简化计算)，所有datetime默认 tz-naive -> tz-aware
TZ = "UTC"
data_now = datetime.now().strftime('%Y-%m-%d')
START = '1900-01-01'
END = data_now
# in/out sample partition
oos = pd.Timestamp(END) - pd.Timedelta('30D') # out-of-sample datetime
CASH = 500000.0

data_sel = 'SSE' # dummy/SSE
if data_sel == 'SSE':
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
    sids = [random.randint(0, 568) for _ in range(round(569*0.01))]
    NO_SID = len(sids)
elif data_sel == 'dummy':
    # fast dummy data
    modpath = os.path.dirname(os.path.abspath(__file__))
    dataspath = './datas'
    datafiles = [
        #'2006-01-02-volume-min-001.txt',
        'nvda-1999-2014.txt',
        #'2006-week-001.txt',
    ]

# Market (default), Close, Limit, Stop, StopLimit
# market open usually has higher slippage due to high external volume
exectype = bt.Order.Close
enable_log = False
plot = True
analysis_factor = False
analysis_portfolio = True

# =============================================================================================
class Strategy(bt.Strategy):
    def __init__(self):
        self.orderid = None
        datas = [self.data.close,] #(self.data.close+self.data.open)/2
        
    def start(self):
        self.broker.setcommission(commission=0.000, mult=1.0, margin=0.0)
        
    def next(self): 
        # executed after 1st bar close, but still in the 1st timestamp
        # in the 2nd timestamp, the order is executed
        portfolio_value = self.broker.get_value()
        portfolio_cash = self.broker.get_cash()
        slipage = 0
        for data in self.datas:
            self.orderid = self.close(
                data=data,
                exectype=exectype)
        for data in self.datas:
            size = round(portfolio_value/NO_SID/(data.close[0] * (1+slipage)))
            self.orderid = self.buy(
                data=data,
                size = size,
                exectype=exectype)            

    def stop(self):
        from _2_bt_misc import print_data_size
        super(Strategy, self).stop()
        print_data_size(self)

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
                self.log('BUY EXECUTED, Price: %.2f, Cash-: %.1f(%.1f), Comm %.2f' %(
                    order.executed.price,
                    order.executed.value,
                    order.executed.size,
                    order.executed.comm))
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cash+: %.1f( %.2f), Comm %.2f' %(
                    order.executed.price,
                    -1*order.executed.price * order.executed.size,
                    order.executed.pnl/order.executed.value*100,
                    order.executed.comm))
        # Sentinel to None: new orders allowed
        self.order = None
        
def runtest(datas,
            strategy,
            runonce=None,
            preload=None,
            exbar=0,
            plot=False,
            optimize=False,
            maxcpus=None,
            writer=None,
            analyzer=None,
            **kwargs):
    
    args = parse_args()

    cerebro = bt.Cerebro(
        runonce=runonce,# runonce: indicator in vectorized mode 
        preload=preload,# preload: preload datafeed for strategy(strategy/observer always in event mode)
        maxcpus=maxcpus,
        exactbars=exbar,# exbars:   1: deactivate preload/runonce/plot
        stdstats=True,
    )
    if isinstance(datas, bt.LineSeries):
        datas = [datas]
    for data in datas:
        data.plotinfo.plot = False
        cerebro.adddata(data)
        #cerebro.resampledata(data, timeframe=bt.TimeFrame.Weeks)
    if not optimize:
        cerebro_idx = cerebro.addstrategy(strategy, **kwargs)
        if writer:
            wr = writer[0]
            wrkwargs = writer[1]
            cerebro.addwriter(wr, **wrkwargs)
        if analyzer:
            al = analyzer[0]
            alkwargs = analyzer[1]
            cerebro.addanalyzer(al, **alkwargs)
    else:
        cerebro_idx = cerebro.optstrategy(strategy, **kwargs)
    cerebro.addsizer_byidx(cerebro_idx, bt.sizers.FixedSize)
    cerebro.broker.setcash(CASH)
    if analysis_portfolio:
        cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
    results = cerebro.run()
    if analysis_portfolio:
        strat = results[0]
        pyfoliozer = strat.analyzers.getbyname('pyfolio')
        pf_returns, pf_positions, pf_transactions, gross_lev = pyfoliozer.get_pf_items()
        pf_benchmark = pf_returns
        print(type(pf_returns))
        import _001_pyfolio as pf
        from _001_pyfolio.utils import extract_rets_pos_txn_from_zipline
        from _001_pyfolio.plotting import (
            plot_perf_stats,
            show_perf_stats,
            plot_rolling_beta,
            plot_rolling_returns,
            plot_rolling_sharpe,
            plot_drawdown_periods,
            plot_drawdown_underwater)
        # pf.create_full_tear_sheet(
        #     returns,
        #     positions=positions,
        #     transactions=transactions,
        #     gross_lev=gross_lev,
        #     live_start_date=START,
        #     round_trips=True)
        pf.tears.create_full_tear_sheet(
            pf_returns,
            positions=pf_positions,
            transactions=pf_transactions,
            benchmark_rets=pf_benchmark, # factor-universe-mean-daily-return (index benchmark) / daily-return of a particular asset
            hide_positions=True
            )
        fig, ax_heatmap = plt.subplots(figsize=(15, 8))
        sns.heatmap(pf_positions.replace(0, np.nan).dropna(how='all', axis=1).T, 
        cmap=sns.diverging_palette(h_neg=20, h_pos=200), ax=ax_heatmap, center=0)

        # special requirements :(
        data_intersec = pf_returns.index & pf_benchmark.index
        pf_returns = pf_returns.loc[data_intersec]
        pf_positions = pf_positions.loc[data_intersec]
        fig, ax_perf = plt.subplots(figsize=(15, 8))
        plot_perf_stats(returns=pf_returns, 
                        factor_returns=pf_benchmark,     
                        ax=ax_perf)
        show_perf_stats(returns=pf_returns, 
                        factor_returns=pf_benchmark, 
                        positions=pf_positions, 
                        transactions=pf_transactions, 
                        live_start_date=oos)
        fig, ax_rolling = plt.subplots(figsize=(15, 8))
        plot_rolling_returns(
            returns=pf_returns, 
            factor_returns=pf_benchmark, 
            live_start_date=oos, 
            cone_std=(1.0, 1.5, 2.0),
            ax=ax_rolling)
        plt.gcf().set_size_inches(14, 8)
    if plot:
        # plotstyle for OHLC bars: line/bar/candle (on close)
        cerebro.plot(style='bar')
    return [cerebro]

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='This is the helper function')
    parser.add_argument('--plot', required=False, action='store_true',
                        help=('Plot the result'))
    return parser.parse_args()

if __name__ == '__main__':
    from _2_bt_misc import data_feed_dummy, data_feed_SSE

    if data_sel == 'SSE':
        datas = data_feed_SSE()
    elif data_sel == 'dummy':
        datas = data_feed_dummy()

    runtest(datas=datas,strategy=Strategy,plot=plot)