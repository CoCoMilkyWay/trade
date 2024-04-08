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
import random
random.seed(datetime.now().timestamp())
warnings.filterwarnings('ignore')

# append module root directory to sys.path
os.sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backtrader as bt
import backtrader.indicators as btind
from indicators import *

# =============================================================================================
# 假装 UTC 就是 Asia/Shanghai (简化计算)，所有datetime默认 tz-naive -> tz-aware
TZ = "UTC"
data_now = datetime.now().strftime('%Y-%m-%d')
START = '2020-07-09'
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
cheat_on_open = True # not necessarily cheating, especially for longer period like day bar
exectype = bt.Order.Market # use Market if set cheat_on_open
enable_log = True
plot = True
plot_data = False
analysis_factor = False
analysis_portfolio = False
# =============================================================================================
class Strategy(bt.Strategy):
    def __init__(self):
        if cheat_on_open:
            self.cheating = self.cerebro.p.cheat_on_open
        self.orderid = None
        datas = [self.data.close,] #(self.data.close+self.data.open)/2
        
    def start(self):
        self.broker.setcommission(commission=0.000, mult=1.0, margin=0.0)

    def next(self):
        if not self.cheating:
            self.operate()
    def next_open(self):
        if self.cheating:
            self.operate()
    def operate(self):
        # executed after 1st bar close, but still in the 1st timestamp
        # in the 2nd timestamp, the order is executed
        # use cerebro = bt.Cerebro(cheat_on_open=True) at day-bar level, it makes sense
        # next_open, nextstart_open and prenext_open
        portfolio_value = self.broker.get_value()
        portfolio_cash = self.broker.get_cash()
        slipage = 0
        for data in self.datas:
            self.orderid = self.close(
                data=data,
                size=self.getposition(data=data).size,
                exectype=exectype)
        for data in self.datas:
            size = round(portfolio_value/NO_SID/(data.open[0] * (1+slipage)))
            print(data.open[0])
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
                self.log('BUY EXECUTED, Price: %.2f, Cash-: %.1f(%.1f shares), Comm %.2f' %(
                    order.executed.price,
                    order.executed.value,
                    order.executed.size,
                    order.executed.comm))
                print(self.broker.get_cash())
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cash+: %.1f( %.1f shares), pnl:%.2f %%, Comm %.2f' %(
                    order.executed.price,
                    -1*order.executed.price * order.executed.size,
                    order.executed.size,
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
    
    # not work for ipynb
    # args = parse_args()
    
    cerebro = bt.Cerebro(
        runonce=runonce,# runonce: indicator in vectorized mode 
        preload=preload,# preload: preload datafeed for strategy(strategy/observer always in event mode)
        maxcpus=maxcpus,
        exactbars=exbar,# exbars:   1: deactivate preload/runonce/plot
        stdstats=True,
        cheat_on_open=True,
    )
    if isinstance(datas, bt.LineSeries):
        datas = [datas]
    for data in datas:
        data.plotinfo.plot = plot_data
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
        from _2_bt_misc import analyze_portfolio
        analyze_portfolio(results)
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

def main():
    from _2_bt_misc import data_feed_dummy, data_feed_SSE
    if data_sel == 'SSE':
        datas = data_feed_SSE()
    elif data_sel == 'dummy':
        datas = data_feed_dummy()
    runtest(datas=datas,strategy=Strategy,plot=plot)

if __name__ == '__main__':
    main()