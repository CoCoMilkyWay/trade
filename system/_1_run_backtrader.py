# run 'pip install ~/work/trade/backtrader' to update locally-modified package

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
from datetime import datetime
import pytz
import argparse
import numpy as np
import pandas as pd

# append module root directory to sys.path
os.sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backtrader as bt
import backtrader.indicators as btind
from indicators import *

from _2_csv_data_parse import parse_csv_tradedate, parse_csv_metadata, parse_csv_kline_d1

# =============================================================================================
# 假装 UTC 就是 Asia/Shanghai (简化计算)，所有datetime默认 tz-naive -> tz-aware
TZ = "UTC"
data_now = datetime.now().strftime('%Y-%m-%d')
START = '1900-01-01'
END = data_now

data_sel = 'SSE' # dummy/SSE
if data_sel == 'SSE':
    # real SSE data
    DATAFEED = bt.feeds.PandasData
    sids = [0]
elif data_sel == 'dummy':
    # fast dummy data
    modpath = os.path.dirname(os.path.abspath(__file__))
    dataspath = './datas'
    datafiles = [
        #'2006-01-02-volume-min-001.txt',
        'nvda-1999-2014.txt',
        #'2006-week-001.txt',
    ]

# after a bar is closed, the order is executed as early as the second open price
# Market (default), Close, Limit, Stop, StopLimit
exectype = bt.Order.Close # market open usually has higher slippage due to high external volume
plot = False

# =============================================================================================
class Strategy(bt.Strategy):
    def __init__(self):
        self.orderid = None
        datas = [self.data.close,] #(self.data.close+self.data.open)/2
        
    def start(self):
        self.broker.setcommission(commission=0.0, mult=1.0, margin=0.0)
        
    def next(self):
        # if self.orderid:
        #     # if an order is active, no new orders are allowed
        #     return
        
        self.orderid = self.close(exectype=exectype)
        self.orderid = self.buy(exectype=exectype)
            

    def stop(self):
        from _2_bt_misc import print_data_size
        super(Strategy, self).stop()
        print_data_size(self)

    def log(self, txt, dt=None, nodate=False):
        if not nodate:
            dt = dt or self.data.datetime[0]
            dt = bt.num2date(dt)
            print(f'{dt.isoformat()}, {txt}')
        else:
            print(f'---------- {txt}')
        
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            self.log('ORDER ACCEPTED/SUBMITTED', dt=order.created.dt)
            self.order = order
            return
        if order.status in [order.Expired]:
            self.log('BUY EXPIRED')
        elif order.status in [order.Completed]:
            if order.isbuy():
                self.log('BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %(
                    order.executed.price,
                    order.executed.value,
                    order.executed.comm))
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %(
                    order.executed.price,
                    order.executed.value,
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
        exactbars=exbar)# exbars:   1: deactivate preload/runonce/plot
    if isinstance(datas, bt.LineSeries):
        datas = [datas]
    for data in datas:
        cerebro.adddata(data)
        #cerebro.resampledata(data, timeframe=bt.TimeFrame.Weeks)
    if not optimize:
        cerebro.addstrategy(strategy, **kwargs)
        if writer:
            wr = writer[0]
            wrkwargs = writer[1]
            cerebro.addwriter(wr, **wrkwargs)
        if analyzer:
            al = analyzer[0]
            alkwargs = analyzer[1]
            cerebro.addanalyzer(al, **alkwargs)
    else:
        cerebro.optstrategy(strategy, **kwargs)
    cerebro.run()
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