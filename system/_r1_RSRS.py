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
import math
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
date_now = datetime.now().strftime('%Y-%m-%d')
START = '2020-07-09'
END = date_now
# in/out sample partition
oos = pd.Timestamp(END) - pd.Timedelta('30D') # out-of-sample datetime
CASH = 500000.0
data_sel = 'dummy' # dummy/SSE
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
    NO_SID = 5
    def stock_pool(): # this should only be init-ed once
        sids = random.sample(range(569+1), NO_SID)
        print(sids)
        return sids
elif data_sel == 'dummy':
    # fast dummy data
    modpath = os.path.dirname(os.path.abspath(__file__))
    dataspath = './datas'
    datafiles = [
        #'2006-01-02-volume-min-001.txt',
        'nvda-1999-2014.txt',
        #'2006-week-001.txt',
    ]
class CFG:
    def __init__(self, globals_dict):
        self.__dict__.update(globals_dict)
cfg = CFG(globals())

# not necessarily cheating, especially for longer period like day bar
cheat_on_open = False
# market open usually has higher slippage due to high external volume
cheat_on_close = True
# Market (default), Close, Limit, Stop, StopLimit
exectype = bt.Order.Market # use Market if set cheat_on_xxxx
enable_log = True
plot = True # plot default observers (portfolio)
plot_assets = False # plot assets price/buy/sells
analysis_factor = False
analysis_portfolio = False

# =============================================================================================
class Strategy(bt.Strategy):
    def __init__(self):
        if cheat_on_open:
            self.cheating = self.cerebro.p.cheat_on_open
        if cheat_on_close:
            self.cheating = self.cerebro.p.cheat_on_close
        self.orderid = None
        datas = [self.data.close,] #(self.data.close+self.data.open)/2

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
    def operate(self):
        portfolio_value = self.broker.get_value() # last close value (share only) when called
        portfolio_cash = self.broker.get_cash() # cash at last close when called
        total_value_now = portfolio_cash
        def current_price(data):
            return data.close[0]
        for data in self.datas:
            total_value_now += self.getposition(data=data).size*current_price(data)
        for data in self.datas:
            # print('open today: ', data.open[0],' close today: ', data.close[0])
            self.orderid = self.close(
                data=data,
                size=self.getposition(data=data).size,
                exectype=exectype)
        for data in self.datas:
            size = math.floor(total_value_now)
            self.orderid = self.buy(
                data=data,
                # size=portfolio_cash,
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
        cheat_on_open=cheat_on_open,
        cheat_on_close=cheat_on_close,
    )
    # if isinstance(datas, bt.LineSeries):
    #     datas = [datas]
    for data in datas:
        data.plotinfo.plot = plot_assets
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
        analyze_portfolio(results, cfg)
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
        datas = data_feed_SSE(cfg)
    elif data_sel == 'dummy':
        datas = data_feed_dummy(cfg)
    runtest(datas=datas,strategy=Strategy,plot=plot)

if __name__ == '__main__':
    main()
