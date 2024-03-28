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


# 假装 UTC 就是 Asia/Shanghai (简化计算)，所有datetime默认 tz-naive -> tz-aware
TZ = "UTC"
data_now = datetime.now().strftime('%Y-%m-%d')
START = '1900-01-01'
END = data_now


data_sel = 'dummy' # dummy/SSE
if data_sel == 'SSE':
    # real SSE data
    DATAFEED = bt.feeds.PandasData
    sids = [1]
elif data_sel == 'dummy':
    # fast dummy data
    modpath = os.path.dirname(os.path.abspath(__file__))
    dataspath = './datas'
    datafiles = [
        #'2006-01-02-volume-min-001.txt',
        '2006-day-001.txt',
        #'2006-week-001.txt',
    ]


class Strategy(bt.Strategy):
    params = (
        ('datalines', False),  # Print data lines
        ('lendetails', False), # Print individual items memory usage
    )

    def __init__(self):
        datas = [self.data.close,] #(self.data.close+self.data.open)/2
        
    def next(self):
        if self.p.datalines:
            txt = ','.join([
                '%04d' % len(self),
                '%04d' % len(self.data0),
                self.data.datetime.date(0).isoformat()
            ])

            print(txt)

    def loglendetails(self, msg):
        if self.p.lendetails:
            print(msg)

    def stop(self):
        super(Strategy, self).stop()

        tlen = 0
        self.loglendetails('-- Evaluating Datas')
        for i, data in enumerate(self.datas):
            tdata = 0
            for line in data.lines:
                tdata += len(line.array)
                tline = len(line.array)

            tlen += tdata
            logtxt = '---- Data {} Total Cells {} - Cells per Line {}'
            self.loglendetails(logtxt.format(i, tdata, tline))

        self.loglendetails('-- Evaluating Indicators')
        for i, ind in enumerate(self.getindicators()):
            tlen += self.rindicator(ind, i, 0)

        self.loglendetails('-- Evaluating Observers')
        logtxt = '---- Observer {} Total Cells {} - Cells per Line {}'
        for i, obs in enumerate(self.getobservers()):
            tobs = 0
            for line in obs.lines:
                tobs += len(line.array)
                tline = len(line.array)

            tlen += tdata
            self.loglendetails(logtxt.format(i, tobs, tline))

        print(f'Total memory cells used: {tlen}')

    def rindicator(self, ind, i, deep):
        tind = 0
        for line in ind.lines:
            tind += len(line.array)
            tline = len(line.array)

        thisind = tind

        tsub = sum(
            self.rindicator(sind, j, deep + 1)
            for j, sind in enumerate(ind.getindicators())
        )
        iname = ind.__class__.__name__.split('.')[-1]

        logtxt = '---- Indicator {}.{} {} Total Cells {} - Cells per line {}'
        self.loglendetails(logtxt.format(deep, i, iname, tind, tline))
        logtxt = '---- SubIndicators Total Cells {}'
        self.loglendetails(logtxt.format(deep, i, iname, tsub))

        return tind + tsub

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
def data_feed_dummy():
    datas = []
    for datafile in datafiles:
        datapath = os.path.join(modpath, f'{dataspath}/{datafile}')
        #datapath = os.path.join(modpath, './datas/nvda-1999-2014.txt')
        data = bt.feeds.YahooFinanceCSVData(
            dataname=datapath,
            # Do not pass values before this date
            fromdate=datetime(2000, 1, 1),
            # Do not pass values before this date
            todate=datetime(2000, 12, 31),
            # Do not pass values after this date
            reverse=False)
        datas.append(data)
    return datas

def data_feed_SSE():
        # real SSE data
    datas = []
    start_session = pytz.timezone(TZ).localize(datetime.strptime(START, '%Y-%m-%d'))
    end_session = pytz.timezone(TZ).localize(datetime.strptime(END, '%Y-%m-%d'))
    # trade_days, special_trade_days, special_holiday_days = parse_csv_tradedate()
    metadata, index_info = parse_csv_metadata() # index_info = [asset_csv_path, num_lines]
    symbol_map = metadata.loc[:,['symbol','asset_name','first_traded']]
    print(metadata.iloc[0,:3])
        # split:除权, merge:填权, dividend:除息
        # 用了后复权数据，不需要adjast factor
        # parse_csv_split_merge_dividend(symbol_map, start_session, end_session)
        # (Date) * (Open, High, Low, Close, Volume, OpenInterest)
    for kline in parse_csv_kline_d1(symbol_map, index_info, start_session, end_session, sids):
        data = DATAFEED(dataname=kline)
        datas.append(data)
    return datas

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='This is the helper function')
    parser.add_argument('--plot', required=False, action='store_true',
                        help=('Plot the result'))
    return parser.parse_args()


if __name__ == '__main__':
    if data_sel == 'SSE':
        datas = data_feed_SSE()
    elif data_sel == 'dummy':
        datas = data_feed_dummy()

    runtest(datas=datas,strategy=Strategy)