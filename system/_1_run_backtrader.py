# run 'pip install ~/work/trade/backtrader' to update locally-modified package

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import argparse
import os
import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pandas as pd
from datetime import datetime

from _2_csv_data_parse import parse_csv_tradedate, parse_csv_metadata, parse_csv_kline_d1

from indicators import *

# 假装 UTC 就是 Asia/Shanghai (简化计算)，所有datetime默认 tz-naive -> tz-aware
tz = "UTC"
data_start = '1900-01-01'
data_end = datetime.now().strftime('%Y-%m-%d')

#class TestInd(bt.Indicator):
#    lines = ('a', 'b')
#
#    def __init__(self):
#        self.lines.a = b = self.data.close - self.data.high
#        self.lines.b = btind.SMA(b, period=20)

class Strategy(bt.Strategy):
    params = (
        ('datalines', False),  # Print data lines
        ('lendetails', False), # Print individual items memory usage
    )

    def __init__(self):
        bit = False
        datas = [self.data.close,] #(self.data.close+self.data.open)/2
        for data in datas:
            if bit:
                # Moving Average (subplot=Flase)
                N=10
                SmoothedMovingAverage(data,period=N)
                MovingAverageSimple(data, period=N)
                ExponentialMovingAverage(data,period=N)
                WeightedMovingAverage(data,period=N)
                DoubleExponentialMovingAverage(data,period=N,_movav=MovAv.EMA)
                TripleExponentialMovingAverage(data,period=N,_movav=MovAv.EMA)
                KaufmanMovingAverage(data,period=N, slow=2, fast=30)
                FractalAdaptiveMovingAverage(data,period=N) # TODO
                VariableIndexDynamicAverage(data,period=N,short=0,long=0,smooth=0) # TODO
                ZeroLagIndicator(data,gainlimit=50,_movav=MovAv.EMA)
                ZeroLagExponentialMovingAverage(data,period=N,_movav=MovAv.EMA)
                HullMovingAverage(data,period=N,_movav=MovAv.WMA)
                DicksonMovingAverage(data,gainlimit=50, hperiod=N, _movav=MovAv.EMA, _hma=MovAv.HMA)
                JurikMovingAverage(data,period=N) # TODO

                # volatility(subplot=True)
                AverageTrueRange(period=N,movav=MovAv.Smoothed)
                BollingerBands(period=N,devfactor=2.,movav=MovAv.Simple)

                # momentum
                Aroon_axis = AroonUpDown()
                AroonOscillator(plotmaster = Aroon_axis)
                CommodityChannelIndex(period=N,factor=0.015,movav=MovAv.Simple,upperband=100,lowerband=100)

                # unclassified
                DetrendedPriceOscillator
            # test

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

def runstrat():
    args = parse_args()

    cerebro = bt.Cerebro(stdstats=False)

    # # data-feeds
    # sids = [1]
    # start_session = pytz.timezone(tz).localize(datetime.strptime(data_start, '%Y-%m-%d'))
    # end_session = pytz.timezone(tz).localize(datetime.strptime(data_end, '%Y-%m-%d'))
    # trade_days, special_trade_days, special_holiday_days = parse_csv_tradedate()
    # metadata, index_info = parse_csv_metadata() # index_info = [asset_csv_path, num_lines]
    # symbol_map = metadata.loc[:,['symbol','asset_name','first_traded']]
    # print(metadata.iloc[0,:3])
    # # split:除权, merge:填权, dividend:除息
    # # 用了后复权数据，不需要adjast factor
    # # parse_csv_split_merge_dividend(symbol_map, start_session, end_session)
    # # (Date) * (Open, High, Low, Close, Volume, OpenInterest)
    # for kline in parse_csv_kline_d1(symbol_map, index_info, start_session, end_session, sids):
    #     data = bt.feeds.PandasData(dataname=kline)
    #     cerebro.adddata(data)

    # Create a Data Feed
    modpath = os.path.dirname(os.path.abspath(os.sys.argv[0]))
    datapath = os.path.join(modpath, './datas/orcl-1995-2014.txt')
    data = bt.feeds.YahooFinanceCSVData(
        dataname=datapath,
        # Do not pass values before this date
        fromdate=datetime(2000, 1, 1),
        # Do not pass values before this date
        todate=datetime(2000, 12, 31),
        # Do not pass values after this date
        reverse=False)
    cerebro.adddata(data)
    datapath = os.path.join(modpath, './datas/nvda-1999-2014.txt')
    data = bt.feeds.YahooFinanceCSVData(
        dataname=datapath,
        # Do not pass values before this date
        fromdate=datetime(2000, 1, 1),
        # Do not pass values before this date
        todate=datetime(2000, 12, 31),
        # Do not pass values after this date
        reverse=False)
    cerebro.adddata(data)


    # datapath = os.path.join(modpath, './datas/2006-01-02-volume-min-001.txt')
    # data = bt.feeds.BacktraderCSVData(
    #     dataname=datapath,
    #     fromdate=datetime(2006, 1, 2),
    #     todate=datetime(2006, 2, 27),
    #     reverse=False
    #     )
    # cerebro.adddata(data)

    # cerebro.resampledata(data, timeframe=bt.TimeFrame.Weeks)

    cerebro.addstrategy(Strategy)

    cerebro.run(runonce=False, exactbars=0) # mem_save: [1, 0, -1, -2]
    cerebro.plot(style='bar')

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Check Memory Savings')
    # args.plot
    # parser.add_argument('--plot', required=False, action='store_true',
    #                     help=('Plot the result'))

    return parser.parse_args()


if __name__ == '__main__':
    runstrat()