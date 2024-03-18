# run 'pip install ~/work/trade/backtrader' to update locally-modified package

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import argparse

import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pandas as pd
import pytz
from datetime import datetime

from NO2_csv_data_parse import parse_csv_tradedate, parse_csv_metadata, parse_csv_kline_d1

# 假装 UTC 就是 Asia/Shanghai (简化计算)，所有datetime默认 tz-naive -> tz-aware
tz = "UTC"
data_start = '1900-01-01'
data_end = datetime.now().strftime('%Y-%m-%d')

class TestInd(bt.Indicator):
    lines = ('a', 'b')

    def __init__(self):
        self.lines.a = b = self.data.close - self.data.high
        self.lines.b = btind.SMA(b, period=20)


class Strategy(bt.Strategy):
    params = (
        ('datalines', False),  # Print data lines
        ('lendetails', False), # Print individual items memory usage
    )

    def __init__(self):
        btind.SMA()
        btind.Stochastic()
        btind.RSI()
        btind.MACD()
        btind.CCI()
        TestInd().plotinfo.plot = False

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

    cerebro = bt.Cerebro()

    # data-feeds
    start_session = pytz.timezone(tz).localize(datetime.strptime(data_start, '%Y-%m-%d'))
    end_session = pytz.timezone(tz).localize(datetime.strptime(data_end, '%Y-%m-%d'))
    trade_days, special_trade_days, special_holiday_days = parse_csv_tradedate()
    metadata, index_info = parse_csv_metadata() # index_info = [asset_csv_path, num_lines]
    symbol_map = metadata.loc[:,['symbol','asset_name','first_traded']]
    print(metadata.iloc[:,:3].tail(1))
    # split:除权, merge:填权, dividend:除息
    # 用了后复权数据，不需要adjast factor
    # parse_csv_split_merge_dividend(symbol_map, start_session, end_session)
    
    sid = 10
    # (Date) * (Open, High, Low, Close, Volume, OpenInterest)
    kline = parse_csv_kline_d1(symbol_map, index_info, start_session, end_session, sid)

    data = bt.feeds.PandasData(dataname=kline)

    cerebro.adddata(data)
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