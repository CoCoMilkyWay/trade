#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from ..basicops import Indicator, MovAv
from ..basicops_lvl2 import MeanDev
from backtrader.functions import Max


class CommodityChannelIndex(Indicator):
    '''
    商品路径指标/顺势指标(period=N,factor=0.015,movav=Simple,upperband=100,lowerband=100)
    研判短线反弹的顶点和短线回调的底部拐点,适用短期内暴涨暴跌的非常态行情
    CCI指标却是波动于正无穷大到负无穷大之间,因此不会出现指标钝化现象
    强调价格与固定期间的股价平均区间的偏离程度，股价平均绝对偏差
    Introduced by Donald Lambert in 1980 to measure variations of the
    "typical price" (see below) from its mean to identify extremes and
    reversals

    Formula:
      - tp = typical_price = (high + low + close) / 3
      - tpmean = MovingAverage(tp, period)
      - deviation = tp - tpmean
      - meandev = MeanDeviation(tp)
      - cci = deviation / (meandeviation * factor)

    See:
      - https://en.wikipedia.org/wiki/Commodity_channel_index
    '''
    alias = ('CCI',)

    lines = ('cci',)

    params = (('period', 20),
              ('factor', 0.015),
              ('movav', MovAv.Simple),
              ('upperband', 100.0),
              ('lowerband', -100.0),)

    def _plotlabel(self):
        plabels = [self.p.period, self.p.factor]
        plabels += [self.p.movav] * self.p.notdefault('movav')
        return plabels

    def _plotinit(self):
        self.plotinfo.plotyhlines = [0.0, self.p.upperband, self.p.lowerband]

    def __init__(self):
        tp = (self.data.high + self.data.low + self.data.close) / 3.0
        tpmean = self.p.movav(tp, period=self.p.period)

        dev = tp - tpmean
        meandev = MeanDev(tp, tpmean, period=self.p.period)

        self.lines.cci = dev / (self.p.factor * meandev)

        super(CommodityChannelIndex, self).__init__()
