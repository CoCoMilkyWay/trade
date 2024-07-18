#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from ..basicops import Indicator, MovAv
from backtrader.functions import Max, Min

class TrueHigh(Indicator):
    '''
    Defined by J. Welles Wilder, Jr. in 1978 in his book *"New Concepts in
    Technical Trading Systems"* for the ATR

    Records the "true high" which is the maximum of today's high and
    yesterday's close

    Formula:
      - truehigh = max(high, close_prev)

    See:
      - http://en.wikipedia.org/wiki/Average_true_range
    '''
    lines = ('truehigh',)

    def __init__(self):
        self.lines.truehigh = Max(self.data.high, self.data.close(-1))
        super(TrueHigh, self).__init__()


class TrueLow(Indicator):
    '''
    Defined by J. Welles Wilder, Jr. in 1978 in his book *"New Concepts in
    Technical Trading Systems"* for the ATR

    Records the "true low" which is the minimum of today's low and
    yesterday's close

    Formula:
      - truelow = min(low, close_prev)

    See:
      - http://en.wikipedia.org/wiki/Average_true_range
    '''
    lines = ('truelow',)

    def __init__(self):
        self.lines.truelow = Min(self.data.low, self.data.close(-1))
        super(TrueLow, self).__init__()


class TrueRange(Indicator):
    '''
    Defined by J. Welles Wilder, Jr. in 1978 in his book New Concepts in
    Technical Trading Systems.

    Formula:
      - max(high - low, abs(high - prev_close), abs(prev_close - low)

      which can be simplified to

      - max(high, prev_close) - min(low, prev_close)

    See:
      - http://en.wikipedia.org/wiki/Average_true_range

    The idea is to take the previous close into account to calculate the range
    if it yields a larger range than the daily range (High - Low)
    '''
    alias = ('TR',)

    lines = ('tr',)

    def __init__(self):
        self.lines.tr = TrueHigh(self.data) - TrueLow(self.data)
        super(TrueRange, self).__init__()


class AverageTrueRange(Indicator):
    '''
    真实波幅(period=14,movav=Smoothed)
    根据市场波动性调整其风险敞口,设置止盈止损,判断真假突破:
    高ATR值可能表明市场处于较高波动性状态,可能不适宜进入市场。
    (波动性较高的市场或资产,高的ATR值可能是正常的)
    低ATR值可能表明市场相对稳定,有利于交易。

    波动性的增加通常伴随着趋势的加强。ATR的增加可能意味着趋势持续,而ATR的减少可能预示着趋势的减弱或反转。

    Defined by J. Welles Wilder, Jr. in 1978 in his book *"New Concepts in
    Technical Trading Systems"*.

    The idea is to take the close into account to calculate the range if it
    yields a larger range than the daily range (High - Low)

    Formula:
      - SmoothedMovingAverage(TrueRange, period)

    See:
      - http://en.wikipedia.org/wiki/Average_true_range
    '''
    alias = ('ATR',)

    lines = ('atr',)
    params = (('period', 14), ('movav', MovAv.Smoothed))

    def _plotlabel(self):
        plabels = [self.p.period]
        plabels += [self.p.movav] * self.p.notdefault('movav')
        return plabels

    def __init__(self):
        self.lines.atr = self.p.movav(TR(self.data), period=self.p.period)
        super(AverageTrueRange, self).__init__()
