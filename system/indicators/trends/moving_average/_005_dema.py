#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


from ...basicops import Indicator, MovingAverageBase, MovAv


class DoubleExponentialMovingAverage(MovingAverageBase):
    '''
    双指数均线(period, _movav=EMA)
    DEMA was first time introduced in 1994, in the article "Smoothing Data with
    Faster Moving Averages" by Patrick G. Mulloy in "Technical Analysis of
    Stocks & Commodities" magazine.

    It attempts to reduce the inherent lag associated to Moving Averages

    Formula:
      - dema = (2.0 - ema(data, period) - ema(ema(data, period), period)

    See:
      (None)
    '''
    alias = ('DEMA', 'MovingAverageDoubleExponential',)

    lines = ('dema',)
    params = (('_movav', MovAv.EMA),)

    def __init__(self):
        ema = self.p._movav(self.data, period=self.p.period)
        ema2 = self.p._movav(ema, period=self.p.period)
        self.lines.dema = 2.0 * ema - ema2

        super(DoubleExponentialMovingAverage, self).__init__()


class TripleExponentialMovingAverage(MovingAverageBase):
    '''
    三指数均线(period, _movav=EMA)
    TEMA was first time introduced in 1994, in the article "Smoothing Data with
    Faster Moving Averages" by Patrick G. Mulloy in "Technical Analysis of
    Stocks & Commodities" magazine.

    It attempts to reduce the inherent lag associated to Moving Averages

    Formula:
      - ema1 = ema(data, period)
      - ema2 = ema(ema1, period)
      - ema3 = ema(ema2, period)
      - tema = 3 * ema1 - 3 * ema2 + ema3

    See:
      (None)
    '''
    alias = ('TEMA', 'MovingAverageTripleExponential',)

    lines = ('tema',)
    params = (('_movav', MovAv.EMA),)

    def __init__(self):
        ema1 = self.p._movav(self.data, period=self.p.period)
        ema2 = self.p._movav(ema1, period=self.p.period)
        ema3 = self.p._movav(ema2, period=self.p.period)

        self.lines.tema = 3.0 * ema1 - 3.0 * ema2 + ema3
        super(TripleExponentialMovingAverage, self).__init__()
