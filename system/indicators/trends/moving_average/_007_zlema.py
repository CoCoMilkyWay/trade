#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


from ...basicops import MovingAverageBase, MovAv


class ZeroLagExponentialMovingAverage(MovingAverageBase):
    '''
    零延迟均线(period, _movav=EMA)
    通过从单位中减去高通滤波器的传递响应来创建。
    由于高通滤波器在低频时的幅度非常小, 因此产生的低频滞后差值只是单位滞后, 即0
    采用同样alpha参数的瞬时趋势线和EMA两条平均值的平滑程度大致相同
    The zero-lag exponential moving average (ZLEMA) is a variation of the EMA
    which adds a momentum term aiming to reduce lag in the average so as to
    track current prices more closely. by John Ehlers and Ric Way

    Formula:
      - lag = (period - 1) / 2
      - zlema = ema(2 * data - data(-lag))

    See also:
      - https://en.wikipedia.org/wiki/Zero_lag_exponential_moving_average
      - http://user42.tuxfamily.org/chart/manual/Zero_002dLag-Exponential-Moving-Average.html

    '''
    alias = ('ZLEMA', 'ZeroLagEma',)
    lines = ('zlema',)
    params = (('_movav', MovAv.EMA),)

    def __init__(self):
        lag = (self.p.period - 1) // 2
        data = 2 * self.data - self.data(-lag)
        self.lines.zlema = self.p._movav(data, period=self.p.period)

        super(ZeroLagExponentialMovingAverage, self).__init__()
