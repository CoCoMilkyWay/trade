#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


from ...basicops import MovingAverageBase, MovAv
from ._009_zlind import ZeroLagIndicator


class DicksonMovingAverage(MovingAverageBase):
    '''
    迪克森均线(gainlimit=50, hperiod=7, _movav=EMA, _hma=HMA)
    合并赫尔均线和零延时均线, 类似Jurik均线
    The *Dickson Moving Average* combines the ``ZeroLagIndicator`` (aka
    *ErrorCorrecting* or *EC*) by *Ehlers*, and the ``HullMovingAverage`` to
    try to deliver a result close to that of the *Jurik* Moving Averages
    By Nathan Dickson

    Formula:
      - ec = ZeroLagIndicator(period, gainlimit)
      - hma = HullMovingAverage(hperiod)

      - dma = (ec + hma) / 2

      - The default moving average for the *ZeroLagIndicator* is EMA, but can
        be changed with the parameter ``_movav``

        .. note:: the passed moving average must calculate alpha (and 1 -
                  alpha) and make them available as attributes ``alpha`` and
                  ``alpha1``

      - The 2nd moving averag can be changed from *Hull* to anything else with
        the param *_hma*

    See also:
      - https://www.reddit.com/r/algotrading/comments/4xj3vh/dickson_moving_average
    '''
    alias = ('DMA', 'DicksonMA',)
    lines = ('dma',)
    params = (
        ('gainlimit', 50),
        ('hperiod', 7),
        ('_movav', MovAv.EMA),
        ('_hma', MovAv.HMA),
    )

    def _plotlabel(self):
        plabels = [self.p.period, self.p.gainlimit, self.p.hperiod]
        plabels += [self.p._movav] * self.p.notdefault('_movav')
        plabels += [self.p._hma] * self.p.notdefault('_hma')
        return plabels

    def __init__(self):
        ec = ZeroLagIndicator(period=self.p.period,
                              gainlimit=self.p.gainlimit,
                              _movav=self.p._movav)

        hull = self.p._hma(period=self.p.hperiod)

        self.lines.dma = (ec + hull) / 2.0

        # To make mixins work - super at the end for cooperative inheritance
        super(DicksonMovingAverage, self).__init__()
