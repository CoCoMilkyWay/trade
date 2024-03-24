#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


from ...basicops import MovingAverageBase


# Inherits from MovingAverageBase to auto-register as MovingAverage type
class FractalAdaptiveMovingAverage(MovingAverageBase):
    '''
    分形自适应均线(period, _movav=WMA)
    FRAMA 均线对滞后性的提高非常明显
    取决于选取的参数, FRAMA 均线在一些局部可能不够平滑
    The fractal adaptive moving average, FRAMA, tracks price closely. 
    But when there is increased volatility, the indicator will slow down. 
    The indicator can pinpoint market turning points and help reduce 
    noise from price movements.
    '''
    alias = ('FRAMA', 'FractalMA',)
    lines = ('frama',)