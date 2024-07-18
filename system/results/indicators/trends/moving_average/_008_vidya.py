#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


from ...basicops import MovingAverageBase


# Inherits from MovingAverageBase to auto-register as MovingAverage type
class VariableIndexDynamicAverage(MovingAverageBase):
    '''
    指数动态均线(period,short,long,smooth)
    自适应均线的一种，通过根据最近的波动性改变平滑度来修改指数移动平均线
    短周期n、长周期m和平滑因子a
    VIDYA 本身一般不用于交易，而是使用 VIDYA 上下 N% 的边界(轨线)
    VIDyA = 2 / (BP +1) * VI * (Close-Previous VIDYA) + Previous VIDYA
    BP = the user bar period for the MA
    VI = Volatility Index which is used dynamically to adapt the bar period to a trend.
    While VI could be any indicator, Efficiency Ratio is the most used for this purpose
    VI = ER = Change / Sum of absolute changes

    By Tushar Chande
    '''
    alias = ('VIDYA', 'VaryIndexMA',)
    lines = ('vidya',)
