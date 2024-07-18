#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


from ...basicops import MovingAverageBase


class JurikMovingAverage(MovingAverageBase):
    '''
    朱里克均线(period)
    Adaptive, low lag, data smoother, Less noise + better timing

    See also:
      - http://jurikres.com/catalog1/ms_ama.htm
    '''
    alias = ('JMA', 'JurikMA',)
    lines = ('jma',)