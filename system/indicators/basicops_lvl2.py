#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import contextlib
import sys
from .basicops import Indicator, MovAv, MovingAverage, BaseApplyN
from backtrader.functions import And

class StandardDeviation(Indicator):
    '''
    方差 (period=N, movav=SMA, safepow=True)
    Calculates the standard deviation of the passed data for a given period

    Note:
      - If 2 datas are provided as parameters, the 2nd is considered to be the
        mean of the first

      - ``safepow`` (default: False) If this parameter is True, the standard
        deviation will be calculated as pow(abs(meansq - sqmean), 0.5) to safe
        guard for possible negative results of ``meansq - sqmean`` caused by
        the floating point representation.

    Formula:
      - meansquared = SimpleMovingAverage(pow(data, 2), period)
      - squaredmean = pow(SimpleMovingAverage(data, period), 2)
      - stddev = pow(meansquared - squaredmean, 0.5)  # square root

    See:
      - http://en.wikipedia.org/wiki/Standard_deviation
    '''
    alias = ('StdDev',)

    lines = ('stddev',)
    params = (('period', 20), ('movav', MovAv.SMA), ('safepow', True),)

    def _plotlabel(self):
        plabels = [self.p.period]
        plabels += [self.p.movav] * self.p.notdefault('movav')
        return plabels

    def __init__(self):
        if len(self.datas) > 1:
            mean = self.data1
        else:
            mean = self.p.movav(self.data, period=self.p.period)

        meansq = self.p.movav(pow(self.data, 2), period=self.p.period)
        sqmean = pow(mean, 2)

        if self.p.safepow:
            self.lines.stddev = pow(abs(meansq - sqmean), 0.5)
        else:
            self.lines.stddev = pow(meansq - sqmean, 0.5)


class MeanDeviation(Indicator):
    '''MeanDeviation (alias MeanDev)

    Calculates the Mean Deviation of the passed data for a given period

    Note:
      - If 2 datas are provided as parameters, the 2nd is considered to be the
        mean of the first

    Formula:
      - mean = MovingAverage(data, period) (or provided mean)
      - absdeviation = abs(data - mean)
      - meandev = MovingAverage(absdeviation, period)

    See:
      - https://en.wikipedia.org/wiki/Average_absolute_deviation
    '''
    alias = ('MeanDev',)

    lines = ('meandev',)
    params = (('period', 20), ('movav', MovAv.Simple),)

    def _plotlabel(self):
        plabels = [self.p.period]
        plabels += [self.p.movav] * self.p.notdefault('movav')
        return plabels

    def __init__(self):
        if len(self.datas) > 1:
            mean = self.data1
        else:
            mean = self.p.movav(self.data, period=self.p.period)

        absdev = abs(self.data - mean)
        self.lines.meandev = self.p.movav(absdev, period=self.p.period)

class NonZeroDifference(Indicator):
    '''
    Keeps track of the difference between two data inputs skipping, memorizing
    the last non zero value if the current difference is zero

    Formula:
      - diff = data - data1
      - nzd = diff if diff else diff(-1)
    '''
    _mindatas = 2  # requires two (2) data sources
    alias = ('NZD',)
    lines = ('nzd',)

    def nextstart(self):
        self.l.nzd[0] = self.data0[0] - self.data1[0]  # seed value

    def next(self):
        d = self.data0[0] - self.data1[0]
        self.l.nzd[0] = d or self.l.nzd[-1]

    def oncestart(self, start, end):
        self.line.array[start] = (
            self.data0.array[start] - self.data1.array[start])

    def once(self, start, end):
        d0array = self.data0.array
        d1array = self.data1.array
        larray = self.line.array

        prev = larray[start - 1]
        for i in range(start, end):
            d = d0array[i] - d1array[i]
            larray[i] = prev = d or prev


class _CrossBase(Indicator):
    _mindatas = 2

    lines = ('cross',)

    plotinfo = dict(plotymargin=0.05, plotyhlines=[0.0, 1.0])

    def __init__(self):
        nzd = NonZeroDifference(self.data0, self.data1)

        if self._crossup:
            before = nzd(-1) < 0.0  # data0 was below or at 0
            after = self.data0 > self.data1
        else:
            before = nzd(-1) > 0.0  # data0 was above or at 0
            after = self.data0 < self.data1

        self.lines.cross = And(before, after)


class CrossUp(_CrossBase):
    '''
    This indicator gives a signal if the 1st provided data crosses over the 2nd
    indicator upwards

    It does need to look into the current time index (0) and the previous time
    index (-1) of both the 1st and 2nd data

    Formula:
      - diff = data - data1
      - upcross =  last_non_zero_diff < 0 and data0(0) > data1(0)
    '''
    _crossup = True


class CrossDown(_CrossBase):
    '''
    This indicator gives a signal if the 1st provided data crosses over the 2nd
    indicator upwards

    It does need to look into the current time index (0) and the previous time
    index (-1) of both the 1st and 2nd data

    Formula:
      - diff = data - data1
      - downcross = last_non_zero_diff > 0 and data0(0) < data1(0)
    '''
    _crossup = False


class CrossOver(Indicator):
    '''
    This indicator gives a signal if the provided datas (2) cross up or down.

      - 1.0 if the 1st data crosses the 2nd data upwards
      - -1.0 if the 1st data crosses the 2nd data downwards

    It does need to look into the current time index (0) and the previous time
    index (-1) of both the 1t and 2nd data

    Formula:
      - diff = data - data1
      - upcross =  last_non_zero_diff < 0 and data0(0) > data1(0)
      - downcross = last_non_zero_diff > 0 and data0(0) < data1(0)
      - crossover = upcross - downcross
    '''
    _mindatas = 2

    lines = ('crossover',)

    plotinfo = dict(plotymargin=0.05, plotyhlines=[-1.0, 1.0])

    def __init__(self):
        upcross = CrossUp(self.data, self.data1)
        downcross = CrossDown(self.data, self.data1)

        self.lines.crossover = upcross - downcross


class OscillatorMixIn(Indicator):
    '''
    MixIn class to create a subclass with another indicator. The main line of
    that indicator will be substracted from the other base class main line
    creating an oscillator

    The usage is:

      - Class XXXOscillator(XXX, OscillatorMixIn)

    Formula:
      - XXX calculates lines[0]
      - osc = self.data - XXX.lines[0]
    '''
    plotlines = dict(_0=dict(_name='osc'))

    def _plotinit(self):
        with contextlib.suppress(AttributeError):
            lname = self.lines._getlinealias(0)
            self.plotlines._0._name = f'{lname}_osc'

    def __init__(self):
        self.lines[0] = self.data - self.lines[0]
        super(OscillatorMixIn, self).__init__()


class Oscillator(Indicator):
    '''
    Oscillation of a given data around another data

    Datas:
      This indicator can accept 1 or 2 datas for the calculation.

      - If 1 data is provided, it must be a complex "Lines" object (indicator)
        which also has "datas". Example: A moving average

        The calculated oscillation will be that of the Moving Average (in the
        example) around the data that was used for the average calculation

      - If 2 datas are provided the calculated oscillation will be that of the
        2nd data around the 1st data

    Formula:
      - 1 data -> osc = data.data - data
      - 2 datas -> osc = data0 - data1
    '''
    lines = ('osc',)

    # Have a default value which can be later modified if needed
    plotlines = dict(_0=dict(_name='osc'))

    def _plotinit(self):
        with contextlib.suppress(AttributeError):
            lname = self.dataosc._getlinealias(0)
            self.plotlines._0._name = f'{lname}_osc'

    def __init__(self):
        super(Oscillator, self).__init__()

        if len(self.datas) > 1:
            datasrc = self.data
            self.dataosc = self.data1
        else:
            datasrc = self.data.data
            self.dataosc = self.data

        self.lines[0] = datasrc - self.dataosc


# Automatic creation of Oscillating Lines

_newclsdoc = '''
    Oscillation of a %s around its data
    '''
for movav in MovingAverage._movavs[1:]:
    # Skip aliases - they will be created automatically
    if getattr(movav, 'aliased', ''):
        continue

    movname = movav.__name__
    linename = movav.lines._getlinealias(0)
    newclsname = f'{movname}Oscillator'

    newaliases = [f'{movname}Osc']
    for alias in getattr(movav, 'alias', []):
        newaliases.extend(alias + suffix for suffix in ['Oscillator', 'Osc'])
    newclsdoc = _newclsdoc % movname
    newclsdct = {'__doc__': newclsdoc,
                 '__module__': OscillatorMixIn.__module__,
                 '_notregister': True,
                 'alias': newaliases}

    newcls = type(str(newclsname), (movav, OscillatorMixIn), newclsdct)
    module = sys.modules[OscillatorMixIn.__module__]
    setattr(module, newclsname, newcls)

class PercentChange(Indicator):
    '''
      Measures the perccentage change of the current value with respect to that
      of period bars ago
    '''
    alias = ('PctChange',)
    lines = ('pctchange',)

    # Fancy plotting name
    plotlines = dict(pctchange=dict(_name='%change'))

    # update value to standard for Moving Averages
    params = (('period', 30),)

    def __init__(self):
        self.lines.pctchange = self.data / self.data(-self.p.period) - 1.0
        super(PercentChange, self).__init__()
        
from math import fsum
class PercentRank(BaseApplyN):
    '''
    Measures the percent rank of the current value with respect to that of
    period bars ago
    '''
    alias = ('PctRank',)
    lines = ('pctrank',)
    params = (
        ('period', 50),
        ('func', lambda d: fsum(x < d[-1] for x in d) / len(d)),
    )
