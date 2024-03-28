#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################

# https://github.com/bukosabino/ta
# https://github.com/twopirllc/pandas-ta?tab=readme-ov-file#indicators-by-category
# https://btalib.backtrader.com/indalpha/

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from backtrader.functions import *

# The modules below should/must define __all__ with the Indicator objects
# of prepend an "_" (underscore) to private classes/variables

from .basicops import *

# process: alert -> predict -> confirm
# timing effectiveness: leading/lagging/coincident
# source: trend/cycle/momentum/volatility/volume/fundamental
# usage:    trend: trend/super-trend/fundamental
#           space: oversell-buy/
#           timing: 

## trend (lagging >> leading) ==========================================

# moving averages (so envelope and oscillators can be auto-generated)
from .trends.moving_average._001_smma import *          # lag       ; trend
from .trends.moving_average._002_sma import *           # lag       ; trend
from .trends.moving_average._003_ema import *           # lag       ; trend
from .trends.moving_average._004_wma import *           # lag       ; trend
from .trends.moving_average._005_dema import *          # lag       ; trend
from .trends.moving_average._006_kama import *          # lag       ; trend
from .trends.moving_average._007_frama import * # TODO  # lag       ; trend
from .trends.moving_average._008_vidya import * # TODO  # lag       ; trend
from .trends.moving_average._009_zlind import *         # lag       ; trend
from .trends.moving_average._010_zlema import *         # lag       ; trend
from .trends.moving_average._011_hma import *           # lag       ; trend
from .trends.moving_average._012_dma import *           # lag       ; trend
from .trends.moving_average._013_jma import * # TODO    # lag       ; trend

# others:
# ADX (Average Directional Movement Index)
# ADXR (Average Directional Movement Index Rating)
# MIDPOINT (MidPoint over period)
# MIDPRICE (Midpoint Price over period)
# TRIX (1-day Rate-Of-Change (ROC) of a Triple Smooth EMA)
# TSF (Time Series Forecast)

from .basicops_lvl2 import * #stdev, crossover, oscillator, pctchange

## Volume (leading >> lagging) ========================================
#AD (Chaikin A/D Line)
#OBV (On Balance Volume)
from .volume.prettygoodoscillator import *
from .volume.accdecoscillator import *

# Volitality (lagging >> leading) ====================================

# Average True Range:
from .volatility.atr import *                      # lag       ; oversell-buy

# Normalized ATR:
# NATR (Normalized Average True Range)

# True Range:
# TRANGE (True Range)

# Standard Deviation:
from .volatility.stddev import *
from .volatility.bollinger import *                     # lag       ; oversell-buy
from .envelope import *
from .volatility.heikinashi import *
from .pivotpoint import *
from .volatility.psar import *
from .volatility.dv2 import *
from .volatility.kst import *
from .volatility.ichimoku import *
from .volatility.ols import *
from .volatility.hadelta import *

## Momentum (leading >> lagging) =====================================

# Chande Momentum Oscillator:
# CMO (Chande Momentum Oscillator)

# Rate of Change:
# ROC (Rate of Change)
# ROCP (Rate of Change Percentage)
# ROCR (Rate of Change Ratio)
# ROCR100 (Rate of Change Ratio 100 Scale)

# Relative Strength:
# RSI (Relative Strength Index)
from .momentum.rsi import *
from .momentum.lrsi import *

# Stochastic Oscillator:
# STOCH (Stochastic)
# STOCHF (Stochastic Fast)
# STOCHRSI (Stochastic Relative Strength Index)

# Williams' %R:
from .momentum.williams import *
from .momentum.momentum import *
from .momentum.aroon import *                           # coincident; oversell-buy
from .momentum.cci import *                             # coincident; oversell-buy

# MACD related:
# MACD (Moving Average Convergence/Divergence)
# MACDEXT (MACD with controllable MA type)
# MACDFIX (Moving Average Convergence/Divergence Fix 12/26)
from .momentum.macd import *

from .momentum.rmi import *
from .momentum.dpo import *
from .momentum.trix import *
from .momentum.tsi import *
from .momentum.ultimateoscillator import *
from .momentum.awesomeoscillator import *
from .momentum.priceoscillator import *
from .momentum.stochastic import *

## cycle(vs. trend) (leading >> lagging) based on Hurst theory =======
# 和分形维数一起使用，不构建策略，做风控
# HT_DCPERIOD (Hilbert Transform - Dominant Cycle Period)
# HT_DCPHASE (Hilbert Transform - Dominant Cycle Phase)
# HT_TRENDLINE (Hilbert Transform - Instantaneous Trendline)
# HT_TRENDMODE (Hilbert Transform - Trend vs Cycle Mode)
from .cycle.hurst import *

## Pattern Recognition Indicators: ===================================

# Candlestick Patterns:
# Various candlestick pattern indicators (CDL*)

# Other Pattern Indicators:
# Hikkake Pattern
# Hikkake Modified Pattern

## others ============================================================

## Beta:
# BETA (Beta)

## Correlation:
# CORREL (Pearson's Correlation Coefficient)

# Linear Regression:
# LINEARREG (Linear Regression)
# LINEARREG_ANGLE (Linear Regression Angle)
# LINEARREG_INTERCEPT (Linear Regression Intercept)
# LINEARREG_SLOPE (Linear Regression Slope)

# Extrema:
# MAX (Highest value over a specified period)
# MAXINDEX (Index of highest value over a specified period)
# MEDPRICE (Median Price)
# MIN (Lowest value over a specified period)
# MININDEX (Index of lowest value over a specified period)
# MINMAX (Lowest and highest values over a specified period)
# MINMAXINDEX (Indexes of lowest and highest values over a specified period)

# Directional Movement:
# PLUS_DI (Plus Directional Indicator)
# PLUS_DM (Plus Directional Movement)
# MINUS_DI (Minus Directional Indicator)
# MINUS_DM (Minus Directional Movement)
from .others.directionalmove import *
from .others.vortex import *

# Summation:
# SUM (Summation)

# Variance:
# VAR (Variance)

# best indicator 2024:
#
# amazing oscillator
# tri-state supertrend
# okx signal bot(Turtle Trade Channel TCI, Donchain channel)
# trend channels with liquidity breaks
# ranges with targets
# volume supertrend AI
# ultimate buy and sell indicator
# treadline breakouts with targets
# diy custome strategy builder
# imbalance algo
# TradeIQ
#
#