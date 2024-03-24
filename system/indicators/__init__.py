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

# trend (lagging >> leading) ==========================================
## moving averages (so envelope and oscillators can be auto-generated)
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

from .basicops_lvl2 import * #stdev, crossover

# Volitality (lagging >> leading) ====================================
from .volatility._001_atr import *                      # lag       ; oversell-buy
from .volatility.bollinger import *                     # lag       ; oversell-buy


# Momentum (leading >> lagging)
from .momentum.aroon import *                           # coincident; oversell-buy
from .momentum.cci import *                             # coincident; oversell-buy


# Volume (leading >> lagging)

# cycle(vs. trend) (leading >> lagging) based on Hurst theory
# 和分形维数一起使用，不构建策略，做风控
# Hibert Transform: 
#   Dominant Cycle Period
#   Dominant Cycle Phase
#   Phasor
#   Sine
#   Trend Mode

# others

# from .dpo import *
# from .directionalmove import *
# from .envelope import *
# from .heikinashi import *
# from .lrsi import *
# from .macd import *
# from .momentum import *
# from .oscillator import *
# from .percentchange import *
# from .percentrank import *
# from .pivotpoint import *
# from .prettygoodoscillator import *
# from .priceoscillator import *
# from .psar import *
# from .rsi import *
# from .stochastic import *
# from .trix import *
# from .tsi import *
# from .ultimateoscillator import *
# from .williams import *
# from .rmi import *
# from .awesomeoscillator import *
# from .accdecoscillator import *
# 
# from .dv2 import *  # depends on percentrank
# 
# # Depends on Momentum
# from .kst import *
# 
# from .ichimoku import *
# 
# from .hurst import *
# from .ols import *
# from .hadelta import *

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