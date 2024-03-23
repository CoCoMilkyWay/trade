#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################

# https://github.com/bukosabino/ta
# https://github.com/twopirllc/pandas-ta?tab=readme-ov-file#indicators-by-category
# https://btalib.backtrader.com/indalpha/

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from backtrader import Indicator
from backtrader.functions import *

# The modules below should/must define __all__ with the Indicator objects
# of prepend an "_" (underscore) to private classes/variables

from .basicops import *

# moving averages (so envelope and oscillators can be auto-generated)
from .trends.moving_average._001_sma import *
from .trends.moving_average._002_wma import *
from .trends.moving_average._003_ema import *
from .trends.moving_average._004_smma import *
from .trends.moving_average._005_dema import *
from .trends.moving_average._006_kama import *
from .trends.moving_average._007_zlema import *
from .trends.moving_average._008_hma import *
from .trends.moving_average._009_frama import * # TODO
from .trends.moving_average._010_vidya import * # TODO
from .trends.moving_average._011_zlind import *
from .trends.moving_average._012_dma import *
from .trends.moving_average._013_jma import *
from .trends.moving_average._014_deviation import *
#
## depend on basicops, moving averages and deviations
#from .trends.moving_average.atr import *
#from .trends.moving_average.aroon import *
#from .trends.moving_average.bollinger import *
#from .trends.moving_average.cci import *
#from .trends.moving_average.crossover import *
#from .trends.moving_average.dpo import *
#from .trends.moving_average.directionalmove import *
#from .trends.moving_average.envelope import *
#from .trends.moving_average.heikinashi import *
#from .trends.moving_average.lrsi import *
#from .trends.moving_average.macd import *
#from .momentum import *
#from .trends.moving_average.oscillator import *
#from .trends.moving_average.percentchange import *
#from .trends.moving_average.percentrank import *
#from .trends.moving_average.pivotpoint import *
#from .trends.moving_average.prettygoodoscillator import *
#from .trends.moving_average.priceoscillator import *
#from .trends.moving_average.psar import *
#from .trends.moving_average.rsi import *
#from .trends.moving_average.stochastic import *
#from .trends.moving_average.trix import *
#from .trends.moving_average.tsi import *
#from .trends.moving_average.ultimateoscillator import *
#from .trends.moving_average.williams import *
#from .rmi import *
#from .awesomeoscillator import *
#from .accdecoscillator import *
#
#from .dv2 import *  # depends on percentrank
#
## Depends on Momentum
#from .kst import *
#
#from .ichimoku import *
#
#from .hurst import *
#from .ols import *
#from .hadelta import *
