# sample=============================================================
"""from zipline import run_algorithm
from zipline.api import order_target, record, symbol
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader.data as web

def initialize(context):
    context.i = 0
    context.asset = symbol('AAPL')


def handle_data(context, data):
    # Skip first 300 days to get full windows
    context.i += 1
    if context.i < 300:
        return

    # Compute averages
    # data.history() has to be called with the same params
    # from above and returns a pandas dataframe.
    short_mavg = data.history(context.asset, 'price', bar_count=100, frequency="1d").mean()
    long_mavg = data.history(context.asset, 'price', bar_count=300, frequency="1d").mean()

    # Trading logic
    if short_mavg > long_mavg:
        # order_target orders as many shares as needed to
        # achieve the desired number of shares.
        order_target(context.asset, 100)
    elif short_mavg < long_mavg:
        order_target(context.asset, 0)

    # Save values for later inspection
    record(AAPL=data.current(context.asset, 'price'),
           short_mavg=short_mavg,
           long_mavg=long_mavg)


def analyze(context, perf):
    print(perf.info)
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    perf.portfolio_value.plot(ax=ax1)
    ax1.set_ylabel('portfolio value in $')
    ax1.set_xlim(min(perf.period_open), max(perf.period_open))

    ax2 = fig.add_subplot(212)
    perf['AAPL'].plot(ax=ax2)
    perf[['short_mavg', 'long_mavg']].plot(ax=ax2)

    perf_trans = perf.loc[[t != [] for t in perf.transactions]]
    buys = perf_trans.loc[[t[0]['amount'] > 0 for t in perf_trans.transactions]]
    sells = perf_trans.loc[
        [t[0]['amount'] < 0 for t in perf_trans.transactions]]
    ax2.plot(buys.index, perf.short_mavg.loc[buys.index],
             '^', markersize=10, color='m')
    ax2.plot(sells.index, perf.short_mavg.loc[sells.index],
             'v', markersize=10, color='k')
    ax2.set_ylabel('price in $')
    ax2.set_xlim(min(perf.period_open), max(perf.period_open))

    plt.legend(loc=0)
    plt.show()

start = pd.Timestamp('2014')
end = pd.Timestamp('2018')

sp500 = web.DataReader('SP500', 'fred', start, end).SP500
print(sp500)

benchmark_returns = sp500.pct_change()

result = run_algorithm(start=start.tz_localize('UTC'),
                       end=end.tz_localize('UTC'),
                       initialize=initialize,
                       handle_data=handle_data, (on data_frequency)
                       before_trading_start=before_trading_start, (on day frequency)
                       analyze=analyze,
                       capital_base=100000,
                       benchmark_returns=benchmark_returns,
                       bundle='quandl',
                       data_frequency='daily')"""

# library============================================================
# (python 3.7)
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import re # regular expression
from zipline import run_algorithm
from zipline.api import (
    attach_pipeline,
    date_rules,
    time_rules,
    order_target_percent,
    pipeline_output,
    record,
    schedule_function,
    get_open_orders,
    calendars
)
from zipline.finance import commission, slippage
from zipline.pipeline import *
from zipline.pipeline.factors import *
from alphalens.utils import get_clean_factor_and_forward_returns
from alphalens.performance import *
from alphalens.plotting import *
from alphalens.tears import *
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# in-house library
import pylib_misc.helper as helper
import factors

# strategy parameters================================================
# recent historical_return (normalized by stdev)
MONTH = 21 # number of market opening days in a month
YEAR = 12 * MONTH
# portfolio management
N_LONGS = N_SHORTS = 25
# screened by dollar volume (size+liquidity)
VOL_SCREEN = 10000 #top 1000
start = pd.Timestamp('2014-1-1')
end = pd.Timestamp('2018-1-1')

# strategy===========================================================

def compute_factors0():
    """ call factor results
        rank assets using results
        further filter assets"""
    results = factors.Factor_MeanReversion()
    filter = AverageDollarVolume(window_length=30)
    return Pipeline(columns={'longs': results.bottom(N_LONGS),
                             'shorts': results.top(N_SHORTS),
                             'ranking': results.rank(ascending=False)},
                    screen=filter.top(VOL_SCREEN))

## order/rebalance 
## (use: 1.daily pipeline results
##       2.content in BarData)
def exec_trades(data, assets, target_percent):
    """Place orders for assets using target portfolio percentage"""
    for asset in assets:
        if data.can_trade(asset) and not get_open_orders(asset):
            order_target_percent(asset, target_percent)
            #print(asset, target_percent)

def rebalance(context, data):
    """Compute long, short and obsolete holdings; place trade orders"""
    factor_data = context.factor_data
    record(factor_data=factor_data.ranking)

    assets = factor_data.index
    record(prices=data.current(assets, 'price'))

    longs = assets[factor_data.longs]
    shorts = assets[factor_data.shorts]
    divest = set(context.portfolio.positions.keys()) - set(longs.union(shorts))

    exec_trades(data, assets=divest, target_percent=0)
    exec_trades(data, assets=longs, target_percent=1 / N_LONGS)
    exec_trades(data, assets=shorts, target_percent=-1 / N_SHORTS)

# strategy wrapper (backtest interface)=============================
# queue intra-day pipeline to update context in sequence
# schedule intra-day(e.g. weekly) data handler
# set simulation parameter (commission/slippage model)
def initialize(context):
    """Setup: register pipeline, schedule rebalancing,
        and set trading params"""
    attach_pipeline(compute_factors0(), 'pipeline_1')
    schedule_function(rebalance,
                      date_rules.week_start(),
                      time_rules.market_open(),
                      calendar=calendars.US_EQUITIES)
    context.set_commission(commission.PerShare(cost=.01, min_trade_cost=0))
    context.set_slippage(slippage.VolumeShareSlippage())

## inter-day routine
def handle_data (context, data):
    ...

## intra-day routine 
## pipeline 1: compute factors (update to context)
## pipeline 2: ...
def before_trading_start(context, data): 
    """Run factor pipeline"""
    context.factor_data = pipeline_output('pipeline_1')

def analyze(context, perf):
    # "Fundamental Law of Active Management” asserts that the maximum attainable
    # IR is approximately the product of the Information Coefficient (IC) times the square root of the
    # breadth (BR) of the strategy

    # https://bigquant.com/wiki/home
    # Survivorship bias:                    we have
    # Look-ahead bias(future function):     use point-at-time data in regression engine
    # The sin of storytelling:              all time/universe robust
    # Data mining and data snooping:        logic-driven strategy
    # Signal decay, turnover, trans cost:   price/volume factors usually have worse decay
    # Outliers                              winsorization/ cut-off
    # The asymmetric payoff/shorting        option/future(stock index)
    
    # asset to sector dictionary
    asset_sector_mapping = pd.read_csv('../machine-learning-for-trading/data/us_equities_meta_data.csv',
                                       header=0,
                                       usecols=["ticker", "sector"],
                                       delimiter=','
                                       ).set_index('ticker')['sector'].to_dict()

    # wash zipline dumped data to:======================
    #   dataframe:(date,asset)x(factor1,factor2): factor_values
    #   dataframe:(date)x(asset_names): price_values
    factors = pd.concat([df.to_frame(d) for d, df in perf.factor_data.dropna().items()],axis=1).T
    factors.index = factors.index.normalize() # normalize pandas datetime to 00:00
    factors = factors.stack(level=0)
    factors.index.names = ['date', 'asset']
    factors = factors.to_frame(name='factor_1')
    factors['sector'] = factors.index.get_level_values(1).values
    factors['sector'] = [re.findall(r"\[(.+)\]", str(equity_string))[0] for equity_string in factors['sector']]
    factors['sector'] = factors['sector'].map(asset_sector_mapping)
    
    prices = pd.concat([df.to_frame(d) for d, df in perf.prices.dropna().items()],axis=1).T
    prices.index = prices.index.normalize() # normalize pandas datetime to 00:00

    # market beta benchmark
    # resample, fill NaN, change timezone, filter to match dates with price info
    sp500 = web.DataReader('SP500', 'fred', start, end).SP500
    sp500 = sp500.resample('D').ffill().tz_localize('utc').filter(prices.index.get_level_values(0))

    HOLDING_PERIODS = (1, 5, 10, 21)
    QUANTILES = 5

    # feed to alphalens:================================
    # require: max(periods) < end-start
    print(factors); print(prices)
    factor_data = get_clean_factor_and_forward_returns(  
        factor=factors['factor_1'],
        prices=prices,
        groupby=factors["sector"],
        binning_by_group=False, # compute quantile buckets separately for each group (sector specific factor)
        quantiles=QUANTILES, # Number of equal-sized quantile buckets to use in factor bucketing
        bins=None, # Number of equal-width (valuewise) bins to use in factor bucketing
        periods=HOLDING_PERIODS, # periods to compute forward returns on
        # filter_zscore=20, # standard deviation filter(on forward returns)
        groupby_labels=None, # A dictionary {group code:name} for display
        # max_loss=0.35, # Maximum percentage (0.00 to 1.00) of factor data dropping allowed
        zero_aware=False, # your signal is centered and zero is the separation between long and short signals
        cumulative_returns=True
        )

    # analysis:===============================
    '''
    1. return analysis:
        Event-Study-Like Return:
            1. see factor signal impact on return before/after xx days 
                (e.g. the second day after rebalance has best return gain)
        Mean-Return-by-quantile/period/sector
            1. possible that multiple rebalances in one period could have overlapped effects
            2. cannot see the effect at a particular date (e.g. second day) like event-based plot
            3. violin plot: see the full distribution (check noise/variance of the factor)
        Factor-weighted-cumulative-return by period/sector
            1. not by quantile: long/short every single assets in the universe according to factor value as 
                weight in a rebalance. If this plot shows good result, it means this factor is consistent across
                the whole universe (effective among all) (hint that this is a good factor to weight portfolio position)
        cumulative-return by quantile/period/sector
            1. in each quantile, assets have equal weights
            2. see do we have particularly strong top/bottom quantiles
            3. "Fundamental Law of Active Management":
                IR = IC * sqrt(BR)
                even for 5D period, calculation needs to be done daily that:
                    1. maximize information ratio
                    2. lower volatility
                    3. results are independent to start of each epoch(overlapped with each other)
                    4. increase capacity and thus lower slippage
        Top-minus-bottom return by period
            1. clearly show drawdown period (harder to see in cumulative returns)
            2. also show standard deviation (spread) of +-1 of drawdowns/positive returns
        *essential indicators:
            *annual alpha return (higher)
            *beta (not necessarily higher)
            *mean period return by top/bottom quantile (higher/lower)
            *mean period spread (lower)
    2. information analysis
        Information-Coefficient = Spearman's rank correlation coefficient between factor value and forward return
            = for 2 samples A, B: Cov(Rank(A), Rank(B))/std(A)std(B)
            less sensitive to outliers comparing to Pearson correlation
            - forward returns = projected annual return from daily return (100bps = 100basis points = 1%)
        Forward-return-Information-Coefficients by period/sector
            1. describe how consistent rank of factor value is to rank of forward returns
        Forward-return-Information-Coefficients (distribution: Histogram and Q-Q plot) by period
            1. Histogram: see shape of daily IC distribution
            2. Q-Q plot: compare IC distribution with normal distribution:
                1. normal distribution: factor not predictive at all
                2. predictive-factor: S-shaped curve especially at both tails
        *essential indicators:
            IC mean (higher: 0.1-0.3)
            IC std (lower)
            risk-adjusted IC (higher: factor values vs forward return premium (fr - risk-free return))
            t-stat(IC) (higher: two groups are t times as different from each other 
                as they are within each other) (2 groups: IC and normal distribution)
            p-value(IC) (lower: <0.05, chances that t-stat is only by chance)
            IC skew (lower, positive->0:  >0 means distribution is asymmetric to the right)
            IC Kurtosis (lower, positive->3:  >3 means distribution has less extreme values comparing to normal distribution)
    3. turnover analysis
        top/bottom quantile turnover
            
        turnover by quantile
        mean factor rank autocorrelation

    factor analysis options:
        long-short (you want dollar neutral against beta exposure): factor value is relative(not absolute), 
              it only suggest one asset is better, not necessarily related to returns useful for most strategies
              (e.g. Beta hedging, Long-Short Equity strategies), disable in few (e.g. long only strategy)
        group-neutral (you want group neutral against sector exposure):
    '''
    





    # risk models: 
    #       sector exposures to each sector:
    #       style exposures to size, value, quality, momentum, and volatility (betas)

    # tutorial: https://github.com/quantopian/alphalens/blob/master/alphalens/examples/alphalens_tutorial_on_quantopian.ipynb
    # factor metrics: https://github.com/quantopian/alphalens/blob/master/alphalens/examples/predictive_vs_non-predictive_factor.ipynb
    create_full_tear_sheet(factor_data, 
                           long_short=False, 
                           group_neutral=False, 
                           by_group=True)
    create_event_returns_tear_sheet(factor_data, prices, 
                                    avgretplot=(5, 15), 
                                    # plot quantile average cumulative returns
                                    # as x-axis: days before/after factor signal
                                    long_short=False, # strip beta(de-mean) for dollar neutral strategies
                                    group_neutral=False, # strip sector (de-mean at sector level)
                                    std_bar=True, 
                                    by_group=True)
    # create_summary_tear_sheet(factor_data, long_short=True, group_neutral=False)
    # create_returns_tear_sheet(factor_data, long_short=True, group_neutral=False, by_group=False)
    # create_information_tear_sheet(factor_data, group_neutral=False, by_group=False)
    # create_turnover_tear_sheet(factor_data)

    '''
    # factor_data.reset_index().to_csv('factor_data.csv', index=False)
    plot_quantile_returns_bar(mean_return_by_q)
    plot_cumulative_returns_by_quantile(mean_return_by_q_daily['5D'], period='5D', freq=None)
    plot_quantile_returns_violin(mean_return_by_q_daily)
    plot_ic_ts(ic[['5D']])
    ic_by_year.plot.bar(figsize=(14, 6))
    create_turnover_tear_sheet(factor_data)
    # create_summary_tear_sheet(factor_data)


    plt.close('all')
    fig = plt.figure()

    # basic
    ax1 = fig.add_subplot(311)
    perf[['algorithm_period_return',
          #'benchmark_period_return',
          'algo_volatility',
          #'benchmark_volatility',
          #'returns',
          #'excess_return'
          ]].plot(ax=ax1)
    perf_trans = perf.loc[[t != [] for t in perf.transactions]]
    buys = perf_trans.loc[[t[0]['amount'] > 0 for t in perf_trans.transactions]]
    sells = perf_trans.loc[[t[0]['amount'] < 0 for t in perf_trans.transactions]]
    ax1.plot(buys.index, perf.algorithm_period_return.loc[buys.index],
             '^', markersize=10, color='m')
    ax1.plot(sells.index, perf.algorithm_period_return.loc[sells.index],
             'v', markersize=10, color='k')
    #ax1.legend(['Cumulative Return', 'Volatility', 'buys', 'sells'])

    ax2 = fig.add_subplot(312)

    # plot 3
    ax3 = fig.add_subplot(313)
    perf.sharpe.plot(ax=ax3)
    ax3.set_ylabel('Sharpe')

    # align x axis
    ax1.set_xlim(min(perf.period_open), max(perf.period_close))
    ax2.set_xlim(min(perf.period_open), max(perf.period_close))
    ax3.set_xlim(min(perf.period_open), max(perf.period_close))

    # plot to remote terminal through SSH
    # sns.despine()
    # fig.tight_layout()
    plt.legend(loc=0)
    snap_cursor = helper.SnappingCursor_UTC_timestamp(ax1, 0)
    fig.canvas.mpl_connect('motion_notify_event', snap_cursor.on_mouse_move)
    plt.show() # would hang terminal
    '''

sp500 = web.DataReader('SP500', 'fred', start, end).SP500
benchmark_returns = sp500.pct_change()

# 由于股票的几种行为，股价不连续
#       除权（降低股价增加流动性），
#       除息（不影响公司正常运行的情况下，短期无用，长期降低股东持股成本）
#       盘前盘后交易，集合竞价（higher slippage）：
#           提高流动性（吸引其他市场投资者）
#           缓解突发消息带来的交易系统流动性负担
# 前复权：以除权除息后股价为基准（收益率直观显示买入/持仓成本，和当前股价match）
# 后复权：以除权除息前股价为基准（收益率更接近实际收益率，量化回测用（无未来信息））
perf_result = run_algorithm(start=start.tz_localize('UTC'),
                       end=end.tz_localize('UTC'),
                       initialize=initialize,
                       # handle_data=handle_data,
                       before_trading_start=before_trading_start,
                       analyze=analyze,
                       capital_base=100000,
                       benchmark_returns=benchmark_returns,
                       bundle='quandl',
                       data_frequency='daily')
