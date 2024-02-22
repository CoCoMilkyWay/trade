# sample=============================================================
'''
'''
# library============================================================
# (python 3.7)
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pandas_datareader.data as web
# regular expression
import re
import zipline
import alphalens
import pyfolio
import pypfopt
from zipline.finance import commission, slippage
from zipline.pipeline import *
from zipline.pipeline.factors import *
from zipline.api import (attach_pipeline, 
                         date_rules, 
                         time_rules,
                         get_datetime,
                         order_target_percent,
                         pipeline_output, 
                         record, 
                         schedule_function, 
                         get_open_orders, 
                         calendars,
                         set_commission, 
                         set_slippage)
from alphalens.utils import get_clean_factor_and_forward_returns
from alphalens.performance import *
from alphalens.plotting import *
from pyfolio.utils import extract_rets_pos_txn_from_zipline
from pyfolio.plotting import (plot_perf_stats,
                              show_perf_stats,
                              plot_rolling_beta,
                              plot_rolling_returns,
                              plot_rolling_sharpe,
                              plot_drawdown_periods,
                              plot_drawdown_underwater)
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, objective_functions
from pypfopt import expected_returns
from pypfopt.exceptions import OptimizationError
import sys
from logbook import (NestedSetup, NullHandler, Logger, StreamHandler, StderrHandler, 
                     INFO, WARNING, DEBUG, ERROR)
from datetime import datetime, timezone
from pathlib import Path
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# in-house library
import pylib_misc.helper as helper
#import factors

# strategy parameters================================================
start = pd.Timestamp('2014-1-1')
end = pd.Timestamp('2015-1-1')
analyze_factor = False; analyze_portfolio = True

# recent historical_return (normalized by stdev)
MONTH = 21 # number of market opening days in a month
YEAR = 12 * MONTH
# portfolio management
N_LONGS = N_SHORTS = 25
# when have enough long/short options, can enable holding optimization
MIN_POS = 5
# screened by dollar volume (size+liquidity)
VOL_SCREEN = 1000 #top 1000
capital_base = 1e7
long_short_neutral=True; group_neutral=False
strategy_type = 'eq_weight'
'''
    'eq_weight':        1/N in long/short
    'optimize_weight':  if sufficient long/short choices, 
                        given historic price, optimize weight
'''
optimize_type = 'max_sharpe'
'''
       historic price/proprietary model 
    -> expected return/risk model(covariance) 
    -> optimizer(objective:max (weighted)sharpe/min variance/... + constraints)
        Efficient Frontier
        Black-Litterman
        Hierarchical Risk Parity
        Mean-semivariance optimization
        Mean-CVaR optimization
        Hierarchical Risk Parity(clustering algorithms choose uncorrelated assets)
        Markowitz's critical line algorithm (CLA)
    -> diversified/weight-optimized portfolio
 
    'None':                 None
    'max_sharpe':           M.V.F.
    'max_weighted_sharpe':  M.V.F.
    'min_variance':         M.V.F.
'''
#====================================================================
prefix = f'{strategy_type}_{sys.argv[1]}' # for backtest data storage
HDF_PATH = Path('results/backtests.h5')
# python script argument parse (pick the right factor)===============
import importlib
module_factor = importlib.import_module('factors')
class_factor = getattr(module_factor, sys.argv[1])
instance_factor = class_factor()

simulation_time = pd.DataFrame(
    data=np.array([start.to_datetime64(), 
                   end.to_datetime64()]),
    index=['start', 'end'])

analysis_factor = False; analysis_portfolio = False

print("factor analysis begin: ", sys.argv[1])
print("analysis factor/portfolio: ", f'{analyze_factor}/{analyze_portfolio}')
print("long-short/group neutral: ", f'{long_short_neutral}/{group_neutral}')
print("strategy: ", f'{strategy_type}', "optimize: ", f'{optimize_type}')
print("simulation_time: ", simulation_time)

# setup stdout logging===============================================
format_string = '[{record.time: %H:%M:%S.%f}]: {record.level_name}: {record.message}'
zipline_logging = NestedSetup([NullHandler(level=DEBUG),
                               StreamHandler(sys.stdout, format_string=format_string, level=INFO),
                               StreamHandler(sys.stderr, level=ERROR)])
zipline_logging.push_application()
log = Logger('Algorithm')

# strategy===========================================================
def compute_factors0():
    """ call factor results
        rank assets using results
        further filter assets"""
    results = instance_factor
    filter = AverageDollarVolume(window_length=30)
    #eq_weight
    strategy_type = 'eq_weight' # global
    return Pipeline(columns={'longs': results.bottom(N_LONGS),
                             'shorts': results.top(N_SHORTS),
                             'ranking': results.rank(ascending=False)},
                    screen=filter.top(VOL_SCREEN))

## rebalance: order, optimize 
## (use: 1.daily pipeline results
##       2.content in BarData)
def exec_trades(data, assets, target_percent):
    """Place orders for assets using target portfolio percentage"""
    for asset in assets:
        if data.can_trade(asset) and not get_open_orders(asset):
            order_target_percent(asset, target_percent)
            #print(asset, target_percent)

def optimize_weights(prices, short=False):

    returns = expected_returns.mean_historical_return(
        prices=prices, frequency=252)
    cov = risk_models.sample_cov(prices=prices, frequency=252)

    # get weights that maximize the Sharpe ratio
    ef = EfficientFrontier(expected_returns=returns,
                           cov_matrix=cov,
                           weight_bounds=(0, 1),
                           solver='SCS')
    ef.max_sharpe()
    if short:
        return {asset: -weight for asset, weight in ef.clean_weights().items()}
    else:
        return ef.clean_weights()

def rebalance(context, data):
    """Compute long, short and obsolete holdings; place trade orders"""
    factor_data = context.factor_data
    assets = factor_data.index

    longs = assets[factor_data.longs]
    shorts = assets[factor_data.shorts]
    divest = set(context.portfolio.positions.keys()) - set(longs.union(shorts)) #lower invest, but still hold

    if strategy_type=='eq_weight':
        exec_trades(data, assets=longs, target_percent=1 / N_LONGS)
        exec_trades(data, assets=shorts, target_percent=-1 / N_SHORTS)
    elif strategy_type=='optimize_weight':
        # get price history
        prices = data.history(assets, fields='price',
                              bar_count=252+1, # for 1 year of returns 
                              frequency='1d')

        # get optimal weights if sufficient candidates
        if len(longs) > MIN_POS and len(shorts) > MIN_POS:
            try:
                long_weights = optimize_weights(prices.loc[:, longs], short=False)
                short_weights = optimize_weights(prices.loc[:, shorts], short=True)
                exec_trades(data, assets=longs, target_percent=long_weights)
                exec_trades(data, assets=shorts, target_percent=short_weights)
            except Exception as e: # optimize_weights function error
                log.warn('{} {}'.format(get_datetime().date(), e))
    log.info('{} | Longs: {:2.0f} | Shorts: {:2.0f} | {:,.2f}'.format(
        get_datetime(),
        len(longs), 
        len(shorts),
        context.portfolio.portfolio_value)
    )
    # exit remaining positions
    exec_trades(data, assets=divest, target_percent=0)

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
    context.set_commission(us_equities=commission.PerShare(cost=0.00075, 
                                                           min_trade_cost=.01))
    context.set_slippage(us_equities=slippage.VolumeShareSlippage(volume_limit=0.0025,
                                                                  price_impact=0.01))

## inter-day routine
## pipeline 1: compute factors (update to context)
## pipeline 2: ...
def before_trading_start(context, data): 
    """Run factor pipeline"""
    context.factor_data = pipeline_output('pipeline_1')
    record(factor_data=context.factor_data.ranking)
    assets = context.factor_data.index
    record(prices=data.current(assets, 'price'))

## intra-day routine 
def handle_data (context, data):
    ...

def analyze(context, perf):
    ...

sp500 = web.DataReader('SP500', 'fred', start, end).SP500
benchmark_returns = sp500.pct_change().tz_localize('UTC', level=0)

# pull stored data if avaliable=====================================
try:
    datetime_check = pd.read_hdf(HDF_PATH, f'datetime/{prefix}')
    start_stored = datetime_check.at['start', 0]
    end_stored = datetime_check.loc['end', 0]
    data_integrity = True #TODO
    use_storage_data = start_stored <= start and end <= end_stored and data_integrity==True
except Exception as e:
    use_storage_data = False
    pass
print('use_storage_data: ', use_storage_data)
if use_storage_data == True:
    analysis_factor = analyze_factor; analysis_portfolio = analyze_portfolio

# perf_result = store[f'backtest_meta/{prefix}'] # simulation results
factors = pd.read_hdf(HDF_PATH, f'factors/{prefix}').tz_localize('UTC', level=0) # complex data type loss tz info
prices = pd.read_hdf(HDF_PATH, f'prices/{prefix}')
pf_returns = pd.read_hdf(HDF_PATH, f'pf_returns/{prefix}')
pf_positions = pd.read_hdf(HDF_PATH, f'pf_positions/{prefix}')
pf_transactions = pd.read_hdf(HDF_PATH, f'pf_transactions_meta/{prefix}')
pf_benchmark = pd.read_hdf(HDF_PATH, f'pf_benchmark/{prefix}')

'''
before use data(e.g. quandl)
由于股票的几种行为，股价不连续
      除权（降低股价增加流动性），
      除息（不影响公司正常运行的情况下，短期无用，长期降低股东持股成本）
      盘前盘后交易，集合竞价（higher slippage）：
          提高流动性（吸引其他市场投资者）
          缓解突发消息带来的交易系统流动性负担
前复权：以除权除息后股价为基准（收益率直观显示买入/持仓成本，和当前股价match）
后复权：以除权除息前股价为基准（收益率更接近实际收益率，量化回测用（无未来信息））
'''
if use_storage_data == False:
    perf_result = zipline.run_algorithm(
        start=start.tz_localize('UTC'),
        end=end.tz_localize('UTC'),
        initialize=initialize,
        # handle_data=handle_data,
        before_trading_start=before_trading_start,
        analyze=analyze,
        capital_base=capital_base,
        benchmark_returns=benchmark_returns,
        bundle='quandl',
        data_frequency='daily'
    )

'''
"Fundamental Law of Active Management” asserts that the maximum attainable
IR is approximately the product of the Information Coefficient (IC) times the square root of the
breadth (BR) of the strategy
https://bigquant.com/wiki/home
Survivorship bias:                    we have
Look-ahead bias(future function):     use point-at-time data in regression engine
The sin of storytelling:              all time/universe robust
Data mining and data snooping:        logic-driven strategy
Signal decay, turnover, trans cost:   price/volume factors usually have worse decay
Outliers                              winsorization/ cut-off
The asymmetric payoff/shorting        option/future(stock index)
asset to sector dictionary
'''
asset_sector_mapping = pd.read_csv('../machine-learning-for-trading/data/us_equities_meta_data.csv',
                                   header=0,
                                   usecols=["ticker", "sector"],
                                   delimiter=','
                                   ).set_index('ticker')['sector'].to_dict()
if use_storage_data == False:
    # wash zipline dumped data to:======================
    #   dataframe:(date,asset)x(factor1,factor2): factor_values
    #   dataframe:(date)x(asset_names): price_values
    factors = pd.concat([df.to_frame(d) for d, df in perf_result.factor_data.dropna().items()],axis=1).T
    factors.index = factors.index.normalize() # normalize pandas datetime to 00:00
    factors = factors.stack(level=0)
    factors.index.names = ['date', 'asset']
    factors = factors.to_frame(name='factor_1')
    factors['sector'] = factors.index.get_level_values(1).values
    factors['sector'] = [re.findall(r"\[(.+)\]", str(equity_string))[0] for equity_string in factors['sector']]
    factors['sector'] = factors['sector'].map(asset_sector_mapping)
    prices = pd.concat([df.to_frame(d) for d, df in perf_result.prices.dropna().items()],axis=1).T
    prices.index = prices.index.normalize() # normalize pandas datetime to 00:00

# market beta benchmark
# resample, fill NaN, change timezone, filter to match dates with price info
# sp500 = web.DataReader('SP500', 'fred', start, end).SP500
# sp500 = sp500.resample('D').ffill().tz_localize('utc').filter(prices.index.get_level_values(0))
HOLDING_PERIODS = (1, 5, 10)
QUANTILES = 5
# feed to alphalens:================================
# require: max(periods) < end-start
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
    max_loss=0.5, # Maximum percentage (0.00 to 1.00) of factor data dropping allowed
    zero_aware=False, # your signal is centered and zero is the separation between long and short signals
    cumulative_returns=True
    )
if use_storage_data == False:
    # [(datetime):daily_return, (datetime * assets):holdings(percentage/dollar), (datetime):factor_universe_mean_daily_returns]
    al_returns, al_positions, al_benchmark = \
    create_pyfolio_input(
        factor_data,
        period='5D', # specify one period in [HOLDING_PERIODS]
        capital=capital_base, # percentage or dollar
        long_short=long_short_neutral,
        group_neutral=group_neutral,
        equal_weight=True, # factor weighted or equal weighted
        quantiles=[1,QUANTILES],
        groups=None,
        benchmark_period='1D'
    )
    _, _, pf_transactions = extract_rets_pos_txn_from_zipline(perf_result)

# analysis:===============================
'''
1. return analysis: (as factor)
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
        *annual alpha return (higher) (portfolio return unexplained by the benchmark return)
        *beta (not necessarily higher) (exposure to the benchmark)
        *mean period return by top/bottom quantile (higher/lower)
        *mean period spread (lower)
        *Calmar ratio: Annual portfolio return relative to maximal drawdown
        *Omega ratio: The probability-weighted ratio of gains versus losses for a return target, zero per default
        *Sortino ratio: Excess return relative to downside standard deviation
        *Daily value at risk (VaR): Loss corresponding to a return two standard deviations below the daily mean
2. information analysis (as factor)
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
        *IC mean (higher: 0.1-0.3)
        *IC std (lower)
        *risk-adjusted IC (higher: factor values vs forward return premium (fr - risk-free return))
        *t-stat(IC) (higher: two groups are t times as different from each other 
            as they are within each other) (2 groups: IC and normal distribution)
        *p-value(IC) (lower: <0.05, chances that t-stat is only by chance)
        *IC skew (lower, positive->0:  >0 means distribution is asymmetric to the right)
        *Tail ratio: Size of the right tail (gains, abs, 95th percentile value) relative to size of 
            left tail (losses) 
        *IC Kurtosis (lower, positive->3:  >3 means distribution has less extreme values comparing to normal distribution)
3. turnover analysis (as factor)
    top/bottom quantile turnover by period
        1. proportion of new emerging assets in this quantile in this day
        2. cannot be too high to avoid excessive commission
    mean factor rank autocorrelation by period
        1. correlation between current factor value rank to its previous rank
        2. should be close to 1, otherwise means excessive turnover(commission)
    *essential indicators:
        *turnover by all quantiles/period (<1% in a day)

factor analysis options:
    long-short (you want dollar neutral against beta exposure): factor value is relative(not absolute), 
          it only suggest one asset is better, not necessarily related to returns useful for most strategies
          (e.g. Beta hedging, Long-Short Equity strategies), disable in few (e.g. long only strategy)
    group-neutral (you want group neutral against sector exposure)

4. portfolio analysis: (as strategy)
    1. returns:
        cumulative return/volatility by portfolio/benchmark
        block plot of return by month/year
        daily/weekly/monthly return distributions
    2. rolling average:
        rolling volatility
        rolling sharpe ratio
        total/long/short holdings (daily/month): capacity of the strategy
    3. drawdown analysis
        top 5 drawdown periods
        underwater plot
    4. exposure analysis
        rolling portfolio beta: (risk exposure to benchmark(market/sector/asset) overtime)
        rolling Fama-French single factor betas (exposure to market, SMB/HML/UMD calculated natively)
        top 10 assets holding overtime (exposure to assets)
        long/short max/median position concentration overtime (<5%, exposure to assets)
        long/short holdings overtime (exposure to holdings)
    5. leverage analysis
        gross leverage
    6. exposure analysis to sector/style overtime
        sector + (momentum size value short_term_reversal volatility)
        (quantopian only)

risk models: 
      sector exposures to each sector:
      style exposures to size, value, quality, momentum, and volatility (betas)
tutorial: https://github.com/quantopian/alphalens/blob/master/alphalens/examples/alphalens_tutorial_on_quantopian.ipynb
factor metrics: https://github.com/quantopian/alphalens/blob/master/alphalens/examples/predictive_vs_non-predictive_factor.ipynb
'''
plt.close('all')

# check data integrity
print('        factors: ',         factors.shape)#index.get_level_values(0))
print('         prices: ',          prices.shape)#index                    )
print('     pf_returns: ',      pf_returns.shape)#index                    )
print('   pf_positions: ',    pf_positions.shape)#index                    )
print('pf_transactions: ', pf_transactions.shape)#index                    )
print('   pf_benchmark: ',    pf_benchmark.shape)#index                    )

pf_positions.columns = [c for c in pf_positions.columns[:-1]] + ['cash']
pf_positions.index = pf_positions.index.normalize()

if(analysis_factor):
    alphalens.tears.create_full_tear_sheet(
        factor_data,
        long_short=long_short_neutral,
        group_neutral=group_neutral,
        by_group=False
    )
    alphalens.tears.create_event_returns_tear_sheet(
        factor_data, prices, 
        avgretplot=(5, 15), 
        # plot quantile average cumulative returns
        # as x-axis: days before/after factor signal
        long_short=long_short_neutral, # strip beta(de-mean) for dollar neutral strategies
        group_neutral=group_neutral, # strip sector (de-mean at sector level)
        std_bar=True, 
        by_group=False
        )
        # create_summary_tear_sheet(factor_data, long_short=True, group_neutral=False)
        # create_returns_tear_sheet(factor_data, long_short=True, group_neutral=False, by_group=False)
        # create_information_tear_sheet(factor_data, group_neutral=False, by_group=False)
        # create_turnover_tear_sheet(factor_data)
if(analysis_portfolio):
    #pyfolio.tears.create_full_tear_sheet(
    #    pf_returns,
    #    positions=pf_positions,
    #    transactions=pf_transactions,
    #    benchmark_rets=pf_benchmark, # factor-universe-mean-daily-return (index benchmark) / daily-return of a particular asset
    #    hide_positions=True
    #    )
    fig, ax = plt.subplots(figsize=(15, 8))
    #sns.heatmap(pf_positions.replace(0, np.nan).dropna(how='all', axis=1).T, 
    #cmap=sns.diverging_palette(h_neg=20, h_pos=200), ax=ax, center=0)
    # filter common datas for these functions
    data_intersec = pf_returns.index & pf_benchmark.index
    print(data_intersec)
    pf_returns = pf_returns.loc[data_intersec]
    pf_positions = pf_positions.loc[data_intersec]
    print(pf_returns.info())
    plot_perf_stats(returns=pf_returns, 
                    factor_returns=pf_benchmark,     
                    ax=ax)
    show_perf_stats(returns=pf_returns, 
                    factor_returns=pf_benchmark, 
                    positions=pf_positions, 
                    transactions=pf_transactions, 
                    live_start_date=start)
    plot_rolling_returns(returns=pf_returns, 
                         factor_returns=pf_benchmark, 
                         live_start_date=start, 
                         cone_std=(1.0, 1.5, 2.0))
    plt.gcf().set_size_inches(14, 8)
if use_storage_data == False:
    with pd.HDFStore(HDF_PATH) as store: # use UTC because some packages would output UTC data anyways
        # store.put(f'backtest_meta/{prefix}', perf) # simulation results
        store.put(f'datetime/{prefix}',             simulation_time) # reuse database
        store.put(f'factors/{prefix}',              factors.tz_convert('UTC', level=0))
        store.put(f'prices/{prefix}',               prices.tz_convert('UTC', level=0))
        store.put(f'pf_returns/{prefix}',           pf_returns.tz_convert('UTC', level=0))
        store.put(f'pf_transactions_meta/{prefix}', pf_transactions.tz_convert('UTC', level=0)) # sid,symbol,price,order_id,amount,commission,dt,txn_dollar
        store.put(f'pf_positions/{prefix}',         pf_positions.tz_convert('UTC', level=0))
        store.put(f'pf_benchmark/{prefix}',         pf_benchmark.tz_convert('UTC', level=0))
'''
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
# align x axis
ax1.set_xlim(min(perf.period_open), max(perf.period_close))
# plot to remote terminal through SSH
# sns.despine()
# fig.tight_layout()
plt.legend(loc=0)
snap_cursor = helper.SnappingCursor_UTC_timestamp(ax1, 0)
fig.canvas.mpl_connect('motion_notify_event', snap_cursor.on_mouse_move)
plt.show() # would hang terminal
'''







