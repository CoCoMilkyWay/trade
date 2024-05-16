from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
import warnings
from datetime import datetime
import pytz
import numpy as np
import pandas as pd
import random
random.seed(datetime.now().timestamp())
warnings.filterwarnings('ignore')
import backtrader as bt

from _2_csv_data_parse import parse_csv_tradedate, parse_csv_metadata, parse_csv_kline_d1

def data_feed_dummy(cfg):
    datas = []
    for datafile in cfg.datafiles:
        datapath = os.path.join(cfg.modpath, f'{cfg.dataspath}/{datafile}')
        data = bt.feeds.YahooFinanceCSVData(
            dataname=datapath,
            # Do not pass values before this date
            fromdate=datetime(2000, 1, 1),
            # Do not pass values before this date
            todate=datetime(2000, 12, 31),
            # Do not pass values after this date
            reverse=False)
        datas.append(data)
    return datas

def data_feed_SSE(cfg):
        # real SSE data
    datas = []
    start_session = pytz.timezone(cfg.TZ).localize(datetime.strptime(cfg.START, '%Y-%m-%d'))
    end_session = pytz.timezone(cfg.TZ).localize(datetime.strptime(cfg.END, '%Y-%m-%d'))
    # trade_days, special_trade_days, special_holiday_days = parse_csv_tradedate()
    metadata, index_info = parse_csv_metadata(cfg) # index_info = [asset_csv_path, num_lines]
    symbol_map = metadata.loc[:,['symbol','asset_name','first_traded']]
    print(metadata.iloc[0,:3])
        # split:除权, merge:填权, dividend:除息
        # 用了后复权数据，不需要adjast factor
        # parse_csv_split_merge_dividend(symbol_map, start_session, end_session)
        # (Date) * (Open, High, Low, Close, Volume, OpenInterest)
    sids = cfg.stock_pool()
    for kline, pending_sids in parse_csv_kline_d1(symbol_map, index_info, start_session, end_session, sids):
        data = cfg.DATAFEED(dataname=kline)
        datas.append(data)
    if pending_sids:
        print('missing sids: ',pending_sids)
        # exit()
    return datas

def print_data_size(self):
    data_num = 0
    cell_len = 0
    for data in self.datas:
        data_num += 1
        tdata = sum(len(line.array) for line in data.lines)
        cell_len += tdata
    print(f'Total data memory cells ({data_num}) used: {cell_len}') # not counting indicator/observer

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

def analyze_portfolio(results, cfg):
    strat = results[0]
    pyfoliozer = strat.analyzers.getbyname('pyfolio')
    pf_returns, pf_positions, pf_transactions, gross_lev = pyfoliozer.get_pf_items()
    pf_benchmark = pf_returns
    print(type(pf_returns))
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('whitegrid')
    import _001_pyfolio as pf
    from _001_pyfolio.utils import extract_rets_pos_txn_from_zipline
    from _001_pyfolio.plotting import (
        plot_perf_stats,
        show_perf_stats,
        plot_rolling_beta,
        plot_rolling_returns,
        plot_rolling_sharpe,
        plot_drawdown_periods,
        plot_drawdown_underwater)
    pf.tears.create_full_tear_sheet(
        pf_returns,
        positions=pf_positions,
        transactions=pf_transactions,
        benchmark_rets=pf_benchmark, # factor-universe-mean-daily-return (index benchmark) / daily-return of a particular asset
        hide_positions=True
        )
    fig, ax_heatmap = plt.subplots(figsize=(15, 8))
    sns.heatmap(pf_positions.replace(0, np.nan).dropna(how='all', axis=1).T, 
    cmap=sns.diverging_palette(h_neg=20, h_pos=200), ax=ax_heatmap, center=0)
    # special requirements :(
    data_intersec = pf_returns.index & pf_benchmark.index
    pf_returns = pf_returns.loc[data_intersec]
    pf_positions = pf_positions.loc[data_intersec]
    fig, ax_perf = plt.subplots(figsize=(15, 8))
    plot_perf_stats(returns=pf_returns, 
                    factor_returns=pf_benchmark,     
                    ax=ax_perf)
    show_perf_stats(returns=pf_returns, 
                    factor_returns=pf_benchmark, 
                    positions=pf_positions, 
                    transactions=pf_transactions, 
                    live_start_date=cfg.oos)
    fig, ax_rolling = plt.subplots(figsize=(15, 8))
    plot_rolling_returns(
        returns=pf_returns, 
        factor_returns=pf_benchmark, 
        live_start_date=cfg.oos, 
        cone_std=(1.0, 1.5, 2.0),
        ax=ax_rolling)
    plt.gcf().set_size_inches(14, 8)

'''
MM定理(莫迪利安尼-米勒定理(股利政策无关论)): 在'无税收的完美市场'上，分红确实是毫无意义的
由于股票的几种行为，股价不连续
    除权(split/merge) (e.g. 送转分红后; usually not taxed; 降低/增加股价提升流动性,由于投机心理，实际上经常是弱利好),
    除息(dividend)    (e.g. 现金分红后; taxed;             短期无用(甚至手续费)，长期降低股东持股成本, 不如回购注销)
    apply to all data before effective date:
        split: OCHL multiply, volume divide
        merge: OCHL multiply, volume unaffected
        dividend(cash/stock): specify date, calculate on the fly
    
    盘前盘后交易,集合竞价(higher slippage):
        提高流动性(吸引其他市场投资者）
        缓解突发消息带来的交易系统流动性负担
前复权：以除权除息后股价为基准(收益率直观显示买入/持仓成本,和当前股价match)
后复权：以除权除息前股价为基准(收益率更接近实际收益率，量化回测用(无未来信息))
'''

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









