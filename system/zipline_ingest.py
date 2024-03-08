#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
https://financialzipline.wordpress.com/2016/08/24/importing-south-african-equities-data-into-zipline/
https://blog.csdn.net/weixin_30793643/article/details/97290610?spm=1001.2101.3001.6650.7&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-7-97290610-blog-62069281.235%5Ev43%5Econtrol&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-7-97290610-blog-62069281.235%5Ev43%5Econtrol&utm_relevant_index=8
https://github.com/rainx/zipline_cn_databundle/tree/master
https://blog.csdn.net/weixin_41245990/article/details/108320577?spm=1001.2101.3001.6650.6&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-6-108320577-blog-62069281.235%5Ev43%5Econtrol&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-6-108320577-blog-62069281.235%5Ev43%5Econtrol&utm_relevant_index=7
https://blog.csdn.net/weixin_30793643/article/details/97290610?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522170989065216800185847838%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=170989065216800185847838&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-21-97290610-null-null.142^v99^control&utm_term=zipline%20ingest&spm=1018.2226.3001.4187
https://blog.csdn.net/stefanie927/article/details/62069281?spm=1001.2101.3001.6650.8&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-8-62069281-blog-66007052.235%5Ev43%5Econtrol&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-8-62069281-blog-66007052.235%5Ev43%5Econtrol&utm_relevant_index=16
https://luyixiao.blog.csdn.net/article/details/73436626?spm=1001.2101.3001.6650.9&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-9-73436626-blog-66007052.235%5Ev43%5Econtrol&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-9-73436626-blog-66007052.235%5Ev43%5Econtrol&utm_relevant_index=17
'''


import sqlite3
from pathlib import Path
from os import getenv
import numpy as np
import pandas as pd
import sys

np.random.seed(42)

idx = pd.IndexSlice

data_path = Path('~', 'trade', 'data')

ZIPLINE_ROOT = getenv('ZIPLINE_ROOT')
if not ZIPLINE_ROOT:
    bundle_path = Path('~', '.zipline', 'data', 'A_stock').expanduser()
else:
    bundle_path = Path(ZIPLINE_ROOT, 'data', 'A_stock')

downloads = sorted([f.name for f in quandl_path.iterdir() if f.is_dir()])
if not downloads:
    print('Need to run "zipline ingest" first')
    exit()

adj_db_path = bundle_path / 'adjustments.sqlite'
equities_db_path = bundle_path / 'assets-7.sqlite'

#warnings.filterwarnings('ignore')

idx = pd.IndexSlice


def create_split_table():
    with pd.HDFStore('stooq.h5') as store:
        store.put('jp/splits', pd.DataFrame(columns=['sid', 'effective_date', 'ratio'],
                                            data=[[1, pd.to_datetime('2010-01-01'), 1.0]]), format='t')


def load_prices():
    df = pd.read_hdf(data_path / 'assets.h5', 'stooq/jp/tse/stocks/prices')

    return (df.loc[idx[:, '2014': '2019'], :]
            .unstack('ticker')
            .sort_index()
            .tz_localize('UTC')
            .ffill(limit=5)
            .dropna(axis=1)
            .stack('ticker')
            .swaplevel())


def load_symbols(tickers):
    df = pd.read_hdf(data_path / 'assets.h5', 'stooq/jp/tse/stocks/tickers')
    return (df[df.ticker.isin(tickers)]
            .reset_index(drop=True)
            .reset_index()
            .rename(columns={'index': 'sid'}))


if __name__ == '__main__':
    prices = load_prices()
    print(prices.info(null_counts=True))
    tickers = prices.index.unique('ticker')

    symbols = load_symbols(tickers)
    print(symbols.info(null_counts=True))
    symbols.to_hdf('stooq.h5', 'jp/equities', format='t')

    dates = prices.index.unique('date')
    start_date = dates.min()
    end_date = dates.max()

    for sid, symbol in symbols.set_index('sid').symbol.items():
        p = prices.loc[symbol]
        p.to_hdf('stooq.h5', 'jp/{}'.format(sid), format='t')

    with pd.HDFStore('stooq.h5') as store:
        print(store.info())

    create_split_table()
# ===================================================================================================
zipline_root = None

try:
    zipline_root = os.environ['ZIPLINE_ROOT']
except KeyError:
    print('Please ensure a ZIPLINE_ROOT environment variable is defined and accessible '
          '(or alter the script and manually set the path')
    exit()

custom_data_path = Path(zipline_root, 'custom_data')

# custom_data_path = Path('~/.zipline/custom_data').expanduser()


def load_equities():
    return pd.read_hdf(custom_data_path / 'stooq.h5', 'jp/equities')


def ticker_generator():
    """
    Lazily return (sid, ticker) tuple
    """
    return (v for v in load_equities().values)


def data_generator():
    for sid, symbol, asset_name in ticker_generator():
        df = pd.read_hdf(custom_data_path / 'stooq.h5', 'jp/{}'.format(sid))

        start_date = df.index[0]
        end_date = df.index[-1]

        first_traded = start_date.date()
        auto_close_date = end_date + pd.Timedelta(days=1)
        exchange = 'XTKS'

        yield (sid, df), symbol, asset_name, start_date, end_date, first_traded, auto_close_date, exchange


def metadata_frame():
    dtype = [
        ('symbol', 'object'),
        ('asset_name', 'object'),
        ('start_date', 'datetime64[ns]'),
        ('end_date', 'datetime64[ns]'),
        ('first_traded', 'datetime64[ns]'),
        ('auto_close_date', 'datetime64[ns]'),
        ('exchange', 'object'), ]
    return pd.DataFrame(np.empty(len(load_equities()), dtype=dtype))


def stooq_jp_to_bundle(interval='1d'):
    def ingest(environ,
               asset_db_writer,
               minute_bar_writer,
               daily_bar_writer,
               adjustment_writer,
               calendar,
               start_session,
               end_session,
               cache,
               show_progress,
               output_dir
               ):
        metadata = metadata_frame()

        def daily_data_generator():
            return (sid_df for (sid_df, *metadata.iloc[sid_df[0]]) in data_generator())

        daily_bar_writer.write(daily_data_generator(), show_progress=True)

        metadata.dropna(inplace=True)
        asset_db_writer.write(equities=metadata)
        # empty DataFrame
        adjustment_writer.write(splits=pd.read_hdf(custom_data_path / 'stooq.h5', 'jp/splits'))

    return ingest

sys.path.append(Path('~', '.zipline').expanduser().as_posix())
from zipline.data.bundles import register
from stooq_jp_stocks import stooq_jp_to_bundle
from datetime import time
from pytz import timezone

register('A_stock',
         stooq_jp_to_bundle(),
         calendar_name='XTKS',
         )