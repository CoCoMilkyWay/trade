import numpy as np
import pandas as pd
import click
import os
from datetime import datetime
import time
import pytz
from tqdm import tqdm

from zipline.data.bundles import register

# pretend UTC is Asia/Shanghai (easy calculation)
# tz = "Asia/Shanghai"
tz = "UTC"
API='baostock'

def injest_bundle(
    environ,
    asset_db_writer,
    minute_bar_writer,
    daily_bar_writer,
    adjustment_writer,
    calendar,
    start_session,
    end_session,
    cache,
    show_progress,
    output_dir):
    '''
    move this script to ~/.zipline/
    modify:  ~/.zipline/extension.py
    terminal run:zipline ingest -b A_stock
    '''
    metadata = parse_api_metadata()
    symbol_map = metadata.loc[:,['symbol','first_traded']]
    print(metadata)
    # 写入股票基础信息
    asset_db_writer.write(metadata)
    
    # 准备写入 dailybar/minutebar(lazzy iterable)
    # ('day'/'min') * ('open','high','low','close','volume')
    # minute_bar_writer.write(parse_api_kline_m5(symbol_map, start_session, end_session), show_progress=show_progress)
    daily_bar_writer.write(data=parse_api_kline_d1(symbol_map, start_session, end_session), show_progress=show_progress)
    
    # split:除权, merge:填权, dividend:除息
    # 用了后复权数据，不需要adjast factor
    #adjustment_writer.write(
    #    splits=pd.concat(splits, ignore_index=True),
    #    dividends=pd.concat(dividends, ignore_index=True),
    #)

def parse_api_metadata(show_progress=True, type='equity'):
    '''
    type = 
        1. equity(stock, funds, ETF)
        2. fixed income(bond, debt)
        3. derivatives(option, furture, swap)
    '''
    def metadata_frame_equity():
        dtype = [
            ('symbol', 'object'),
            ('asset_name', 'object'),
            ('start_date', 'datetime64[ns]'),
            ('end_date', 'datetime64[ns]'),
            ('first_traded', 'datetime64[ns]'),
            ('auto_close_date', 'datetime64[ns]'),
            ('exchange', 'object'), ]
        return pd.DataFrame(np.empty(stock_len, dtype=dtype))
    
    def metadata_frame_future(): # not implemented
        pass

    time0 = time.time()
    # 获取证券基本资料
    rs = api.query_stock_basic() # code="sh.600000" code_name="浦发银行"
    stock_len = len(rs.data)
    time1 = time.time()
    '''
    type: 1:股票 2:指数 3:其他 4:可转债(cb) 5:etf
    status: 1:上市 0:退市
    上市公司发行可转债:
        不特定对象: 可公开,票面利率一般低于普通公司债(低成本融资)
        特定对象: 不可公开
    '''
    if(type=='equity'):
        metadata = metadata_frame_equity()
    elif(type=='future'):
        metadata = metadata_frame_future()
    
    # 打印结果集
    row_index = 0
    p_bar = tqdm(range(stock_len))
    while (rs.error_code == '0') & rs.next():
        if(row_index>10):
            break
        # 获取一条记录，将记录合并在一起
        # [code code_name ipoDate outDate type status]
        data_row = rs.get_row_data()
        if(data_row[5]=='0'): # 只保留未停牌的
            continue
        if(data_row[4]!='1'): # 只要股票
            continue
        start_date = pd.to_datetime(data_row[2])
        first_traded = start_date.date()
        end_date = datetime.now().date()
        auto_close_date = end_date + pd.Timedelta(days=1)
        metadata.loc[row_index] = [
            data_row[0], # 'symbol'
            data_row[1], # 'asset_name'
            start_date, # 'start_date' (ipo date)
            end_date, # 'end_date' (data_end)
            first_traded, # 'first_traded' (data_start)
            auto_close_date, # 'auto_close_date'
            get_exchange(data_row[0]), # 'exchange',
        ]
        p_bar.n = row_index
        p_bar.refresh()
        row_index = row_index + 1
    metadata = metadata.iloc[:row_index]
    time2 = time.time()
    if show_progress:
        click.echo(f"API: 获取股票基础信息: {time1 - time0:.2f}s")
        click.echo(f"proc:写入股票列表: {time2 - time1:.2f}s")
    return metadata

def parse_api_kline_d1(symbol_map, start_session, end_session):
    '''
    分钟线: date,time,code,open,high,low,close,volume,amount,adjustflag
    日线:   date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,
            tradestatus,pctChg,peTTM,psTTM,pcfNcfTTM,pbMRQ,isST(特殊处理股票)
    周月线: date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg
    adjustflag: 1:后复权; 2:前复权; 3:默认不复权; (涨跌幅复权算法)
    frequency: d=日k线,w=周,m=月,5=5分钟,15=15分钟,30=30分钟,60=60分钟k线数据
    '''
    for sid, itmes in symbol_map.iterrows():
        code = itmes[0]
        first_traded = pytz.timezone("Asia/Shanghai").localize(itmes[1], is_dst=None)
        
        start_date = max(start_session, first_traded).strftime('%Y-%m-%d')
        end_date = end_session.strftime('%Y-%m-%d')
        rs = api.query_history_k_data_plus(code,
            "date,open,high,low,close,volume",
            start_date=start_date, end_date=end_date,
            frequency="d", adjustflag="1")
        if(rs.error_code!='0'):
            click.echo('query_history_k_data_plus respond error_code:'+rs.error_code)
            click.echo('query_history_k_data_plus respond  error_msg:'+rs.error_msg)
        #### 打印结果集 ####
        data_list = []
        while (rs.error_code == '0') & rs.next():
            # 获取一条记录，将记录合并在一起
            data_row = rs.get_row_data()
            data_list.append([
                pd.to_datetime(data_row[0],format='%Y-%m-%d',utc=True),
                float(data_row[1]),
                float(data_row[2]),
                float(data_row[3]),
                float(data_row[4]),
                float(data_row[5]),
            ])
        kline = pd.DataFrame(data_list, columns=['day','open','high','low','close','volume'])
        kline.set_index('day',inplace=True, drop=True)#.sort_index()
        yield sid, kline

#def parse_api_split_merge_dividend(symbol_map, start_session, end_session):
#    data_list = []
#    for sid, code, first_traded in symbol_map.iteritems():
#        # 查询每年每股除权除息信息
#        rs = api.query_dividend_data(code=code, year="2017", yearType="operate")
#        while (rs.error_code == '0') & rs.next():
#            data_row = rs.get_row_data()
#            data_list.append([
#                sid,         # list 编号
#                data_row[6], # 除权除息日期 dividOperateDate 
#            ])
#        return pd.DataFrame(data_list, columns=data_list.fields)

def parse_api_tradedate():
    #### 获取交易日信息 ####
    rs = api.query_trade_dates(start_date="1900-01-01")
    if(rs.error_code!='0'):
        click.echo('query_trade_dates respond error_code:'+rs.error_code)
        click.echo('query_trade_dates respond  error_msg:'+rs.error_msg)

    #### 打印结果集 ####
    trade_days = []
    non_trade_days = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        row_data = rs.get_row_data()
        if(row_data[1]=='1'): # 'is_trading_day'
            trade_days.append(row_data[0]) # 'calendar_date'
        else:
            non_trade_days.append(row_data[0]) # 'calendar_date'
    non_business_days_present = []
    business_days_not_present = []
    for date_str in trade_days:
        date = datetime.strptime(date_str, '%Y-%m-%d')
        if date.weekday() >= 5: # Saturday=5, Sunday=6
            non_business_days_present.append(date)
    for date_str in non_trade_days:
        date = datetime.strptime(date_str, '%Y-%m-%d')
        if date.weekday() < 5:
            business_days_not_present.append(date)
    return non_business_days_present, business_days_not_present

def get_exchange(index):    # sh.xxxxxx sz.xxxxxx
    '''
    主板, 中小板: 10%
    创业板, 科创板: 20%
    新三板: 30%
    '''
    index_items = index.split(".")
    code = index_items[1]
    if code[:2] == '60':
        exchg = "SSE.A" # 沪市主板
    elif code[:3] == '900':
        exchg = "SSE.B"
    elif code[:2] == '68':
        exchg = "SSE.STAR" # 科创板（Sci-Tech innovAtion boaRd）
    elif code[:3] in ['000', '001']:
        exchg = "SZSE.A" # 深市主板
    elif code[:3] == '200':
        exchg = "SZSE.B"
    elif code[:3] in ['300', '301']:
        exchg = "SZSE.SB" # 创业板（second-board）
    elif code[:3] in ['002', '003']:
        exchg = "SZSE.A" # 中小板（深市主板）
    elif code[:3] in ['440', '430', '830']:
        exchg = "NQ"
                        # 新三板 （National Equities Exchange and Quotations）
                        # OC(over-the-counter market) -> NQ(2021场内改革)
    else:
        exchg = "???"
        click.echo(f"Err: 不识别的股票代码：{index}")
    return exchg

def auth():
    if(API=='baostock'):
        import baostock as bs
        lg = bs.login()
        if(lg.error_code!='0'):
            click.echo('login respond error_code:'+lg.error_code)
            click.echo('login respond  error_msg:'+lg.error_msg)
        else:
            click.echo('证券宝已登入')
        api = bs
    elif(API=='tushare'):
        import tushare as ts
        pro = ts.pro_api('20231208200557-eb280087-82b0-4ac9-8638-4f96f8f4d14c')
        pro._DataApi__http_url = 'http://tsapi.majors.ltd:7000'
        api = pro
        click.echo('tushare已登入')
    return api

if __name__ == '__main__': # test if called alone
    api = auth()
    metadata = parse_api_metadata()
    symbol_map = metadata.loc[:,['symbol','first_traded']]
    parse_api_kline_d1(
        symbol_map, 
        pd.Timestamp('1900-01-01', tz=tz),
        pd.Timestamp('today', tz=tz)
        )
    special_trade_days, special_holiday_days = parse_api_tradedate() # non_business_days_present, business_days_not_present
    # A-stock has no special_trade_days
    print('special_trade_days: ',special_trade_days)
    print('special_holiday_days: ',special_holiday_days)
    
    #from pathlib import Path
    #idx = pd.IndexSlice
    #data_path = Path('~', 'trade', 'data')
    
    #if not os.getenv('ZIPLINE_ROOT'):
    #    bundle_path = Path('~', '.zipline', 'data', 'A_stock').expanduser()
    #else:
    #    bundle_path = Path(os.getenv('ZIPLINE_ROOT'), 'data', 'A_stock')
    
    #adj_db_path = bundle_path / 'adjustments.sqlite'
    #equities_db_path = bundle_path / 'assets-7.sqlite'
    #
    ## edit ~/.zipline/extension.py
    
    
    
    
    
    
    
    
    

