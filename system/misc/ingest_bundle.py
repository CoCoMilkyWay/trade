import numpy as np
import pandas as pd
from datetime import datetime,time
import time as t
import pytz
import os
import csv
import click
from tqdm import tqdm
from pathlib import Path
from pandas.tseries.offsets import CustomBusinessDay
import trading_calendars
from zipline.data.bundles import register

# 很蠢的做法: 一边ingest，一边pull API的数据，请先准备好本地csv/sql (flow解耦)
# 假装 UTC 就是 Asia/Shanghai (简化计算)，所有datetime默认 tz-naive -> tz-aware
tz = "UTC"
bundle_name = 'A_stock'
API='baostock'
assets_list = [
    #'1沪A_不包括科创板',
    #'2深A_不包括创业板',
    '3科创板',
    #'4创业板',
    #'5北A_新老三板',
    # '6上证股指期权',
    # '7深证股指期权',
]
data_start = '1900-01-01'
data_end = '2024-02-19'

proj_path = '/home/chuyin/trade/'
# proj_path = '/root/work/trade/'
password = True
max_num_assets = 10000 # 只ingest前n个asset
enable_profiling = False # 运行分析

from zipline.utils.run_algo import load_extensions # for zipline to work
load_extensions(
    default=True,
    extensions=[],
    strict=True,
    environ=os.environ,
)

def ingest_bundle(
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
    '''this script would automatically run by zipline, as extension.py'''
    metadata, index_info = parse_csv_metadata() # index_info = [asset_csv_path, num_lines]
    symbol_map = metadata.loc[:,['symbol','asset_name','first_traded']]
    print(metadata.iloc[:,:3].head(2))
    
    # 写入股票基础信息
    #TODO 
    asset_db_writer.write(metadata)
    
    # split:除权, merge:填权, dividend:除息
    # 用了后复权数据，不需要adjast factor
    # adjustment_writer.write(
    #     splits=parse_csv_split_merge_dividend(symbol_map, start_session, end_session)
    # )
    
    # 准备写入 dailybar/minutebar(lazzy iterable)
    # ('day'/'min') * ('open','high','low','close','volume')
    # minute_bar_writer.write(parse_api_kline_m5(symbol_map, start_session, end_session), show_progress=show_progress)
    daily_bar_writer.write(
        data=parse_csv_kline_d1(symbol_map, index_info, start_session, end_session), 
        show_progress=show_progress,
        invalid_data_behavior = 'raise' #{'warn', 'raise', 'ignore'}
    )
    
csv_path = f'{proj_path}data/'
if not os.getenv('ZIPLINE_ROOT'):
    zipline_path = Path('~', '.zipline').expanduser()
else:
    zipline_path = Path(os.getenv('ZIPLINE_ROOT'))
bundle_path = Path(zipline_path, 'data', bundle_name)
log = f'{proj_path}system/logfile.txt'
assets_path_list = [csv_path + asset for asset in assets_list]
# API parsing=================================================================================
def parse_csv_metadata(show_progress=True, type='equity'):
    extracted_rows = []
    index_info = []
    for assets_path in assets_path_list:
        assets = assets_path.split('/')[-1]
        for asset_csv_name in tqdm(os.listdir(assets_path), desc=f'{assets}'):
            if asset_csv_name.endswith('.txt'):
                asset_csv_path = os.path.join(assets_path, asset_csv_name)
                num_lines = sum(1 for line in open(asset_csv_path, encoding='gbk')) # last line
                if(num_lines<10):
                    continue
                lines_to_read = [0, 2, num_lines - 2]  # 1st, 3rd, the 2nd_last line
                lines = pd.read_csv(asset_csv_path, encoding='gbk', sep='\s+', skiprows=lambda x: x not in lines_to_read, header=None)
                if len(lines) > 2:
                    extracted_rows.append([             # line1 = [600000 浦发银行 日线 后复权]
                        asset_csv_name.split('.')[0],   # 'asset_file_name'
                        lines.iloc[0, 0],               # 'code' (-> exchange)
                        lines.iloc[0, 1],               # 'symbol'
                        lines.iloc[1, 0].split(',')[0], # 'start_date'
                        lines.iloc[2, 0].split(',')[0], # 'end_date'
                        ])
                    index_info.append([asset_csv_path, num_lines])
    
    def metadata_frame_equity():
        dtype = [
            ('symbol', 'object'),
            ('asset_name', 'object'),
            ('start_date', 'datetime64[ns]'),
            ('end_date', 'datetime64[ns]'),
            ('first_traded', 'datetime64[ns]'),
            ('auto_close_date', 'datetime64[ns]'),
            ('exchange', 'object'), ]
        return pd.DataFrame(np.empty(len(extracted_rows), dtype=dtype))
    
    if(type=='equity'):
        metadata = metadata_frame_equity()
    row_index = 0
    for row in tqdm(extracted_rows, desc='parsing metadata: '):
        start_date = pd.to_datetime(row[3], format='%Y-%m-%d', utc=True)
        end_date = pd.to_datetime(row[4], format='%Y-%m-%d', utc=True)
        first_traded = start_date.date()
        auto_close_date = end_date + pd.Timedelta(days=1)
        new_row = [
            row[0], # 'symbol'
            row[2], # 'asset_name'
            start_date, # 'start_date' (ipo date)
            end_date, # 'end_date' (data_end)
            first_traded, # 'first_traded' (data_start)
            auto_close_date, # 'auto_close_date'
            get_exchange(row[1]), # 'exchange',
        ]
        metadata.loc[row_index] = new_row
        row_index += 1
    return metadata, index_info

def parse_csv_kline_d1(symbol_map, index_info, start_session, end_session):
    progress_bar = tqdm(range(len(index_info)))
    for sid, items in symbol_map.iterrows():
        symbol = items[0]
        asset_name = items[1]
        first_traded = pytz.timezone(tz).localize(items[2]) # datetime.date type
        start_date = max(start_session, first_traded).strftime('%Y-%m-%d')
        end_date = end_session.strftime('%Y-%m-%d')
        progress_bar.set_description(f'{symbol}, {asset_name} ')
        lines_to_skip = [0, 1, index_info[sid][1] - 1]
        lines = pd.read_csv(index_info[sid][0], encoding='gbk', sep='\s+', skiprows=lambda x: x in lines_to_skip, header=None)
        kline_raw = []
        for rid, line in lines.iterrows():
            line = line[0].split(',')
            kline_raw.append([
                line[0],
                float(line[1]),
                float(line[2]),
                float(line[3]),
                float(line[4]),
                np.uint32(line[5]),  # use 成交量
                # np.nan if line[5] == '' else np.uint32(line[5]) # may overflow, or null string
            ])
        kline = pd.DataFrame(kline_raw, columns=['day','open','high','low','close','volume'])
        kline['day'] = pd.to_datetime(kline['day'], format='%Y-%m-%d', utc=True)
        kline.set_index('day',inplace=True, drop=True)#.sort_index()
        # 有可能由于bug/公司资产重组等原因，造成缺失日线数据/暂时停牌的问题（zipline会报错，对不上calendar）
        # 检查是否存在数据缺失 (有乱序风险)
        asset_trade_days = [dt for dt in trade_days if dt > first_traded]
        missing_tradedays = set(asset_trade_days) - set(kline.index.tolist())
        for missing_tradeday in missing_tradedays:
            kline.loc[missing_tradeday] = np.nan
        kline = kline.sort_index(axis=0)
        # 使用前一个非空值填充 '' (无乱序风险)
        kline.replace('', np.nan, inplace=True)
        kline_filled = kline.fillna(method='ffill')
        # 打开文件以保存日志信息
        with open(log, 'a') as f: # append new log
            for idx, items in kline.iterrows():
                NaN_column_list_for_this_row = items.index[items.isna()].tolist()
                if(NaN_column_list_for_this_row != []):
                    prev_values = kline_filled.loc[idx, NaN_column_list_for_this_row]
                    # 将标准输出重定向到文件
                    click.echo(f"{symbol}, {asset_name}: 在{NaN_column_list_for_this_row}列 {idx.strftime('%Y-%m-%d')}行 填充了 {prev_values.values}", file=f)
        progress_bar.update(1)
        yield sid, kline_filled

def parse_csv_split_merge_dividend(symbol_map, start_session, end_session):
    '''
    apply to all data before effective date:
        split: OCHL multiply, volume divide
        merge: OCHL multiply, volume unaffected
        dividend(cash/stock): specify date, calculate on the fly
    '''
    #api = auth()
    #data_list = []
    #for sid, items in symbol_map.iterrows():
    #    symbol = items[0]
    #    asset_name = items[1]
    #    first_traded = pytz.timezone(tz).localize(items[2]) # datetime.date type
    #    # 查询每年每股除权除息信息
    #    rs = api.query_dividend_data(code=asset_name, year="2017", yearType="operate")
    #    while (rs.error_code == '0') & rs.next():
    #        data_row = rs.get_row_data()
    #        data_list.append([
    #            sid,         # list 编号
    #            data_row[6], # 除权除息日期 dividOperateDate 
    #        ])
    #return pd.DataFrame(data_list, columns=data_list.fields)
    df = pd.DataFrame([[end_session, 1.00, 1]], columns=['effective_date', 'ratio', 'sid'])
    print(df)
    return df

def get_exchange(code):    # xxxxxx
    '''
    主板, 中小板: 10%
    创业板, 科创板: 20%
    新三板: 30%
    '''
    # index_items = index.split(".")
    # code = index_items[1]
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
    elif code[:3] in ['440', '430'] or code[:2] in ['83','87']:
        exchg = "NQ"
                        # 新三板 （National Equities Exchange and Quotations）
                        # OC(over-the-counter market) -> NQ(2021场内改革)
    else:
        exchg = "???"
        click.echo(f"Err: 不识别的股票代码：{code}")
    return exchg

def parse_csv_tradedate():
    # api = auth()
    # #### 获取交易日信息 ####
    # rs = api.query_trade_dates(
    #     start_date=data_start,
    #     end_date=data_end)
    # if(rs.error_code!='0'):
    #     click.echo('query_trade_dates respond error_code:'+rs.error_code)
    #     click.echo('query_trade_dates respond  error_msg:'+rs.error_msg)
    # data = []
    # while (rs.error_code == '0') & rs.next():
    #     # 获取一条记录，将记录合并在一起
    #     row_data = rs.get_row_data()
    #     data.append(row_data)
    # np.savetxt(f'{proj_path}data/trade_dates.csv', data, fmt='%s', delimiter=',')
    trade_dates = np.loadtxt(f'{proj_path}data/trade_dates.csv', delimiter=',', dtype=str)
    
    trade_days = []
    non_trade_days = []
    non_business_days_present = []
    business_days_not_present = []
    for trade_day, tradable in trade_dates:
        date = datetime.strptime(trade_day, '%Y-%m-%d')
        date = pytz.timezone(tz).localize(date, is_dst=None) # datetime: tz-naive to tz-aware
        if(tradable=='1'): # 'is_trading_day'
            trade_days.append(date) # 'calendar_date'
            if date.weekday() >= 5: # Saturday=5, Sunday=6
                non_business_days_present.append(date)
        else:
            non_trade_days.append(date) # 'calendar_date'
            if date.weekday() < 5:
                business_days_not_present.append(date)
    return trade_days, non_business_days_present, business_days_not_present

def auth():
    if(API=='baostock'):
        import baostock as bs
        lg = bs.login()
        if(lg.error_code!='0'):
            click.echo('login respond error_code:'+lg.error_code)
            click.echo('login respond  error_msg:'+lg.error_msg)
        else:
            click.echo('证券宝(Baostock)已登入')
        api = bs
    elif(API=='tushare'):
        import tushare as ts
        pro = ts.pro_api('20231208200557-eb280087-82b0-4ac9-8638-4f96f8f4d14c')
        pro._DataApi__http_url = 'http://tsapi.majors.ltd:7000'
        api = pro
        click.echo('tushare已登入')
    return api


# 国内市场主要包含: 
# 上海证券交易所、 
# 深圳证券交易所、 
# 香港证券交易所、 
# 全国中小企业股份转让系统有限公司、 
# 中国金融期货交易所、 
# 上海商品期货交易所、 
# 郑州商品期货交易所、 
# 大连商品期货交易所等 

# 上交所盘前集合竞价时间
call_auction_start = time(9, 15)	# 9：15
call_auction_end = time(9, 25)		# 9：25
# 上交所open时间
trade_start = time(9, 31)	        # 9：31
# 上海证券交易所中午休息时间
lunch_break_start = time(11, 30)	# 11：30
lunch_break_end = time(13, 0)		# 13：00
# 上交所close时间
trade_end = time(15, 0)	            # 15：00
# 上交所科创板的盘后固定交易时间
after_close_time = time(15, 30)		# 15：30
# 上海证券交易所开始正式营业时间
start_default = pd.Timestamp('1990-12-19', tz=tz)
end_default = pd.Timestamp('today', tz=tz)


if __name__ == '__main__': # register calendar and call zipline ingest
    with open(log, 'w'): pass          # clear log file
    os.system(f'rm -rf {bundle_path}') # clear data file
    #os.system(f'cp -rf {proj_path}system/ingest_bundle.py {zipline_path}/extension.py')
    os.system(f'zipline -e {proj_path}system/ingest_bundle.py ingest -b {bundle_name} ')
    
else:   # run this script as extension.py
    if enable_profiling:
        # performance profiling
        import subprocess
        PID = subprocess.check_output("pgrep 'zipline'", shell=True).decode('utf-8')
        click.echo(f'Prof: ingest PID = {PID}')
        process = subprocess.Popen(f'sudo py-spy record -o {proj_path}system/profile.svg --pid {PID}', stdin=subprocess.PIPE, shell=True)
        if password:
            process.stdin.write('bj721006'.encode())
            process.stdin.flush()
    
    # register calendar and data bundle
    trade_days, special_trade_days, special_holiday_days = parse_csv_tradedate()
    class SHSZStockCalendar(trading_calendars.TradingCalendar):
        name = "股票白盘"
        tz = pytz.timezone(tz)
        open_times = (
            (None, trade_start),
        )
        close_times = (
            (None, trade_end),
        )
        break_start_times = (
            (None, lunch_break_start),
        )
        break_end_times = (
            (None, lunch_break_end),
        )
        day = CustomBusinessDay(
            holidays=special_holiday_days,
            #calendar=regular_holidays,
            weekmask="Mon Tue Wed Thu Fri",
        )
    
    trading_calendars.register_calendar(
        '股票白盘',
        SHSZStockCalendar(),
        force = True
    )
    register(
        name=bundle_name,
        f=ingest_bundle,
        calendar_name="股票白盘",
        start_session=pd.Timestamp(data_start, tz=tz),
        end_session=pd.Timestamp(data_end, tz=tz),
        # minutes_per_day=390,
        # create_writers=True
        )
    click.echo('已注册: 交易日历, 数据parse函数')

#import numpy as np
#import pandas as pd
#from datetime import datetime,time
#import time as t
#import pytz
#import os
#import click
#from tqdm import tqdm
#from pathlib import Path
#from pandas.tseries.offsets import CustomBusinessDay
#import trading_calendars
#from zipline.data.bundles import register
#
## pretend UTC is Asia/Shanghai (easy calculation)
## tz = "Asia/Shanghai"
#tz = "UTC"
#bundle_name = 'A_stock'
#API='baostock'
#log = './system/logfile.txt'
#max_num_assets = 10000 # 只ingest前n个asset
#enable_profiling = False
#
## 国内市场主要包含 上海证券交易所、 深圳证券交易所、 香港证券交易所、 全国中小企业股份转让系统有限公司、 中国金融期货交易所、 上海商品期货交易所、 郑州商品期货交易所、 大连商品期货交易所等 
## 上交所盘前集合竞价时间
#call_auction_start = time(9, 15)	# 9：15
#call_auction_end = time(9, 25)		# 9：25
## 上交所open时间
#trade_start = time(9, 31)	        # 9：31
## 上海证券交易所中午休息时间
#lunch_break_start = time(11, 30)	# 11：30
#lunch_break_end = time(13, 0)		# 13：00
## 上交所close时间
#trade_end = time(15, 0)	            # 15：00
## 上交所科创板的盘后固定交易时间
#after_close_time = time(15, 30)		# 15：30
## 上海证券交易所开始正式营业时间
#start_default = pd.Timestamp('1990-12-19', tz=tz)
#end_default = pd.Timestamp('today', tz=tz)
#
#def ingest_bundle(
#    environ,
#    asset_db_writer,
#    minute_bar_writer,
#    daily_bar_writer,
#    adjustment_writer,
#    calendar,
#    start_session,
#    end_session,
#    cache,
#    show_progress,
#    output_dir):
#    '''this script would automatically run by zipline, as extension.py'''
#    metadata = parse_api_metadata()
#    symbol_map = metadata.loc[:,['symbol','asset_name','first_traded']]
#    print(metadata.iloc[:,:3])
#    # 写入股票基础信息
#    asset_db_writer.write(metadata)
#    
#    # 准备写入 dailybar/minutebar(lazzy iterable)
#    # ('day'/'min') * ('open','high','low','close','volume')
#    # minute_bar_writer.write(parse_api_kline_m5(symbol_map, start_session, end_session), show_progress=show_progress)
#    daily_bar_writer.write(
#        data=parse_api_kline_d1(symbol_map, start_session, end_session), 
#        show_progress=show_progress,
#        invalid_data_behavior = 'raise' #{'warn', 'raise', 'ignore'}
#    )
#    
#    # split:除权, merge:填权, dividend:除息
#    # 用了后复权数据，不需要adjast factor
#    #adjustment_writer.write(
#    #    splits=pd.concat(splits, ignore_index=True),
#    #    dividends=pd.concat(dividends, ignore_index=True),
#    #)
#    
#if not os.getenv('ZIPLINE_ROOT'):
#    zipline_path = Path('~', '.zipline').expanduser()
#else:
#    zipline_path = Path(os.getenv('ZIPLINE_ROOT'))
#bundle_path = Path(zipline_path, 'data', bundle_name)
## API parsing=================================================================================
#def parse_api_metadata(show_progress=True, type='equity'):
#    '''
#    type = 
#        1. equity(stock, funds, ETF)
#        2. fixed income(bond, debt)
#        3. derivatives(option, furture, swap)
#    '''
#    def metadata_frame_equity():
#        dtype = [
#            ('symbol', 'object'),
#            ('asset_name', 'object'),
#            ('start_date', 'datetime64[ns]'),
#            ('end_date', 'datetime64[ns]'),
#            ('first_traded', 'datetime64[ns]'),
#            ('auto_close_date', 'datetime64[ns]'),
#            ('exchange', 'object'), ]
#        return pd.DataFrame(np.empty(stock_len, dtype=dtype))
#
#    def metadata_frame_future(): # not implemented
#        pass
#
#    time0 = t.time()
#    # 获取证券基本资料
#    rs = api.query_stock_basic() # code="sh.600000" code_name="浦发银行"
#    stock_len = len(rs.data)
#    time1 = t.time()
#    '''
#    type: 1:股票 2:指数 3:其他 4:可转债(cb) 5:etf
#    status: 1:上市 0:退市
#    上市公司发行可转债:
#        不特定对象: 可公开,票面利率一般低于普通公司债(低成本融资)
#        特定对象: 不可公开
#    '''
#    if(type=='equity'):
#        metadata = metadata_frame_equity()
#    elif(type=='future'):
#        metadata = metadata_frame_future()
#
#    # 打印结果集
#    row_index = 0
#    p_bar = tqdm(range(stock_len))
#    while (rs.error_code == '0') & rs.next() and row_index < max_num_assets:
#        data_row = rs.get_row_data()
#        if(data_row[5]=='0'): # 只保留未停牌的
#            continue
#        if(data_row[4]!='1'): # 只要股票
#            continue
#        start_date = pd.to_datetime(data_row[2])
#        first_traded = start_date.date()
#        end_date = datetime.now().date()
#        auto_close_date = end_date + pd.Timedelta(days=1)
#        metadata.loc[row_index] = [
#            data_row[0], # 'symbol'
#            data_row[1], # 'asset_name'
#            start_date, # 'start_date' (ipo date)
#            end_date, # 'end_date' (data_end)
#            first_traded, # 'first_traded' (data_start)
#            auto_close_date, # 'auto_close_date'
#            get_exchange(data_row[0]), # 'exchange',
#        ]
#        p_bar.n = row_index
#        p_bar.refresh()
#        row_index += 1
#    metadata = metadata.iloc[:row_index]
#    time2 = t.time()
#    if show_progress:
#        click.echo(f"API: 获取股票基础信息: {time1 - time0:.2f}s")
#        click.echo(f"proc:写入股票列表: {time2 - time1:.2f}s")
#    return metadata
#
#def parse_api_kline_d1(symbol_map, start_session, end_session):
#    '''
#    分钟线: date,time,code,open,high,low,close,volume,amount,adjustflag
#    日线:   date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,
#            tradestatus,pctChg,peTTM,psTTM,pcfNcfTTM,pbMRQ,isST(特殊处理股票)
#    周月线: date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg
#    adjustflag: 1:后复权; 2:前复权; 3:默认不复权; (涨跌幅复权算法)
#    frequency: d=日k线,w=周,m=月,5=5分钟,15=15分钟,30=30分钟,60=60分钟k线数据
#    '''
#    for sid, itmes in symbol_map.iterrows():
#        code = itmes[0]
#        asset_name = itmes[1]
#        first_traded_no_tz = itmes[2]
#        first_traded = pytz.timezone(tz).localize(first_traded_no_tz, is_dst=None) # datetime: tz-naive to tz-aware
#        start_date = max(start_session, first_traded).strftime('%Y-%m-%d')
#        end_date = end_session.strftime('%Y-%m-%d')
#        time0 = t.time()
#        rs = api.query_history_k_data_plus(code,
#            "date,open,high,low,close,volume",
#            start_date=start_date, end_date=end_date,
#            frequency="d", adjustflag="1")
#        if(rs.error_code!='0'):
#            click.echo('query_history_k_data_plus respond error_code:'+rs.error_code)
#            click.echo('query_history_k_data_plus respond  error_msg:'+rs.error_msg)
#        time1 = t.time()
#        #### 打印结果集 ####
#        data_list = []
#        while (rs.error_code == '0') & rs.next():
#            # 获取一条记录，将记录合并在一起
#            data_row = rs.get_row_data()
#            data_list.append([
#                data_row[0],
#                float(data_row[1]),
#                float(data_row[2]),
#                float(data_row[3]),
#                float(data_row[4]),
#                np.nan if data_row[5] == '' else np.uint32(data_row[5]) # may overflow, or null string
#            ])
#        kline = pd.DataFrame(data_list, columns=['day','open','high','low','close','volume'])
#        kline['day'] = pd.to_datetime(kline['day'], format='%Y-%m-%d', utc=True)
#        kline.set_index('day',inplace=True, drop=True)#.sort_index()
#
#        # 有可能由于bug/公司资产重组等原因，造成缺失日线数据/暂时停牌的问题（zipline会报错，对不上calendar）
#        # 检查是否存在数据缺失 (有乱序风险)
#        asset_trade_days = [dt for dt in trade_days if dt > first_traded]
#        missing_tradedays = set(asset_trade_days) - set(kline.index.tolist())
#        for missing_tradeday in missing_tradedays:
#            kline.loc[missing_tradeday] = np.nan
#        kline = kline.sort_index(axis=0)
#        
#        # 使用前一个非空值填充 '' (无乱序风险)
#        kline.replace('', np.nan, inplace=True)
#        kline_filled = kline.fillna(method='ffill')
#        # 打开文件以保存日志信息
#        with open(log, 'a') as f: # append new log
#            for col in kline.columns:
#                for idx, value in kline[col].items():
#                    if pd.isna(value):
#                        prev_value = kline_filled[col].loc[idx]
#                        # 将标准输出重定向到文件
#                        click.echo(f"{code}, {asset_name}: 在{col}列 {idx.strftime('%Y-%m-%d')}行 填充了 {prev_value}", file=f)
#        time2 = t.time()
#        click.echo(f", {code}, {asset_name}; 获取kline: {time1 - time0:.2f}s, proc:{time2 - time1:.2f}s")
#        yield sid, kline_filled
#
##def parse_api_split_merge_dividend(symbol_map, start_session, end_session):
##    data_list = []
##    for sid, code, first_traded in symbol_map.iteritems():
##        # 查询每年每股除权除息信息
##        rs = api.query_dividend_data(code=code, year="2017", yearType="operate")
##        while (rs.error_code == '0') & rs.next():
##            data_row = rs.get_row_data()
##            data_list.append([
##                sid,         # list 编号
##                data_row[6], # 除权除息日期 dividOperateDate 
##            ])
##        return pd.DataFrame(data_list, columns=data_list.fields)
#
#def get_exchange(index):    # sh.xxxxxx sz.xxxxxx
#    '''
#    主板, 中小板: 10%
#    创业板, 科创板: 20%
#    新三板: 30%
#    '''
#    index_items = index.split(".")
#    code = index_items[1]
#    if code[:2] == '60':
#        exchg = "SSE.A" # 沪市主板
#    elif code[:3] == '900':
#        exchg = "SSE.B"
#    elif code[:2] == '68':
#        exchg = "SSE.STAR" # 科创板（Sci-Tech innovAtion boaRd）
#    elif code[:3] in ['000', '001']:
#        exchg = "SZSE.A" # 深市主板
#    elif code[:3] == '200':
#        exchg = "SZSE.B"
#    elif code[:3] in ['300', '301']:
#        exchg = "SZSE.SB" # 创业板（second-board）
#    elif code[:3] in ['002', '003']:
#        exchg = "SZSE.A" # 中小板（深市主板）
#    elif code[:3] in ['440', '430', '830']:
#        exchg = "NQ"
#                        # 新三板 （National Equities Exchange and Quotations）
#                        # OC(over-the-counter market) -> NQ(2021场内改革)
#    else:
#        exchg = "???"
#        click.echo(f"Err: 不识别的股票代码：{index}")
#    return exchg
#
#def parse_api_tradedate():
#    #### 获取交易日信息 ####
#    rs = api.query_trade_dates(start_date="1900-01-01")
#    if(rs.error_code!='0'):
#        click.echo('query_trade_dates respond error_code:'+rs.error_code)
#        click.echo('query_trade_dates respond  error_msg:'+rs.error_msg)
#
#    #### 打印结果集 ####
#    trade_days = []
#    non_trade_days = []
#    non_business_days_present = []
#    business_days_not_present = []
#    while (rs.error_code == '0') & rs.next():
#        # 获取一条记录，将记录合并在一起
#        row_data = rs.get_row_data()
#        date = datetime.strptime(row_data[0], '%Y-%m-%d')
#        date = pytz.timezone(tz).localize(date, is_dst=None) # datetime: tz-naive to tz-aware
#        if(row_data[1]=='1'): # 'is_trading_day'
#            trade_days.append(date) # 'calendar_date'
#            if date.weekday() >= 5: # Saturday=5, Sunday=6
#                non_business_days_present.append(date)
#        else:
#            non_trade_days.append(date) # 'calendar_date'
#            if date.weekday() < 5:
#                business_days_not_present.append(date)
#    return trade_days, non_business_days_present, business_days_not_present
#
#def auth():
#    if(API=='baostock'):
#        import baostock as bs
#        lg = bs.login()
#        if(lg.error_code!='0'):
#            click.echo('login respond error_code:'+lg.error_code)
#            click.echo('login respond  error_msg:'+lg.error_msg)
#        else:
#            click.echo('证券宝(Baostock)已登入')
#        api = bs
#    elif(API=='tushare'):
#        import tushare as ts
#        pro = ts.pro_api('20231208200557-eb280087-82b0-4ac9-8638-4f96f8f4d14c')
#        pro._DataApi__http_url = 'http://tsapi.majors.ltd:7000'
#        api = pro
#        click.echo('tushare已登入')
#    return api
#
#if __name__ == '__main__': # register calendar and call zipline ingest
#    with open(log, 'w'): pass          # clear log file
#    os.system(f'rm -rf {bundle_path}') # clear data file
#    os.system(f'cp -rf ./system/ingest_bundle.py {zipline_path}/extension.py')
#    os.system(f'zipline ingest -b {bundle_name}')
#
#    #metadata = parse_api_metadata()
#    #symbol_map = metadata.loc[:,['symbol','first_traded']]
#    #parse_api_kline_d1(
#    #    symbol_map, 
#    #    pd.Timestamp('1900-01-01', tz=tz),
#    #    pd.Timestamp('today', tz=tz)
#    #    )
#    #special_trade_days, special_holiday_days = parse_api_tradedate(api) # non_business_days_present, business_days_not_present
#    ## A-stock has no special_trade_days
#    #print('special_trade_days: ',special_trade_days)
#    #print('special_holiday_days: ',special_holiday_days)
#
#else:   # run this script as extension.py
#    if enable_profiling:
#        # performance profiling
#        import subprocess
#        PID = subprocess.check_output("pgrep 'zipline'", shell=True).decode('utf-8')
#        click.echo(f'ingest process PID: {PID}')
#        process = subprocess.Popen(f'sudo py-spy record -o ~/work/trade/system/profile.svg --pid {PID}', stdin=subprocess.PIPE, shell=True)
#        # process.stdin.write('bj721006'.encode())
#        # process.stdin.flush()
#
#    # register calendar and data bundle
#    api = auth()
#    trade_days, special_trade_days, special_holiday_days = parse_api_tradedate()
#    class SHSZStockCalendar(trading_calendars.TradingCalendar):
#        name = "股票白盘"
#        tz = pytz.timezone(tz)
#        open_times = (
#            (None, trade_start),
#        )
#        close_times = (
#            (None, trade_end),
#        )
#        break_start_times = (
#            (None, lunch_break_start),
#        )
#        break_end_times = (
#            (None, lunch_break_end),
#        )
#        day = CustomBusinessDay(
#            holidays=special_holiday_days,
#            #calendar=regular_holidays,
#            weekmask="Mon Tue Wed Thu Fri",
#        )
#
#    trading_calendars.register_calendar(
#        '股票白盘',
#        SHSZStockCalendar(),
#        force = True
#    )
#    register(
#        bundle_name,
#        ingest_bundle,
#        "股票白盘",
#        #pd.Timestamp('1900-01-01', tz=tz),
#        #pd.Timestamp('today', tz=tz)
#        )
#    click.echo('calendar, data-bundle registered')
#
#'''
#https://github.com/rainx/cn_stock_holidays/tree/main
#https://blog.csdn.net/weixin_41245990/article/details/108320577?spm=1001.2101.3001.6650.6&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-6-108320577-blog-62069281.235%5Ev43%5Econtrol&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-6-108320577-blog-62069281.235%5Ev43%5Econtrol&utm_relevant_index=7
#
## 结束日期可以超过当前年份,但是从wind得到的数据不准确,准确数据只到当年结束
#start_date = "1990-12-19"				# 1990年12月19日上海证券交易所开始正式营业
#end_base = pd.Timestamp('today', tz='UTC')
#end_date = end_base + pd.Timedelta(days=365)	# 结束日期：下一年的当天
#
#w.start()
#data = w.tdays(start_date, end_date, "Trading")	# 获取区间内交易日的日期序列
#df = pd.DataFrame(data.Data[0])
#
#start_date, end_date = df[0].iloc[[0, -1]]		# 设置开始日期和结束日期
#weekmask = 'Mon Tue Wed Thu Fri'
#cbday = CustomBusinessDay(weekmask=weekmask)		# 自定义工作日：周一至周五
#dts = pd.date_range(start_date, end_date, freq=cbday)	# 自定义工作日在这段交易日期内的时间范围
#
## 特定时段的非固定假期 = 自定义工作日在这段交易日期内的时间范围-这段交易日期内的交易日
#df_adhoc_holidays = pd.DataFrame({"date": list(set(dts.to_series()) - set(df[0]))}).sort_values('date')
## 将国内市场的非固定假期写入cn_adhoc_holidays.txt
#df_adhoc_holidays ['date'].to_csv('cn_adhoc_holidays.txt', index=False, date_format='%Y%m%d')
#
#
#    def __init__(self, start=start_default, end=end_default):
#        super(SHSZStockCalendar, self).__init__(start=start, end=end)
#    
#        # 增加午休时间
#        self._lunch_break_starts = days_at_time(_all_days, lunch_break_start, self.tz, 0)
#        self._lunch_break_ends = days_at_time(_all_days, lunch_break_end, self.tz, 0)
#
#        # 在sessions中扩展午休时间，每一个sessions中包含开盘/收盘/午休开始/午休结束
#        self.schedule = pd.DataFrame(
#            index=_all_days,
#            columns=['market_open', 'market_close', 'lunch_break_start', 'lunch_break_end'],
#            data={
#                'market_open': self._opens,
#                'market_close': self._closes,
#                'call_auction_start':self._call_auction_start,
#                'call_auction_end':self._call_auction_end,
#                'lunch_break_start': self._lunch_break_starts,
#                'lunch_break_end': self._lunch_break_ends,
#                'after_close_time': self._after_close_time
#            },
#            dtype='datetime64[ns]',
#        )
#        
#        
#	def _get_from_file(filename, use_list=False):
#        with open(filename, 'r') as f:
#            data = f.readlines()
#            if use_list:
#                return [int_to_date(str_to_int(i.rstrip('\n'))) for i in data]
#            else:
#                return set([int_to_date(str_to_int(i.rstrip('\n'))) for i in data])
#        if use_list:
#            return []
#        else:
#            return set([])
#            
#    def _get_adhoc_holidays(use_list=False):
#        data_file_path = os.path.join(os.path.dirname(__file__), 'cn_adhoc_holidays.txt')
#        return _get_from_file(data_file_path, use_list)
#    
#    @property
#    def adhoc_holidays(self):
#        return [Timestamp(t, tz=self.tz) for t in _get_adhoc_holidays(use_list=True)]
#    @lazyval
#    def _minutes_per_session(self):
#        # 上午的分钟数=午休开始时间-开盘时间
#        diff_am = self.schedule.lunch_break_start- self.schedule.market_open
#        diff_am = diff.astype('timedelta64[m]')
#		# diff_am + 1
#		
#        # 下午的分钟数=收盘时间-午休结束时间
#        diff_pm = self.schedule.market_close - self.schedule.lunch_break_end
#        diff_pm = diff.astype('timedelta64[m]')
#        # diff_pm + 1
#        return diff_am + diff_pm + 2
#    @property
#    @remember_last
#    def all_minutes(self):
#        """
#            Returns a DatetimeIndex representing all the minutes in this calendar.
#        """
#        opens_in_ns = \
#            self._opens.values.astype('datetime64[ns]')
#
#        closes_in_ns = \
#            self._closes.values.astype('datetime64[ns]')
#
#		# 扩展午休时间
#        lunch_break_start_in_ns = \
#            self._lunch_break_starts.values.astype('datetime64[ns]')
#        lunch_break_ends_in_ns = \
#            self._lunch_break_ends.values.astype('datetime64[ns]')
#
#        deltas_before_lunch = lunch_break_start_in_ns - opens_in_ns	# 上午
#        deltas_after_lunch = closes_in_ns - lunch_break_ends_in_ns	# 下午
#
#		# 扩展上午分钟线根数
#        daily_before_lunch_sizes = (deltas_before_lunch / NANOS_IN_MINUTE) + 1	
#        # 扩展下午分钟线根数
#        daily_after_lunch_sizes = (deltas_after_lunch / NANOS_IN_MINUTE) + 1、
#        
#		# 全天分钟线根数
#        daily_sizes = daily_before_lunch_sizes + daily_after_lunch_sizes
#
#        num_minutes = np.sum(daily_sizes).astype(np.int64)
#
#        all_minutes = np.empty(num_minutes, dtype='datetime64[ns]')
#
#        idx = 0
#        for day_idx, size in enumerate(daily_sizes):
#            # lots of small allocations, but it's fast enough for now.
#
#            # size is a np.timedelta64, so we need to int it
#            size_int = int(size)
#
#			# 扩展上午和下午的时间索引
#            before_lunch_size_int = int(daily_before_lunch_sizes[day_idx])
#            after_lunch_size_int = int(daily_after_lunch_sizes[day_idx])
#
#            all_minutes[idx:(idx + before_lunch_size_int)] = \
#                np.arange(
#                    opens_in_ns[day_idx],
#                    lunch_break_start_in_ns[day_idx] + NANOS_IN_MINUTE,
#                    NANOS_IN_MINUTE
#                )
#
#            all_minutes[(idx + before_lunch_size_int):(idx + size_int)] = \
#                np.arange(
#                    lunch_break_ends_in_ns[day_idx],
#                    closes_in_ns[day_idx] + NANOS_IN_MINUTE,
#                    NANOS_IN_MINUTE
#                )
#
#            idx += size_int
#        return DatetimeIndex(all_minutes).tz_localize("UTC")
#
#===========================================================================
#打开setup.py,我们在setup的entry_points中加入cn-adhoc-holidays-sync=cn_stock.get_cn_adhoc_holidays
#这样我们可以在crontab中增加cn-adhoc-holidays-sync来更新cn-adhoc-holidays.txt文件
#
#原文链接:https://blog.csdn.net/weixin_41245990/article/details/108320577
#setup(
#    name='zipline',
#    url="http://zipline.io",
#    version=versioneer.get_version(),
#    cmdclass=LazyBuildExtCommandClass(versioneer.get_cmdclass()),
#    description='A backtester for financial algorithms.',
#    entry_points={
#        'console_scripts': [
#            'zipline = zipline.__main__:main',
#            'cn-adhoc-holidays-sync=cn_stock.get_cn_adhoc_holidays',
#        ],
#    },
#
#'''