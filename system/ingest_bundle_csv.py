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

# 一边ingest，一边pull API的数据是很蠢的做法，请先准备好本地csv/sql (flow解耦)
# 假装 UTC 就是 Asia/Shanghai (简化计算)，所有datetime默认 tz-naive -> tz-aware
tz = "UTC"
bundle_name = 'A_stock'
API='baostock'
assets_list = [
    '1沪A_不包括科创板',
    '2深A_不包括创业板',
    '3科创板',
    '4创业板',
    '5北A_新老三板',
    # '6上证股指期权',
    # '7深证股指期权',
]

proj_path = '/root/work/trade/'
max_num_assets = 10000 # 只ingest前n个asset
enable_profiling = True # 运行分析

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
    # print(metadata.iloc[:,:3])
    # 写入股票基础信息
    asset_db_writer.write(metadata)
    
    # 准备写入 dailybar/minutebar(lazzy iterable)
    # ('day'/'min') * ('open','high','low','close','volume')
    # minute_bar_writer.write(parse_api_kline_m5(symbol_map, start_session, end_session), show_progress=show_progress)
    daily_bar_writer.write(
        data=parse_csv_kline_d1(symbol_map, index_info, start_session, end_session), 
        show_progress=show_progress,
        invalid_data_behavior = 'raise' #{'warn', 'raise', 'ignore'}
    )
    
    # split:除权, merge:填权, dividend:除息
    # 用了后复权数据，不需要adjast factor
    #adjustment_writer.write(
    #    splits=pd.concat(splits, ignore_index=True),
    #    dividends=pd.concat(dividends, ignore_index=True),
    #)
    
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
            row[2], # 'symbol'
            row[0], # 'asset_name'
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
    progress_bar = tqdm(len(index_info))
    for sid, items in symbol_map.iterrows():
        code = items[0]
        asset_name = items[1]
        first_traded = pytz.timezone(tz).localize(items[2]) # datetime.date type
        start_date = max(start_session, first_traded).strftime('%Y-%m-%d')
        end_date = end_session.strftime('%Y-%m-%d')
        progress_bar.set_description(f'{code}, {asset_name}: ')
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
            for col in kline.columns:
                for idx, value in kline[col].items():
                    if pd.isna(value):
                        prev_value = kline_filled[col].loc[idx]
                        # 将标准输出重定向到文件
                        click.echo(f"{code}, {asset_name}: 在{col}列 {idx.strftime('%Y-%m-%d')}行 填充了 {prev_value}", file=f)
        # click.echo(f", {code}, {asset_name}; 获取kline: {time1 - time0:.2f}s, proc:{time2 - time1:.2f}s")
        progress_bar.update(1)
        yield sid, kline_filled

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
        click.echo(f"Err: 不识别的股票代码：{index}")
    return exchg

def parse_api_tradedate():
    #### 获取交易日信息 ####
    rs = api.query_trade_dates(start_date="1900-01-01")
    if(rs.error_code!='0'):
        click.echo('query_trade_dates respond error_code:'+rs.error_code)
        click.echo('query_trade_dates respond  error_msg:'+rs.error_msg)

    #### 打印结果集 ####
    trade_days = []
    non_trade_days = []
    non_business_days_present = []
    business_days_not_present = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        row_data = rs.get_row_data()
        date = datetime.strptime(row_data[0], '%Y-%m-%d')
        date = pytz.timezone(tz).localize(date, is_dst=None) # datetime: tz-naive to tz-aware
        if(row_data[1]=='1'): # 'is_trading_day'
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


# 国内市场主要包含 上海证券交易所、 深圳证券交易所、 香港证券交易所、 全国中小企业股份转让系统有限公司、 中国金融期货交易所、 上海商品期货交易所、 郑州商品期货交易所、 大连商品期货交易所等 
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
    os.system(f'cp -rf {proj_path}system/ingest_bundle_csv.py {zipline_path}/extension.py')
    os.system(f'zipline ingest -b {bundle_name}')
    
else:   # run this script as extension.py
    if enable_profiling:
        # performance profiling
        import subprocess
        PID = subprocess.check_output("pgrep 'zipline'", shell=True).decode('utf-8')
        click.echo(f'Prof: ingest PID = {PID}')
        process = subprocess.Popen(f'sudo py-spy record -o {proj_path}system/profile.svg --pid {PID}', stdin=subprocess.PIPE, shell=True)
        # process.stdin.write('bj721006'.encode())
        # process.stdin.flush()
    
    # register calendar and data bundle
    api = auth()
    trade_days, special_trade_days, special_holiday_days = parse_api_tradedate()
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
        bundle_name,
        ingest_bundle,
        "股票白盘",
        #pd.Timestamp('1900-01-01', tz=tz),
        #pd.Timestamp('today', tz=tz)
        )
    click.echo('已注册: 交易日历, 数据parse函数')