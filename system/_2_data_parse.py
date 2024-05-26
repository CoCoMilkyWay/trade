import numpy as np
import pandas as pd
from datetime import datetime
import pytz
import os
import click
from tqdm import tqdm
from pandas.tseries.offsets import CustomBusinessDay

# 很蠢的做法: 一边ingest，一边pull API的数据，请先准备好本地csv/sql (flow解耦)
# 假装 UTC 就是 Asia/Shanghai (简化计算)，所有datetime默认 tz-naive -> tz-aware
tz = "UTC"
data_start = '1900-01-01'
data_end = datetime.now().strftime('%Y-%m-%d') # '2024-02-19'

fill_missing_trade_day_data = False

API='baostock'

# path========================================================================================
modpath = os.path.dirname(os.path.abspath(__file__))
csv_path = f'{modpath}/../data/'
log = f'{modpath}/logfile.txt'

# API parsing=================================================================================
def parse_csv_metadata(cfg, show_progress=True, type='equity'):
    extracted_rows = []
    index_info = []
    assets_path_list = [csv_path + asset for asset in cfg.assets_list]
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

def parse_csv_kline_d1(symbol_map, index_info, start_session, end_session, sids):
    progress_bar = tqdm(sids)
    pending_sids = sids
    for sid, items in symbol_map.iterrows():
        if sid not in pending_sids:
            continue
        pending_sids.remove(sid)
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
                0,
            ])
        kline = pd.DataFrame(kline_raw, columns=['Date','Open','High','Low','Close','Volume', 'OpenInterest'])
        kline['Date'] = pd.to_datetime(kline['Date'], format='%Y-%m-%d', utc=True)
        kline.set_index('Date',inplace=True, drop=True)#.sort_index()
        if fill_missing_trade_day_data:
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
        else:
            kline_filled = kline
        progress_bar.update(1)
        yield kline_filled, pending_sids # use yield if call iteratively

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
    trade_dates = np.loadtxt(f'{modpath}/../data/trade_dates.csv', delimiter=',', dtype=str)
    
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

def parse_api_kline_d1(start_session, end_session):
    '''
    分钟线: date,time,code,open,high,low,close,volume,amount,adjustflag
    日线:   date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,
            tradestatus,pctChg,peTTM,psTTM,pcfNcfTTM,pbMRQ,isST(特殊处理股票)
    周月线: date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg
    adjustflag: 1:后复权; 2:前复权; 3:默认不复权; (涨跌幅复权算法)
    frequency: d=日k线,w=周,m=月,5=5分钟,15=15分钟,30=30分钟,60=60分钟k线数据
    '''
    if 'API' not in locals():
        api = auth()
    sids = ['sh.000300']
    progress_bar = tqdm(sids)
    for sid in sids:
        start_date = start_session.strftime('%Y-%m-%d')
        end_date = end_session.strftime('%Y-%m-%d')
        rs = api.query_history_k_data_plus(sid,
            "date,open,high,low,close,volume",
            start_date=start_date, end_date=end_date,
            frequency="d", adjustflag="1")
        if(rs.error_code!='0'):
            click.echo('query_history_k_data_plus respond error_code:'+rs.error_code)
            click.echo('query_history_k_data_plus respond  error_msg:'+rs.error_msg)
        kline_raw = []
        while (rs.error_code == '0') & rs.next():
            # 获取一条记录，将记录合并在一起
            line = rs.get_row_data()
            kline_raw.append([
                line[0],
                float(line[1]),
                float(line[2]),
                float(line[3]),
                float(line[4]),
                np.uint32(line[5]),  # use 成交量
                0,
            ])
        kline = pd.DataFrame(kline_raw, columns=['Date','Open','High','Low','Close','Volume', 'OpenInterest'])
        kline['Date'] = pd.to_datetime(kline['Date'], format='%Y-%m-%d', utc=True)
        kline.set_index('Date',inplace=True, drop=True)#.sort_index()
        progress_bar.update(1)
        return kline # use yield if call iteratively

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

if __name__ == '__main__':
    TZ = "UTC"
    date_now = datetime.now().strftime('%Y-%m-%d')
    START = '1999-01-01'
    END = date_now
    start_session = pytz.timezone(TZ).localize(datetime.strptime(START, '%Y-%m-%d'))
    end_session = pytz.timezone(TZ).localize(datetime.strptime(END, '%Y-%m-%d'))
    parse_api_kline_d1(start_session, end_session)
