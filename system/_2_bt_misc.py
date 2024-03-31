from _1_run_backtrader import *
from _2_csv_data_parse import parse_csv_tradedate, parse_csv_metadata, parse_csv_kline_d1

def data_feed_dummy():
    datas = []
    for datafile in datafiles:
        datapath = os.path.join(modpath, f'{dataspath}/{datafile}')
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

def data_feed_SSE():
        # real SSE data
    datas = []
    start_session = pytz.timezone(TZ).localize(datetime.strptime(START, '%Y-%m-%d'))
    end_session = pytz.timezone(TZ).localize(datetime.strptime(END, '%Y-%m-%d'))
    # trade_days, special_trade_days, special_holiday_days = parse_csv_tradedate()
    metadata, index_info = parse_csv_metadata() # index_info = [asset_csv_path, num_lines]
    symbol_map = metadata.loc[:,['symbol','asset_name','first_traded']]
    print(metadata.iloc[0,:3])
        # split:除权, merge:填权, dividend:除息
        # 用了后复权数据，不需要adjast factor
        # parse_csv_split_merge_dividend(symbol_map, start_session, end_session)
        # (Date) * (Open, High, Low, Close, Volume, OpenInterest)
    for kline in parse_csv_kline_d1(symbol_map, index_info, start_session, end_session, sids):
        data = DATAFEED(dataname=kline)
        datas.append(data)
    return datas

def print_data_size(self):
    data_num = 0
    cell_len = 0
    for data in self.datas:
        data_num += 1
        tdata = sum(len(line.array) for line in data.lines)
        cell_len += tdata
    print(f'Total data memory cells ({data_num}) used: {cell_len}') # not counting indicator/observer


















