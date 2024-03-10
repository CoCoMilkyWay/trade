import baostock as bs
import pandas as pd
import datetime
import sys
 
 
def get_stock_list(date=None):
    """
    获取指定日期的A股代码列表
    若参数date为空，则返回最近1个交易日的A股代码列表
    若参数date不为空，且为交易日，则返回date当日的A股代码列表
    若参数date不为空，但不为交易日，则打印提示非交易日信息，程序退出
    :param date: 日期
    :return: A股代码的列表
    """
 
    bs.login()
 
    stock_df = bs.query_all_stock(date).get_data()
    print(stock_df)
 
    # 如果获取数据长度为0，表示日期date非交易日
    if 0 == len(stock_df):
 
        # 如果设置了参数date，则打印信息提示date为非交易日
        if date is not None:
            print('当前选择日期为非交易日或尚无交易数据，请设置date为历史某交易日日期')
            sys.exit(0)
 
        # 未设置参数date，则向历史查找最近的交易日，当获取股票数据长度非0时，即找到最近交易日
        delta = 1
        while 0 == len(stock_df):
            stock_df = bs.query_all_stock(datetime.date.today() - datetime.timedelta(days=delta)).get_data()
            delta += 1
 
    bs.logout()
 
    # 筛选股票数据，上证和深证股票代码在sh.600000与sz.39900之间
    stock_df = stock_df[(stock_df['code'] >= 'sh.600000') & (stock_df['code'] < 'sz.399000')]
 
    # 返回股票列表
    return stock_df['code'].tolist()
 
 
def download_stock(code, start, end, freq="d", adjust="2"):
    if freq == "d":
        fields = "date,open,high,low,close,preclose,volume,amount,turn,pctChg"
    elif freq in ["m", "w"]:
        fields = "date,open,high,low,close,volume,amount,turn,pctChg"
    else:
        fields = "time, code,open,high,low,close,volume,amount"
    rs = bs.query_history_k_data_plus(
        code,
        fields,
        start_date=start, end_date=end,
        frequency=freq, adjustflag=adjust)
    # 打印结果集
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
 
    # 结果集输出到csv文件
    if freq in ["5", "15", "30", "60"]:
        result["time"] = [t[:-3] for t in result["time"]]
        result["time"] = pd.to_datetime(result["time"])
        result = result.loc[:, ['time', 'open', 'high', 'low', 'close', 'volume', 'amount']]
        result.rename(columns={'time': 'datetime'}, inplace=True)
        result.set_index("datetime", drop=True, inplace=True)
        result.to_csv(
            "I:\\baostock\\stock_datas\\stock_download\\minute\\" + code + "_" + freq + ".csv")
    elif freq == "d":
        result.set_index("date", drop=True, inplace=True)
        result.to_csv("I:\\baostock\\stock_datas\\stock_download\\day\\" + code + ".csv")
    elif freq == "m":
        result.set_index("date", drop=True, inplace=True)
        result.to_csv(
            "I:\\baostock\\stock_datas\\stock_download\\month\\" + code + ".csv")
    elif freq == "w":
        result.set_index("date", drop=True, inplace=True)
        result.to_csv(
            "I:\\baostock\\stock_datas\\stock_download\\week\\" + code + ".csv")
    else:
        print("freq 错误")
 
 
if __name__ == "__main__":
    stockList = 'stock_list'
    start = "1990-01-01"
    end = datetime.datetime.today().strftime("%Y%m%d")
    list_data = get_stock_list()
 
 
    bs.login()
    for freq in ["d", "w", "m"]:
        for i in range(0, len(list_data)):
            download_stock(code=list_data[i], start=start, end=end, freq=freq)
            print(li1[i])
 
    for freq in ["30", "60"]:
        for i in range(0, len(list_data)):
            download_stock(code=list_data[i], start=start, end=end, freq=freq)
            print(li1[i])
    bs.logout()