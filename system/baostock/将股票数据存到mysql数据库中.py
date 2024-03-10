"""
date:20230318
将CSV文件写入到MySQL中
"""
import pandas as pd
from sqlalchemy import create_engine
 
 
def connect_db(db):
    engine = create_engine('mysql+pymysql://hao:671010@localhost:3306/{}?charset=utf8'.format(db))
    return engine
 
 
def get_all_codes():
    # 登陆系统
    bs.login()
 
    # 获取证券信息
    rs = bs.query_all_stock(day=None)
 
    # 打印结果集
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
 
    # 结果集输出到csv文件
    result.to_csv("I:\\baostock\\stock_datas\\day\\all_stock.csv", index=False)
    print(result)
 
    # 登出系统
    bs.logout()
def create_stock(bscode, db, code, date):
    if date in ['day', 'week', 'month']:
        # 读取本地CSV文件
        df = pd.read_csv(
            'I:\\baostock\\stock_datas\\' + date + '\\{}.csv'.format(
                bscode))
    elif date in ['5', '15', '30', '60']:
        # 读取本地CSV文件
        df = pd.read_csv(
            'I:\\baostock\\stock_datas\\minute\\' + bscode + '_{}.csv'.format(
                date))
    else:
        print("date错误")
    engine = connect_db(db)
    # name='stocklist'全部小写否则会报错
    df.to_sql(name='bs_' + code + '_{}'.format(date), con=engine, index=False, if_exists='replace')
 
 
def read_csv(code):
    df0 = pd.read_csv('I:\\baostock\stock_datas\\day\\{}.csv'.format(code))
    # 筛选股票数据，上证和深证股票代码在sh.600000与sz.39900之间
    df = df0[(df0['code'] >= 'sh.600000') & (df0['code'] < 'sz.399000')]
    return df
 
 
if __name__=="__main__":
    get_all_codes()
    stockDB = 'baostock_db'
    stockList = 'all_stock'
    create_stock(bscode=stockList, db=stockDB, code=stockList, date="day")
    df1 = read_csv(stockList)
    li1 = list(df1['code'])
    # print(li1)
    for date in ["day", "week", "month", "5", "15", "30", "60"]:
        for i in range(0, len(li1)):
            codeStock = li1[i].lstrip('shzbj.')
            create_stock(bscode=li1[i], db=stockDB, code=codeStock, date=date)
            print(codeStock)