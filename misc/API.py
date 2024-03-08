# TUSHARE
# pro._DataApi__http_url = 'http://tsapi.majors.ltd:7000'
# 
# 积分等级	token	生效时间	失效时间
# 5000	
# 20231208200557-eb280087-82b0-4ac9-8638-4f96f8f4d14c	
# 2024-02-25 19:16:01	
# 2024-03-26 19:16:01

##先按照网页示例的代码一行不改试试##先按照网页示例的代码一行不改试试
##需要按照如下代码使用数据接口，您的token已在下方代码设置好

######## 使用方法一 ##############
# pip install tushare==1.2.89 -i https://pypi.tuna.tsinghua.edu.cn/simple
# 导入tushare
import tushare as ts
# 初始化pro接口
pro = ts.pro_api('20231208200557-eb280087-82b0-4ac9-8638-4f96f8f4d14c')
pro._DataApi__http_url = 'http://tsapi.majors.ltd:7000'

#常规接口
df1 = pro.daily(ts_code='000001.SZ', start_date='20180701', end_date='20180718')
print(df1)

#通用行情接口
df2 = ts.pro_bar(api=pro,ts_code='000001.SZ', adj='qfq', start_date='20180101', end_date='20181011')
print(df2)

#股票分钟数据接口,如果您的token不具有分钟权限,每24小时限制访问10次,之后这个接口会报错
df3 = pro.stk_mins(ts_code='600519.SH',start_time='2020-11-17 09:30:00',end_time='2020-11-17 15:00:00',freq='5min')
print(df3)
df3 = ts.pro_bar(api=pro,ts_code='600519.SH',start_date='2020-11-17 09:30:00',end_date='2020-11-17 15:00:00',freq='5min',ma=[5])
print(df3)

#沪深京历史level1 tick数据接口,如果您的token不具有此权限,每24小时限制访问10次,之后这个接口会报错
his_tick_df = pro.his_tick(ts_code='600519.SH',trade_date='20231228')
#因为tick数据量过大，目前只支持一次获取一只标的某一天的全部数据,关于字段含义的解释可访问：https://datashop.majors.ltd:8889/tick_indro
print(his_tick_df)

#沪深京实时level1 tick数据接口,如果您的token不具有此权限,每24小时限制访问10次,之后这个接口会报错
rt_tick_df = pro.rt_tick(ts_codes=['000001.SH','600519.SH','300750.SZ','510210.SH','832171.BJ','113604.SH','131810.SZ'],fields=[])
#'000001.SH',上证指数,'600519.SH',贵州茅台,'300750.SZ'宁德时代,'510210.SH'上证指数ETF,'832171.BJ'志晟信息,'113604.SH'多伦转债,'131810.SZ'一天期逆回购
rt_tick_df

######## 使用方法二 ##############
#安装或更新majorshare
# pip install -U  majorshare  -i https://pypi.tuna.tsinghua.edu.cn/simple
# 引入majorshare
import majorshare as mjs
#设置majorshare的token
mjs.set_token('20231208200557-eb280087-82b0-4ac9-8638-4f96f8f4d14c')
#实例化majorshare接口对象,tushare接口为共享带宽接口与tushare数据保持一致
mjspro = mjs.pro_api()
#常规接口
df1 = mjspro.daily(ts_code='000001.SZ', start_date='20180701', end_date='20180718')
print(df1)
#通用行情接口
df2 = mjs.pro_bar(api=mjspro,ts_code='000001.SZ', adj='qfq', start_date='20180101', end_date='20181011')
print(df2)
#其他tushare接口调用方式和方法一类似，这里不再赘述
# ……………………


# 以下为mjs接口使用方法
# 支持国内所有主流交易所：['上交所','深交所','北交所','港交所','中金所','上期所','大商所','郑商所','上海国际能源交易中心','广期所','上证期权','深证期权','板块指数']
# 获取mjs接口标的命名示例，只要您具有有效的幂级数token，常规情况mjs不对日级别以上行情数据进行限制,日级别以下数据需要根据交易所的不同分别购买
symbol_example = mjspro.get_symbol_example()
print(symbol_example)
#获取板块列表
sector_list = mjspro.get_sector_list() 
print(sector_list)
 #获取指定板块的交易标的代码列表
stock_list_in_sector = mjspro.get_stock_list_in_sector(sector_name='上证A股')  
print(stock_list_in_sector)
#获取指定交易标的基础信息
instrument_detail = mjspro.get_instrument_detail(stock_code='600519.SH') 
print(instrument_detail)
#获取指定交易所交易日历
trading_dates = mjspro.get_trading_dates(market='SH', start_time='20230101', end_time='20231231')  
print(trading_dates)
#获取指定交易标的日线级别历史行情数据，限于带宽，目前只支持一次获取一只标的一天的数据，需要分别购买不同交易所数据的权限
his_price = mjspro.get_his_price(stock_code='600519.SH', trade_date='20240105',period='1d')
print(his_price)
#获取指定交易标的分钟级别历史行情数据，限于带宽，目前只支持一次获取一只标的一天的数据，需要分别购买不同交易所数据的权限
his_mins = mjspro.get_his_price(stock_code='600519.SH', trade_date='20240105',period='1m')
print(his_mins) #16:05后可调用当日，此分钟接口支持级别： 1m,5m,15m,30m,1h,,,,,,暂不支持北交所股票
#获取指定交易标的level1 tick历史数据限于带宽，，目前只支持一次获取一只标的一天的数据，需要分别购买不同交易所数据的权限
his_ticks = mjspro.get_his_price(stock_code='600519.SH', trade_date='20231201',period='tick')
print(his_ticks) #16:05后可调用当日，暂不支持北交所股票
#获取指定交易标的level1 tick实时数据,传入列表返回多个标的，需要单独购买权限
rt_ticks = mjspro.get_rt_ticks(stock_code_list = ['600519.SH','000001.SH','000001.SZ'],field_list=[])
print(rt_ticks)

      
