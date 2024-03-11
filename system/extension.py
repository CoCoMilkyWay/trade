# coding=utf-8
import numpy as np
import pandas as pd
import sys

from pathlib import Path
sys.path.append(Path('~', '.zipline').expanduser().as_posix()) # in /xx/xx format

from datetime import datetime,time
import pytz

# from WindPy import *
import trading_calendars
from trading_calendars import *
from zipline.data.bundles import register
from zipline.utils.memoize import lazyval
from bundle_injest import injest_bundle, auth, parse_api_tradedate
from pandas.tseries.offsets import CustomBusinessDay

'''
国内市场主要包含
上海证券交易所、
深圳证券交易所、
香港证券交易所、
全国中小企业股份转让系统有限公司、
中国金融期货交易所、
上海商品期货交易所、
郑州商品期货交易所、
大连商品期货交易所等
'''
# pretend UTC is Asia/Shanghai (easy calculation)
# tz = "Asia/Shanghai"
tz = "UTC"
api = auth()

# 上交所盘前集合竞价时间
call_auction_start = time(9, 15)	# 9：15
call_auction_end = time(9, 25)		# 9：25
# 上海证券交易所中午休息时间
lunch_break_start = time(11, 30)	# 11：30
lunch_break_end = time(13, 1)		# 13：00
# 上交所科创板的盘后固定交易时间
after_close_time = time(15, 30)		# 15：30
# 上海证券交易所开始正式营业时间
start_default = pd.Timestamp('1990-12-19', tz=tz)
end_default = pd.Timestamp('today', tz=tz)

class SHSZStockCalendar(TradingCalendar):
    name = "股票白盘"
    tz = pytz.timezone(tz)
    open_times = (
        (None, time(9, 31)),
    )

    close_times = (
        (None, time(15, 0)),
    )
    
    day = CustomBusinessDay(weekmask="Mon Tue Wed Thu Fri")

trading_calendars.register_calendar(
        '股票白盘',
        SHSZStockCalendar(),
        force = True
)

register(
    'A_stock',
    injest_bundle,
    "股票白盘",
    #pd.Timestamp('1900-01-01', tz=tz),
    #pd.Timestamp('2024-01-03', tz=tz),
    #pd.Timestamp('today', tz=tz)
    )

'''
https://github.com/rainx/cn_stock_holidays/tree/main
https://blog.csdn.net/weixin_41245990/article/details/108320577?spm=1001.2101.3001.6650.6&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-6-108320577-blog-62069281.235%5Ev43%5Econtrol&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-6-108320577-blog-62069281.235%5Ev43%5Econtrol&utm_relevant_index=7

# 结束日期可以超过当前年份,但是从wind得到的数据不准确,准确数据只到当年结束
start_date = "1990-12-19"				# 1990年12月19日上海证券交易所开始正式营业
end_base = pd.Timestamp('today', tz='UTC')
end_date = end_base + pd.Timedelta(days=365)	# 结束日期：下一年的当天

w.start()
data = w.tdays(start_date, end_date, "Trading")	# 获取区间内交易日的日期序列
df = pd.DataFrame(data.Data[0])

start_date, end_date = df[0].iloc[[0, -1]]		# 设置开始日期和结束日期
weekmask = 'Mon Tue Wed Thu Fri'
cbday = CustomBusinessDay(weekmask=weekmask)		# 自定义工作日：周一至周五
dts = pd.date_range(start_date, end_date, freq=cbday)	# 自定义工作日在这段交易日期内的时间范围

# 特定时段的非固定假期 = 自定义工作日在这段交易日期内的时间范围-这段交易日期内的交易日
df_adhoc_holidays = pd.DataFrame({"date": list(set(dts.to_series()) - set(df[0]))}).sort_values('date')
# 将国内市场的非固定假期写入cn_adhoc_holidays.txt
df_adhoc_holidays ['date'].to_csv('cn_adhoc_holidays.txt', index=False, date_format='%Y%m%d')


    def __init__(self, start=start_default, end=end_default):
        super(SHSZStockCalendar, self).__init__(start=start, end=end)
    
        # 增加午休时间
        self._lunch_break_starts = days_at_time(_all_days, lunch_break_start, self.tz, 0)
        self._lunch_break_ends = days_at_time(_all_days, lunch_break_end, self.tz, 0)

        # 在sessions中扩展午休时间，每一个sessions中包含开盘/收盘/午休开始/午休结束
        self.schedule = pd.DataFrame(
            index=_all_days,
            columns=['market_open', 'market_close', 'lunch_break_start', 'lunch_break_end'],
            data={
                'market_open': self._opens,
                'market_close': self._closes,
                'call_auction_start':self._call_auction_start,
                'call_auction_end':self._call_auction_end,
                'lunch_break_start': self._lunch_break_starts,
                'lunch_break_end': self._lunch_break_ends,
                'after_close_time': self._after_close_time
            },
            dtype='datetime64[ns]',
        )
        
        
	def _get_from_file(filename, use_list=False):
        with open(filename, 'r') as f:
            data = f.readlines()
            if use_list:
                return [int_to_date(str_to_int(i.rstrip('\n'))) for i in data]
            else:
                return set([int_to_date(str_to_int(i.rstrip('\n'))) for i in data])
        if use_list:
            return []
        else:
            return set([])
            
    def _get_adhoc_holidays(use_list=False):
        data_file_path = os.path.join(os.path.dirname(__file__), 'cn_adhoc_holidays.txt')
        return _get_from_file(data_file_path, use_list)
    
    @property
    def adhoc_holidays(self):
        return [Timestamp(t, tz=self.tz) for t in _get_adhoc_holidays(use_list=True)]
    @lazyval
    def _minutes_per_session(self):
        # 上午的分钟数=午休开始时间-开盘时间
        diff_am = self.schedule.lunch_break_start- self.schedule.market_open
        diff_am = diff.astype('timedelta64[m]')
		# diff_am + 1
		
        # 下午的分钟数=收盘时间-午休结束时间
        diff_pm = self.schedule.market_close - self.schedule.lunch_break_end
        diff_pm = diff.astype('timedelta64[m]')
        # diff_pm + 1
        return diff_am + diff_pm + 2
    @property
    @remember_last
    def all_minutes(self):
        """
            Returns a DatetimeIndex representing all the minutes in this calendar.
        """
        opens_in_ns = \
            self._opens.values.astype('datetime64[ns]')

        closes_in_ns = \
            self._closes.values.astype('datetime64[ns]')

		# 扩展午休时间
        lunch_break_start_in_ns = \
            self._lunch_break_starts.values.astype('datetime64[ns]')
        lunch_break_ends_in_ns = \
            self._lunch_break_ends.values.astype('datetime64[ns]')

        deltas_before_lunch = lunch_break_start_in_ns - opens_in_ns	# 上午
        deltas_after_lunch = closes_in_ns - lunch_break_ends_in_ns	# 下午

		# 扩展上午分钟线根数
        daily_before_lunch_sizes = (deltas_before_lunch / NANOS_IN_MINUTE) + 1	
        # 扩展下午分钟线根数
        daily_after_lunch_sizes = (deltas_after_lunch / NANOS_IN_MINUTE) + 1、
        
		# 全天分钟线根数
        daily_sizes = daily_before_lunch_sizes + daily_after_lunch_sizes

        num_minutes = np.sum(daily_sizes).astype(np.int64)

        all_minutes = np.empty(num_minutes, dtype='datetime64[ns]')

        idx = 0
        for day_idx, size in enumerate(daily_sizes):
            # lots of small allocations, but it's fast enough for now.

            # size is a np.timedelta64, so we need to int it
            size_int = int(size)

			# 扩展上午和下午的时间索引
            before_lunch_size_int = int(daily_before_lunch_sizes[day_idx])
            after_lunch_size_int = int(daily_after_lunch_sizes[day_idx])

            all_minutes[idx:(idx + before_lunch_size_int)] = \
                np.arange(
                    opens_in_ns[day_idx],
                    lunch_break_start_in_ns[day_idx] + NANOS_IN_MINUTE,
                    NANOS_IN_MINUTE
                )

            all_minutes[(idx + before_lunch_size_int):(idx + size_int)] = \
                np.arange(
                    lunch_break_ends_in_ns[day_idx],
                    closes_in_ns[day_idx] + NANOS_IN_MINUTE,
                    NANOS_IN_MINUTE
                )

            idx += size_int
        return DatetimeIndex(all_minutes).tz_localize("UTC")

===========================================================================
打开setup.py，我们在setup的entry_points中加入cn-adhoc-holidays-sync=cn_stock.get_cn_adhoc_holidays
这样我们可以在crontab中增加cn-adhoc-holidays-sync来更新cn-adhoc-holidays.txt文件

原文链接：https://blog.csdn.net/weixin_41245990/article/details/108320577
setup(
    name='zipline',
    url="http://zipline.io",
    version=versioneer.get_version(),
    cmdclass=LazyBuildExtCommandClass(versioneer.get_cmdclass()),
    description='A backtester for financial algorithms.',
    entry_points={
        'console_scripts': [
            'zipline = zipline.__main__:main',
            'cn-adhoc-holidays-sync=cn_stock.get_cn_adhoc_holidays',
        ],
    },

'''