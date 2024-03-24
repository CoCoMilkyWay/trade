=============================================================================

# alpha:
一级指标	二级指标	含义
财务类因子	估值因子	主要包括净利润市值比（市盈率倒数）、现金流市值比（市现率倒数）、净资产市值比（市净率倒数）、股息率等指标，用于反映公司盈利、投资回收、分红收益等方面的能力。
财务类因子	规模因子	规模因子从总市值、流通市值、总资产三个方面来衡量大盘规模、流通规模和公司资产规模。
财务类因子	成长因子	是指过去一段时间内股票的各项估值指标或收益指标的增长性因子。
财务类因子	质量因子/盈利能力因子	用于衡量公司整体盈利能力和营运能力的因子，如净资产收益率、资产回报率。
财务类因子	杠杆因子	用于衡量公司整体运行的负债与权益的配比情况的因子，常用于表示公司的债务压力和偿债能力，如资产负债率、流动比率、速动比率。
行情类因子	技术因子	技术因子是由技术指标构造的因子。
行情类因子	动量因子	动量因子是指过去一段时间内股票动量和反转的效应强弱的因子。
行情类因子	风险因子	风险因子主要是利用波动率、方差、标准差等风险衡量指标衡量了公司在收益、损失、成交量等方面的波动情况，值越大，说明风险越大。
行情类因子	流动性因子	流动性因子是指过去一段时间内股票的的换手率等指标。通过改变所取的时间区间的长度，可以观察到不同时长下的股票流动性对股价影响的效应强弱。
预期类因子	盈利预测因子	盈利预测因子是在不同时间段上对净利润、收益、营业收入及其变化率的一致预测，体现了市场预期。
其他	...	...

# textbook
    beginner:
        systematic trading -robert carver ***
        trading systems and methods -perry kaufman *
        advances in financial machine learning -marcos lopez ***
    risk managements:
        the leverage space trading model **
        mathematics of money management -ralph vince
    indicators:
        rocket science for trades -john ehlers
        cybernetic analysis for stocks and futures -john ehlers
        cycle analytics for traders -john ehlers ***
        statistically sound indicators for financial market prediction -timothy masters **
    stratergies:
        the universal tatics of successful trend trading -brent penfold
        stock on the move -andrew clenow
        cybernetic trading strategies -murray ruggireo
    system dev:
        testing and tuning market trading algorithms -timothy masters **
        permutation and randomization tests for trading system developments -timothy masters 
**
    misc:
        numerical recipes -william press
        assessing and improving prediction and classification -timothy masters
        data-driven science and engineering -steve brunton
    niche:
        technical analysis for algorithm pattern recognition -prodromos
        detecting regime change in computational finance -jun chen
        trading on sentiment -richard peterson

# misc
https://cloud.tencent.com/developer/article/1457661
https://github.com/jiangtiantu/factorhub
https://qsdoc.readthedocs.io/zh-cn/latest/index.html
https://www.windquant.com/qntcloud
https://bigquant.com/trading/list
https://www.kaggle.com/competitions/ubiquant-market-prediction/leaderboard
https://www.kaggle.com/competitions/g-research-crypto-forecasting/leaderboardv
https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/leaderboard
https://www.bilibili.com/video/BV18T4m1S7fx/?spm_id_from=333.1007.tianma.1-2-2.click&
vd_source=f78f53fd28f7a2e2c81dfd10d4ab858c


资源目录
97个用于研究和实际交易的库和包
机构和学术界描述的696项战略
55本适合初学者和专业人士的书籍
23个视频和采访
还有一些博客和课程
库和包

回测与实盘交易
一般 - 事件驱动框架
一般 - 基于矢量的框架
加密货币


交易机器人
分析工具
指标
绩效测量计算
优化
定价
风险


经纪人API
数据源
一般
加密货币


数据科学
数据库
图形计算
机器学习
时间序列分析
可视化


策略
债券、商品、货币、股票
债券、商品、股票、REITs
债券、股票
债券、股票、REITs
商品
加密货币
货币
股票


书籍
初学者
传记
编码
加密币
一般
高频交易
机器学习


视频
博客
课程
库和包
97个实现交易机器人、回溯测试器、指标、定价器等的库和包列表。每个库都按其编程语言分类，并按人口降序排列（星星的数量）。

回测和实盘交易
一般 - 事件驱动框架
名字	描述
vnpy	基于Python的开源量化交易系统开发框架，于2015年1月正式发布，已经一步步成长为一个全功能的量化交易平台。
zipline	Zipline是一个Pythonic算法交易库。它是一个事件驱动的系统，用于回溯测试。
backtrader	事件驱动的Python交易策略回测库
QUANTAXIS	QUANTAXIS 支持任务调度 分布式部署的 股票/期货/期权/港股/虚拟货币 数据/回测/模拟/交易/可视化/多账户 纯本地量化解决方案
QuantConnect	QuantConnect的精益算法交易引擎（Python，C#）。
Rqalpha	一个可扩展、可替换的Python算法回测和交易框架，支持多种证券
finmarketpy	用于回测交易策略和分析金融市场的Python库（前身为pythalesians）。
backtesting.py	Backtesting.py是一个Python框架，用于根据历史（过去）数据推断交易策略的可行性。Backtesting.py在Backtrader的基础上进行了改进，并以各种方式超越了其他可获得的替代方案，Backtesting.py是轻量级的、快速的、用户友好的、直观的、互动的、智能的，并希望是面向未来的。
zvt	模块化的量化框架
WonderTrader	WonderTrader——量化研发交易一站式框架
nautilus_trader	一个高性能的算法交易平台和事件驱动的回测器
PandoraTrader	基于c++开发，支持多种交易API，跨平台的高频量化交易平台
aat	一个异步的、事件驱动的框架，用于用python编写算法交易策略，并可选择用C++进行加速。它的设计是模块化和可扩展的，支持各种工具和策略，在多个交易所之间进行实时交易。
sdoosa-algo-trade-python	这个项目主要是为那些有兴趣学习使用python解释器编写自己的交易算法的algo交易新手准备的。
lumibot	一个非常简单而有用的回溯测试和基于样本的实时交易框架（运行速度有点慢......）。
quanttrader	在Python中进行回测和实时交易。基于事件。类似于backtesting.py。
gobacktest	事件驱动的回溯测试框架的Go实现
FlashFunk	Rust中的高性能运行时
一般 - 基于矢量的框架
名字	描述
vectorbt	vectorbt采取了一种新颖的回测方法：它完全在pandas和NumPy对象上运行，并由Numba加速，以速度和规模分析任何数据。这允许在几秒钟内对成千上万的策略进行测试。
pysystemtrade	罗布-卡弗的《系统交易》一书中的python系统交易
bt	基于Algo和策略树的Python的灵活回测
加密货币
名字	描述
Freqtrade	Freqtrade是一个用Python编写的免费和开源的加密货币交易机器人。它被设计为支持所有主要交易所，并通过Telegram进行控制。它包含回测、绘图和资金管理工具，以及通过机器学习进行策略优化。
Jesse	Jesse是一个先进的加密货币交易框架，旨在简化研究和定义交易策略。
OctoBot	用于TA、套利和社会交易的加密货币交易机器人，具有先进的网络界面
Kelp	Kelp是一个免费和开源的交易机器人，适用于Stellar DEX和100多个集中式交易所
openlimits	一个Rust高性能的加密货币交易API，支持多个交易所和语言封装器。
bTrader	Binance的三角套利交易机器人
crypto-crawler-rs	抓取加密货币交易所的订单簿和交易信息
Hummingbot	一个用于加密货币做市的客户
cryptotrader-core	简单的使用Rust中的加密货币交易所REST API客户端。
交易机器人
交易机器人和阿尔法模型。其中一些是旧的，没有维护。

名字	描述
Blackbird	黑鸟比特币套利：市场中立的多/空策略
bitcoin-arbitrage	比特币套利 - 机会检测器
ThetaGang	ThetaGang是一个用于收集资金的IBKR机器人
czsc	缠中说禅技术分析工具；缠论；股票；期货；Quant；量化交易
R2 Bitcoin Arbitrager	R2 Bitcoin Arbitrager是一个由Node.js + TypeScript驱动的自动套利交易系统。
analyzingalpha	实施简单的战略
PyTrendFollow	PyTrendFollow - 使用趋势跟踪的系统性期货交易
分析工具
指标
预测未来价格走势的指标库。

名字	描述
ta-lib	对金融市场数据进行技术分析
pandas-ta	Pandas TA利用pandas软件包的130多个指标和实用功能以及60多个TA Lib蜡烛图。
finta	在Pandas中实施的常见财务技术指标
ta-rust	Rust语言的技术分析库
衡计算
财务衡量标准。

名字	描述
quantstats	用Python编写的面向量化投资人的投资组合分析方法
ffn	一个用于Python的金融函数库
优化
名字	描述
PyPortfolioOpt	在python中进行金融投资组合优化，包括经典的有效边界、Black-Litterman、分级风险平价等。
Riskfolio-Lib	Python中的投资组合优化和定量战略资产配置
empyrial	Empyrial是一个基于Python的开源量化投资库，专门为金融机构和零售投资者服务，于2021年3月正式发布。
Deepdow	连接组合优化和深度学习的Python包。它的目标是促进研究在一次前进过程中进行权重分配的网络。
spectre	Python中的投资组合优化和定量战略资产配置
定价
名字	描述
tf-quant-finance	谷歌为量化金融提供的高性能TensorFlow库
FinancePy	一个Python金融库，专注于金融衍生品的定价和风险管理，包括固定收益、股票、外汇和信用衍生品。
PyQL	著名定价库QuantLib的Python封装器
风险
名字	描述
pyfolio	Python中的投资组合和风险分析
经纪人API
名字	描述
ccxt	一个JavaScript / Python / PHP加密货币交易API，支持100多个比特币/altcoin交易所
Ib_insync	用于交互式经纪人的Python同步/async框架。
Coinnect	Coinnect是一个Rust库，旨在通过REST API提供对主要加密货币交易所的完整访问。
数据源
一般
名字	描述
OpenBB Terminal	为每个人、在任何地方进行投资研究。
TuShare	TuShare是一个用于抓取中国股票历史数据的工具。
yfinance	yfinance提供了一个线程和Pythonic方式，从雅虎金融下载市场数据。
AkShare	AKShare是一个优雅而简单的Python金融数据接口库，它是为人类而建的！它是为人类服务的。
pandas-datareader	为pandas提供最新的远程数据访问，适用于多个版本的pandas。
Quandl	通过一个免费的API，从数百个出版商那里获得数以百万计的金融和经济数据集。
findatapy	findatapy创建了一个易于使用的Python API，使用统一的高级接口从许多来源下载市场数据，包括Quandl、彭博、雅虎、谷歌等。
Investpy	用Python从http://Investing.com提取金融数据
Fundamental Analysis Data	完整的基本面分析软件包能够收集20年的公司简介、财务报表、比率和20,000多家公司的股票数据。
Wallstreet	华尔街。实时股票和期权工具
加密货币
名字	描述
Cryptofeed	使用Asyncio的加密货币交易所Websocket数据源处理程序
Gekko-Datasets	Gekko交易机器人数据集转储。下载和使用SQLite格式的历史文件。
CryptoInscriber	一个实时的加密货币历史交易数据图谱。从任何加密货币交易所下载实时历史交易数据。
数据科学
名字	描述
TensorFlow	Python中科学计算的基本算法
Pytorch	Python中的张量和动态神经网络具有强大的GPU加速功能
Keras	最具用户友好性的Python中的人类深度学习
Scikit-learn	Python中的机器学习
Pandas	灵活而强大的Python数据分析/操作库，提供类似于R data.frame对象的标记数据结构、统计函数以及更多。
Numpy	用Python进行科学计算的基本包
Scipy	Python中科学计算的基本算法
PyMC	Python中的概率编程。用Aesara进行贝叶斯建模和概率机器学习
Cvxpy	一种用于凸优化问题的Python嵌入式建模语言。
数据库
名字	描述
Marketstore	金融时序数据的DataFrame服务器
Tectonicdb	Tectonicdb是一个快速、高度压缩的独立数据库和流媒体协议，用于订单簿上的点子。
ArcticDB (Man Group)	用于时间序列和tick数据的高性能数据存储
机器学习
名字	描述
QLib (Microsoft)	Qlib是一个以人工智能为导向的量化投资平台，旨在实现人工智能技术在量化投资中的潜力，授权研究，并创造价值。通过Qlib，你可以轻松尝试你的想法，创造更好的量化投资策略。越来越多的SOTA量化研究作品/论文在Qlib中发布。
FinRL	FinRL是第一个开源框架，展示了在量化金融中应用深度强化学习的巨大潜力。
MlFinLab (Hudson & Thames)	MlFinLab通过提供可重复的、可解释的和易于使用的工具，帮助那些希望利用机器学习的力量的投资组合经理和交易者。
TradingGym	交易和回测环境，用于训练强化学习代理或简单的规则基础算法。
Stock Trading Bot using Deep Q-Learning	使用深度Q-学习的股票交易机器人
时间序列分析
名字	描述
Facebook Prophet	对具有线性或非线性增长的多季节性的时间序列数据产生高质量的预测的工具。
statsmodels	Python模块，允许用户探索数据，估计统计模型，并进行统计测试。
tsfresh	从时间序列中自动提取相关特征。
pmdarima	一个统计库，旨在填补Python时间序列分析能力的空白，包括相当于R的auto.arima函数。
视觉化
名字	描述
D-Tale (Man Group)	D-Tale是Flask后端和React前端的结合，为你带来查看和分析Pandas数据结构的简单方法。
mplfinance	使用Matplotlib实现金融市场数据可视化
btplotting	btplotting为回测、优化结果和backtrader的实时数据提供绘图。
交易策略
696篇描述原始系统交易策略的学术论文列表。每种策略按其资产类别分类，并按夏普比率降序排列。

策略现在托管在 这里：

债券策略 (7)](https://edarchimbaud.com/bonds)
商品策略 (50)](https://edarchimbaud.com/commodities)
加密货币策略 (12)](https://edarchimbaud.com/cryptocurrencies)
货币策略 (67)
股票策略 (471)](https://edarchimbaud.com/equities)
期权策略 (8)](https://edarchimbaud.com/options)
债券/商品/货币/股票策略 (22)](https://edarchimbaud.com/bonds-commodities-currencies-equities)
债券/商品/股票策略 (6)
债券/商品/股票/房地产投资信托策略 (6)
债券/股票策略 (13)
债券/股票/房地产投资信托策略 (6)](https://edarchimbaud.com/bonds-equities-reits)
商品/股票策略 (3)](https://edarchimbaud.com/commodities-equities)
股票/期权策略 (24)](https://edarchimbaud.com/equities-options)
股票/房地产投资信托策略 (1)](https://edarchimbaud.com/equities-reits)
上一个策略列表：

债券、商品、货币、股票
标题	夏普比率	挥发性	重新平衡	实施	来源
时间序列动量效应	0.576	20.5%	月度	QuantConnect	纸张
利用期货进行短期反转	-0.05	12.3%	每周	QuantConnect	纸张
债券、商品、股票、REITs
标题	夏普比率	挥发性	重新平衡	实施	来源
资产类别的趋势跟踪	0.502	10.4%	月度	QuantConnect	纸张
动量资产配置策略	0.321	11%	月度	QuantConnect	纸张
债券、股票
标题	夏普比率	挥发性	重新平衡	实施	来源
成对切换	0.691	9.5%	季度	QuantConnect	纸张
FED模式	0.369	14.3%	月度	QuantConnect	纸张
债券、股票、REITs
标题	夏普比率	挥发性	重新平衡	实施	来源
各类资产的价值和动量因素	0.155	9.8%	月度	QuantConnect	纸张
商品
标题	夏普比率	挥发性	重新平衡	实施	来源
商品中的偏度效应	0.482	17.7%	月度	QuantConnect	纸张
商品期货的收益不对称效应	0.239	13.4%	月度	QuantConnect	纸张
商品的动量效应	0.14	20.3%	月度	QuantConnect	纸张
商品的期限结构效应	0.128	23.1%	月度	QuantConnect	纸张
交易WTI/BRENT价差	-0.199	11.6%	每日	QuantConnect	纸张
加密货币
标题	夏普比率	挥发性	重新平衡	实施	来源
比特币的隔夜季节性	0.892	20.8%	日内交易	QuantConnect	纸张
加密货币的再平衡溢价	0.698	27.5%	每日	QuantConnect	纸张
货币
标题	夏普比率	挥发性	重新平衡	实施	来源
外汇套利交易	0.254	7.8%	月度	QuantConnect	纸张
美元套利交易	0.113	5.8%	月度	QuantConnect	纸张
货币动量因素	-0.01	6.7%	月度	QuantConnect	纸张
货币价值因素--PPP战略	-0.103	5%	季度	QuantConnect	纸张
股票
标题	夏普比率	挥发性	重新平衡	实施	来源
资产增长效应	0.835	10.2%	每年一次	QuantConnect	纸张
股票的短期反转效应	0.816	21.4%	每周	QuantConnect	纸张
盈利期间的逆转-公告	0.785	25.7%	每日	QuantConnect	纸张
规模因素--小市值股票溢价	0.747	11.1%	每年一次	QuantConnect	纸张
股票中的低波动因素效应	0.717	11.5%	月度	QuantConnect	纸张
如何使用公司文件的词汇密度	0.688	10.4%	月度	QuantConnect	纸张
波动性风险溢价效应	0.637	13.2%	月度	QuantConnect	纸张
与股票的配对交易	0.634	8.5%	每日	QuantConnect	纸张
原油预示着股票收益	0.599	11.5%	月度	QuantConnect	纸张
对赌股票中的贝塔系数	0.594	18.9%	月度	QuantConnect	纸张
股票中的趋势跟踪效应	0.569	15.2%	每日	QuantConnect	纸张
ESG因子动量策略	0.559	21.8%	月度	QuantConnect	纸张
价值（账面价值）因素	0.526	11.9%	月度	QuantConnect	纸张
足球俱乐部的股票套利	0.515	14.2%	每日	QuantConnect	纸张
合成贷款利率预示着随后的市场回报	0.494	13.7%	每日	QuantConnect	纸张
期权到期周效应	0.452	5%	每周	QuantConnect	纸张
分散交易	0.432	8.1%	月度	QuantConnect	纸张
共同基金回报的势头	0.414	13.6%	季度	QuantConnect	纸张
扇形动量--旋转系统	0.401	14.1%	月度	QuantConnect	纸张
结合智能因素的势头和市场组合	0.388	8.2%	月度	QuantConnect	纸张
股票的动量和反转与波动效应的结合	0.375	17%	月度	QuantConnect	纸张
市场情绪和一夜之间的反常现象	0.369	3.6%	每日	QuantConnect	纸张
一月的晴雨表	0.365	7.4%	月度	QuantConnect	纸张
研发支出和股票收益	0.354	8.1%	每年一次	QuantConnect	纸张
价值因素 - 国家内部的CAPE效应	0.351	20.2%	每年一次	QuantConnect	纸张
股票收益横截面的12个月周期	0.34	43.7%	月度	QuantConnect	纸张
股票指数的月度转折	0.305	7.2%	每日	QuantConnect	纸张
发薪日反常现象	0.269	3.8%	每日	QuantConnect	纸张
利用国家ETF进行对价交易	0.257	5.7%	每日	QuantConnect	纸张
剩余动量系数	0.24	9.7%	月度	QuantConnect	纸张
盈利公告溢价	0.192	3.7%	月度	QuantConnect	纸张
股票内部的ROA效应	0.155	8.7%	月度	QuantConnect	纸张
股票的52周高点效应	0.153	19%	月度	QuantConnect	纸张
结合基本面FSCORE和股票短期逆转的情况	0.153	17.6%	月度	QuantConnect	纸张
对抗国际股票中的贝塔系数的赌注	0.142	9.1%	月度	QuantConnect	纸张
一贯的动力策略	0.128	28.8%	6个月	QuantConnect	纸张
空头利息效应--多空版本	0.079	6.6%	月度	QuantConnect	纸张
动量因素与资产增长效应相结合	0.058	25.1%	月度	QuantConnect	纸张
股票中的动量因素效应	-0.008	21.8%	月度	QuantConnect	纸张
动量因素和风格轮换效应	-0.056	10%	月度	QuantConnect	纸张
盈利公告与股票回购的结合	-0.16	0.1%	每日	QuantConnect	纸张
盈利质量因素	-0.18	28.7%	每年一次	QuantConnect	纸张
应计项目的异常情况	-0.272	13.7%	每年一次	QuantConnect	纸张
ESG、价格动量和随机优化	N/A	N/A	月度		纸张
公司申报和股票回报的正相似性	N/A	N/A	月度		纸张
书籍
为量化交易者提供的55本书的综合清单。

初学者
标题
A Beginner’s Guide to the Stock Market: Everything You Need to Start Making Money Today - Matthew R. Kratter
How to Day Trade for a Living: A Beginner’s Guide to Trading Tools and Tactics, Money Management, Discipline and Trading Psychology - Andrew Aziz
The Little Book of Common Sense Investing: The Only Way to Guarantee Your Fair Share of Stock Market Returns - John C. Bogle
Investing QuickStart Guide: The Simplified Beginner’s Guide to Successfully Navigating the Stock Market, Growing Your Wealth & Creating a Secure Financial Future - Ted D. Snow
Day Trading QuickStart Guide: The Simplified Beginner’s Guide to Winning Trade Plans, Conquering the Markets, and Becoming a Successful Day Trader - Troy Noonan
Introduction To Algo Trading: How Retail Traders Can Successfully Compete With Professional Traders - Kevin J Davey
Algorithmic Trading and DMA: An introduction to direct access trading strategies - Barry Johnson
传记
标题
My Life as a Quant: Reflections on Physics and Finance - Emanuel Derman
How I Became a Quant: Insights from 25 of Wall Street’s Elite: - Barry Schachter
量化编程
标题
Python for Finance: Mastering Data-Driven Finance - Yves Hilpisch
Trading Evolved: Anyone can Build Killer Trading Strategies in Python - Andreas F. Clenow
Python for Algorithmic Trading: From Idea to Cloud Deployment - Yves Hilpisch
Algorithmic Trading with Python: Quantitative Methods and Strategy Development - Chris Conlan
Learn Algorithmic Trading: Build and deploy algorithmic trading systems and strategies using Python and advanced data analysis - Sebastien Donadio
加密币
标题
The Bitcoin Standard: The Decentralized Alternative to Central Banking - Saifedean Ammous
Bitcoin Billionaires: A True Story of Genius, Betrayal, and Redemption - Ben Mezrich
Mastering Bitcoin: Programming the Open Blockchain - Andreas M. Antonopoulos
Why Buy Bitcoin: Investing Today in the Money of Tomorrow - Andy Edstrom
一般
标题
The Intelligent Investor: The Definitive Book on Value Investing - Benjamin Graham, Jason Zweig
How I Invest My Money: Finance experts reveal how they save, spend, and invest - Joshua Brown, Brian Portnoy
Naked Forex: High-Probability Techniques for Trading Without Indicators - Alex Nekritin
The Four Pillars of Investing: Lessons for Building a Winning Portfolio - William J. Bernstein
Option Volatility and Pricing: Advanced Trading Strategies and Techniques, 2nd Edition - Sheldon Natenberg
The Art and Science of Technical Analysis: Market Structure, Price Action, and Trading Strategies - Adam Grimes
The New Trading for a Living: Psychology, Discipline, Trading Tools and Systems, Risk Control, Trade Management (Wiley Trading) - Alexander Elder
Building Winning Algorithmic Trading Systems: A Trader’s Journey From Data Mining to Monte Carlo Simulation to Live Trading (Wiley Trading) - Kevin J Davey
Systematic Trading: A unique new method for designing trading and investing systems - Robert Carver
Quantitative Momentum: A Practitioner’s Guide to Building a Momentum-Based Stock Selection System (Wiley Finance) - Wesley R. Gray, Jack R. Vogel
Algorithmic Trading: Winning Strategies and Their Rationale - Ernest P. Chan
Leveraged Trading: A professional approach to trading FX, stocks on margin, CFDs, spread bets and futures for all traders - Robert Carver
Trading Systems: A New Approach to System Development and Portfolio Optimisation - Emilio Tomasini, Urban Jaekle
Trading and Exchanges: Market Microstructure for Practitioners - Larry Harris
Trading Systems 2nd edition: A new approach to system development and portfolio optimisation - Emilio Tomasini, Urban Jaekle
Machine Trading: Deploying Computer Algorithms to Conquer the Markets - Ernest P. Chan
Quantitative Equity Portfolio Management: An Active Approach to Portfolio Construction and Management (McGraw-Hill Library of Investment and Finance) - Ludwig B Chincarini, Daehwan Kim
Active Portfolio Management: A Quantitative Approach for Producing Superior Returns and Controlling Risk - Richard Grinold, Ronald Kahn
Quantitative Technical Analysis: An integrated approach to trading system development and trading management - Dr Howard B Bandy
Advances in Active Portfolio Management: New Developments in Quantitative Investing - Richard Grinold, Ronald Kahn
Professional Automated Trading: Theory and Practice - Eugene A. Durenard
Algorithmic Trading and Quantitative Strategies (Chapman and Hall/CRC Financial Mathematics Series) - Raja Velu, Maxence Hardy, Daniel Nehren
Quantitative Trading: Algorithms, Analytics, Data, Models, Optimization - Xin Guo, Tze Leung Lai, Howard Shek, Samuel Po-Shing Wong
高频交易
标题
Inside the Black Box: A Simple Guide to Quantitative and High Frequency Trading - Rishi K. Narang
Algorithmic and High-Frequency Trading (Mathematics, Finance and Risk) - Álvaro Cartea, Sebastian Jaimungal, José Penalva
The Problem of HFT – Collected Writings on High Frequency Trading & Stock Market Structure Reform - Haim Bodek
An Introduction to High-Frequency Finance - Ramazan Gençay, Michel Dacorogna, Ulrich A. Muller, Olivier Pictet, Richard Olsen
Market Microstructure in Practice - Charles-Albert Lehalle, Sophie Laruelle
The Financial Mathematics of Market Liquidity - Olivier Gueant
High-Frequency Trading - Maureen O’Hara, David Easley, Marcos M López de Prado
机器学习
标题
Dark Pools: The rise of A.I. trading machines and the looming threat to Wall Street - Scott Patterson
Advances in Financial Machine Learning - Marcos Lopez de Prado
Machine Learning for Algorithmic Trading: Predictive models to extract signals from market and alternative data for systematic trading strategies with Python, 2nd Edition - Stefan Jansen
Machine Learning for Asset Managers (Elements in Quantitative Finance) - Marcos M López de Prado
Machine Learning in Finance: From Theory to Practice - Matthew F. Dixon, Igor Halperin, Paul Bilokon
Artificial Intelligence in Finance: A Python-Based Guide - Yves Hilpisch
Algorithmic Trading Methods: Applications Using Advanced Statistics, Optimization, and Machine Learning Techniques - Robert Kissell
视频
标题
Krish Naik - Machine learning tutorials and their Application in Stock Prediction
QuantInsti Youtube - webinars about Machine Learning for trading
Siraj Raval - Videos about stock market prediction using Deep Learning
Quantopian - Webinars about Machine Learning for trading
Sentdex - Machine Learning for Forex and Stock analysis and algorithmic trading
QuantNews - Machine Learning for Algorithmic Trading 3 part series
Sentdex - Python programming for Finance (a few videos including Machine Learning)
Chat with Traders EP042 - Machine learning for algorithmic trading with Bert Mouler
Tucker Balch - Applying Deep Reinforcement Learning to Trading
Ernie Chan - Machine Learning for Quantitative Trading Webinar
Chat with Traders EP147 - Detective work leading to viable trading strategies with Tom Starke
Chat with Traders EP142 - Algo trader using automation to bypass human flaws with Bert Mouler
Master Thesis presentation, Uni of Essex - Analyzing the Limit Order Book, A Deep Learning Approach
Howard Bandy - Machine Learning Trading System Development Webinar
Chat With Traders EP131 - Trading strategies, powered by machine learning with Morgan Slade
Chat with Traders Quantopian 5 - Good Uses of Machine Learning in Finance with Max Margenot
Hitoshi Harada, CTO at Alpaca - Deep Learning in Finance Talk
Better System Trader EP028 - David Aronson shares research into indicators that identify Bull and Bear markets.
Prediction Machines - Deep Learning with Python in Finance Talk
Better System Trader EP064 - Cryptocurrencies and Machine Learning with Bert Mouler
Better System Trader EP023 - Portfolio manager Michael Himmel talks AI and machine learning in trading
Better System Trader EP082 - Machine Learning With Kris Longmore
博客
标题
AAA Quants, Tom Starke Blog
AI & Systematic Trading
Blackarbs blog
Hardikp, Hardik Patel blog
Max Dama on Automated Trading
Medallion.Club on Systematic Trading (FR)
Proof Engineering: The Algorithmic Trading Platform
Quantsportal, Jacques Joubert's Blog
Quantstart - Machine Learning for Trading articles
RobotWealth, Kris Longmore Blog
课程
标题
AI in Finance
AI & Systematic Trading
Algorithmic Trading for Cryptocurrencies in Python
Coursera, NYU - Guided Tour of Machine Learning in Finance
Coursera, NYU - Fundamentals of Machine Learning in Finance
Coursera, NYU - Reinforcement Learning in Finance
Coursera, NYU - Overview of Advanced Methods for Reinforcement Learning in Finance
Hudson and Thames Quantitative Research
NYU: Overview of Advanced Methods of Reinforcement Learning in Finance
Udacity: Artificial Intelligence for Trading
Udacity, Georgia Tech - Machine Learning for Trading