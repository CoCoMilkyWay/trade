=============================================================================

# alpha:
- 一级指标	二级指标	含义
- 财务类因子	估值因子	主要包括净利润市值比（市盈率倒数）、现金流市值比（市现率倒数）、净资产市值比- （市净率倒数）、股息率等指标，用于反映公司盈利、投资回收、分红收益等方面的能力。
- 财务类因子	规模因子	规模因子从总市值、流通市值、总资产三个方面来衡量大盘规模、流通规模和公司资产规- 模。
- 财务类因子	成长因子	是指过去一段时间内股票的各项估值指标或收益指标的增长性因子。
- 财务类因子	质量因子/盈利能力因子	用于衡量公司整体盈利能力和营运能力的因子，如净资产收益率、资产回- 报率。
- 财务类因子	杠杆因子	用于衡量公司整体运行的负债与权益的配比情况的因子，常用于表示公司的债务压力和偿- 债能力，如资产负债率、流动比率、速动比率。
- 行情类因子	技术因子	技术因子是由技术指标构造的因子。
- 行情类因子	动量因子	动量因子是指过去一段时间内股票动量和反转的效应强弱的因子。
- 行情类因子	风险因子	风险因子主要是利用波动率、方差、标准差等风险衡量指标衡量了公司在收益、损失、成- 交量等方面的波动情况，值越大，说明风险越大。
- 行情类因子	流动性因子	流动性因子是指过去一段时间内股票的的换手率等指标。通过改变所取的时间区间- 的长度，可以观察到不同时长下的股票流动性对股价影响的效应强弱。
- 预期类因子	盈利预测因子	盈利预测因子是在不同时间段上对净利润、收益、营业收入及其变化率的一致预- 测，体现了市场预期。
- 其他	...	...
- 
# textbook
-     beginner:
-         systematic trading -robert carver ***
-         trading systems and methods -perry kaufman *
-         advances in financial machine learning -marcos lopez ***
-     risk managements:
-         the leverage space trading model **
-         mathematics of money management -ralph vince
-     indicators:
-         rocket science for trades -john ehlers
-         cybernetic analysis for stocks and futures -john ehlers
-         cycle analytics for traders -john ehlers ***
-         statistically sound indicators for financial market prediction -timothy masters **
-     stratergies:
-         the universal tatics of successful trend trading -brent penfold
-         stock on the move -andrew clenow
-         cybernetic trading strategies -murray ruggireo
-     system dev:
-         testing and tuning market trading algorithms -timothy masters **
-         permutation and randomization tests for trading system developments -timothy - masters 
- **
-     misc:
-         numerical recipes -william press
-         assessing and improving prediction and classification -timothy masters
-         data-driven science and engineering -steve brunton
-     niche:
-         technical analysis for algorithm pattern recognition -prodromos
-         detecting regime change in computational finance -jun chen
-         trading on sentiment -richard peterson
- 
# misc
- https://cloud.tencent.com/developer/article/1457661
- https://github.com/jiangtiantu/factorhub
- https://qsdoc.readthedocs.io/zh-cn/latest/index.html
- https://www.windquant.com/qntcloud
- https://bigquant.com/trading/list
- https://www.kaggle.com/competitions/ubiquant-market-prediction/leaderboard
- https://www.kaggle.com/competitions/g-research-crypto-forecasting/leaderboardv
- https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/leaderboard
- https://www.bilibili.com/video/BV18T4m1S7fx/?spm_id_from=333.1007.tianma.1-2-2.click&
- vd_source=f78f53fd28f7a2e2c81dfd10d4ab858c


本文列举了一些关于高频量化交易代码项目，大部分来自Github；

包括数学/计量/统计/算法的基础教程、订单簿分析与做市策略、传统技术分析、机器学习、深度学习、强化学习等类别；

所用语言均为Python/Jupiter Notebook；





基础教程
https://github.com/crflynn/stochastic

常见随机过程的实现，包括连续、离散、扩散过程、噪声等类别；



https://github.com/jwergieluk/ou_noise

O-U过程的生成、检验和参数估计；



https://github.com/stmorse/hawkes

关于单变量以及多变量Hawkes过程的生成与参数估计，采用MAP EM算法进行参数估计；



https://github.com/AileenNielsen/TimeSeriesAnalysisWithPython

基础时间序列教程，包括时间序列数据的读取、趋势成分与季节成分的分解、谱分析、聚类等内容；



https://github.com/yangwohenmai

进阶时间序列教程，包括基于统计学、基于LSTM、基于深度学习进行时间序列预测的内容；



https://github.com/youngyangyang04/leetcode-master

数据结构与算法的刷题攻略，持续更新中；



https://github.com/dummydoo/Advanced-Algorithmic-Trading

《Advanced Algorithmic Trading》一书的代码实现，使用语言为python/R；



https://github.com/bukosabino

一位Affirm算法工程师的项目主页，内容丰富，包括TA库的实现、时间序列预测、特征工程选择等，主要集中于机器学习领域；





订单簿分析与做市策略
https://github.com/nicolezattarin/LOB-feature-analysis

对限价订单簿进行特征工程分析，包括订单大小的分布、用于价格预测的订单不平衡、知情交易的概率、波动性等方面。作者的文档与代码简洁清晰，包含部分原始文献；



https://github.com/ghgr/HFT_Bitcoin

BTC订单簿的数据分析以及一些传统高频策略的实例图示；



https://github.com/yudai-il/High-Frequency

基于level-2限价订单簿和分笔交易数据的研究，考察了订单不平衡与买卖压力的盘口拓展；



https://github.com/jeremymck/High-Frequency-Data---Limit-Order-Books

本项目包括高频数据描述性分析，Hawkes过程的生成与参数估计以及限价订单簿的模拟；



https://github.com/Macosh/Order_Book

一个订单簿模拟器，实现了创建不同类型的订单、订单匹配、模拟生成，数据库存储历史订单等功能；



https://github.com/fedecaccia/avellaneda-stoikov

Avellaneda-Stoikov做市算法的实现；



https://github.com/mdibo/Avellaneda-Stoikov

Avellaneda-Stoikov做市算法另一个实现版本，比前者更简明些；



https://github.com/jshellen/HFT

采用随机最优控制方法求解AS做市算法及其变种，包含HJB方程的求解程序以及AS做市策略的输出框架；



https://github.com/huangzz119/OptimalExecution_stochastic_control

本项目实现了Frei, C. and N. Westray (2015). Optimal execution of a vwap order: a stochastic control approach. Mathematical Finance 25(3), 612–639.一文提出的VWAP算法的最优执行，项目包括数据过程，参数校准，存货变动轨迹等；



https://github.com/kousik97/Order-Execution-Strategy

三种最优订单执行策略的实现，此外还有Almgren-Chriss框架下的市场冲击函数的实现；

包含原始文献；



https://github.com/mmargenot/machine-learning-market-maker

《Intelligent Market-Making in Artificial Financial Market》一文的实现，基于贝叶斯估计的做市策略模型；



https://github.com/armoreal/hft

高频交易策略，测试了隐马尔科夫模型（HMM）与O-U过程对限价订单簿数据的拟合情况；此外，还测试了几种典型的高频因子；





传统技术分析、对冲
https://gitee.com/xuezhihuan/my-over-sea-cloud/tree/master/quantitative_research_report

一些券商研报的复现；



https://github.com/eyeseaevan/bitmex-algo

基于BitMEX平台ETH/USDT和XBT/USDT1分钟的交易数据的交易策略，采用传统技术分析指标进行交易；



https://github.com/Davarco/AlgoBot

一个使用均值回归或趋势跟踪策略的自动交易机器人；



https://github.com/JunqiLin/High-Frequency-of-BTC-strategy

跨交易所的BTC高频对冲策略；



https://github.com/rlindland/options-market-making

基于期权市场的交易机器人，包含做市、统计套利、delta和vega对冲等；



https://github.com/Harvey-Sun/World_Quant_Alphas

World Quant 101 alphas的计算和策略化；



机器学习
https://github.com/rorysroes/SGX-Full-OrderBook-Tick-Data-Trading-Strategy

采用机器学习方法对限价订单簿动态进行建模的量化策略，包括数据获取、特征选择、模型选择，可作为机器学习类策略的baseline；



深度学习
https://blog.csdn.net/bit452/category_10569531.html

《Pytorch深度学习实践》课程对应的代码，很好的深度学习入门指引；



https://github.com/nicodjimenez/lstm

一个LSTM的简单实现；



https://github.com/rune-l/HighFrequency

采用神经网络方法预测微观层面的价格跳跃，项目完整度较高，从获取数据、异常值清洗、跳跃的统计检验到LSTM、CNN、注意力机制等方法的预测应用；



https://github.com/umeshpalai/AlgorithmicTrading-MachineLearning

用RNN，LSTM，GRU预测股价变动；





强化学习
https://github.com/BGasperov/drlformm

《Deep Reinforcement Learning for Market Making Under a Hawkes Process-Based Limit Order Book Model》一文的代码实现，基于Hawkes过程的深度强化学习做市策略；



https://github.com/lucasrea/algorithmicTrader

一个采用强化学习进行算法交易的项目；



https://github.com/gucciwang/moneyMaker

一个基于强化学习的算法交易策略；



https://github.com/TikhonJelvis/RL-book

《Foundations of Reinforcement Learning with Applications in Finance》一书的对应代码实现；



https://github.com/mfrdixon/dq-MM

Deep Q-Learning用于做市，依赖于开源项目Trading Gym；



【2023年度文章精选(链接列表)】
下面是2023年精选文章的社区贴链接，感兴趣的小伙伴可以克隆对应的文章进行学习。
点击标题可以直接进行跳转哦(序号不分先后)！

-	标题	作者
1	差不多得了	wywy1995
2	“7年40倍”策略扩容到50只	wywy1995
3	【回顾3】ETF策略之核心资产轮动	wywy1995
4	首板低开策略	wywy1995
5	尝试用机器学习批量生产小盘策略	wywy1995
6	高股息低杠杆小市值轮动策略	wywy1995
7	连板龙头策略	wywy1995
8	科技与狠活	wywy1995
9	持仓95只大容量小市值，媲美金元顺安元启	开心果
10	多因子宽基ETF择时轮动改进版-高收益大资金低回撤	养家大哥
11	低价小盘简单有效	wywy1995
12	致敬聚宽: 机器学习多因子,50只持仓,14年37倍	Gyro^.^
13	5年15倍的收益，年化79.93%，可实盘，拿走不谢！	langcheng999
14	四年多翻9倍，年化73%，成长因子评分筛选	wolfman
15	10年52倍，年化59%，全新因子方法超稳定	小白F
16	大市值价值投资，从2005年至今超额稳定	Ahfu
17	5年超额收益1055%，K线处理大阴线，无未来	hello friends
18	微盘股研究	Gyro^.^
19	高股息低市盈率高增长的价投策略	芹菜1303
20	有赚就好-小市值剥头皮改进 2年3.86%回撤	CMA
21	一种宏观数据的中长线策略，年化15%，最大回撤9%	Acne Studio
22	菜场大妈！皇城大妈！年化108%！大妈吉祥！	Clarence.罗
23	wywy1995大神机器学习策略年化提升35pt	斯科尔斯
24	凑波热闹，也来试试多因子线性回归	Plisking
25	低价股优化，18年至今10625.40%，已加防未来函数	南国草
26	【深度解析 六】高股息率-低PEG-低股价 模型	加百力
27	根据大小盘相对强弱选择合适的板块股票进行交易	Jacobb75
28	正黄旗大妈选股法，修改版，年化92%	oupian
29	高增股池小市值轮动	倚A听风雨
30	挖掘**特色的估值体系投资机会，年化150%+	子匀
31	动量因子加RSRS择时和ETF轮动每日调仓	wentaogg
32	非线性市值（非小市值）组合4只	璐璐202006
33	年化46.77，alpha191因子选股	小白F
34	韶华研究之十八，201系列	韶华不负
35	5年12倍-小市值	道尘
36	【菜场大妈】股息率小市值策略,10年206倍,5年10.8倍	120022
37	【择时模块实际效果】--论坛随便选了个策略加装	一只皮卡丘
38	ETF核心资产轮动-添油加醋	hayy
39	大资金优质股策略，总收益1217%！无小市值因子！	hello friends
40	14年初到23年7月，年化37，夏普1.25，稳健高收益！	侧耳闻鹿鸣
41	人工智能强化学习DQN交易智能体（回馈社区公开训练代码）	MarioC
42	大盘ETF动量轮动RSRS择时策略-无纳指（5年半861%）	Gerald3
43	ETF轮动：年化收益82.68%，最大回撤13.54%	小武子
44	分享一个兼顾业绩增长和PEG指标的股票策略	芹菜1303
45	[原创]价值投资+期货对冲V4.0,无惧市场大跌	Eddie79163
46	【深度解析 四】聚宽三因子基本面 模型	加百力
47	随机森林量价多因子选股 短线交易	随云浪
48	低代码迁移成本的实盘方案:jqtrade+one quant	拉姆达投资
49	年化62%的动量策略	逸_
50	新策略，一种新思路，超额376%！	hello friends
51	【复现】行业有效量价因子与行业轮动策略	Hugo2046
52	增强型投资组合优化（EPO）方法试用	开心果
53	微盘400每日再平衡	开心果
54	追市场热点策略	养家大哥
55	Ahfu的大市值价值投资加自定义邮箱推送	不如定投纳指
56	双人工智能配合，样本外夏普3.9	MarioC
57	【复现】技术分析算法、框架与实战之二：识别“圆弧底”	Hugo2046
58	融券做空,3年35倍多！已模拟2年半，动态跑盘收益惊人！	naruto
59	基于Gyro^.^大神的小市值策略的因子匹配研究	热情的刀
60	【深度解析 二】资产负债与ROA模型	加百力
61	聚类算法分析股票，看看今年大体能分出哪几类	MarioC
62	分享一种K线小碎步后突破的分钟级打法	画家
63	怎么让龟速变奔跑？以“首版突破一进二”为例	jqz1226
64	基于XGBoost_6m滚动选股策略	雨汪
65	低开买入小市值策略（剥头皮策略）3.0 总结	langcheng999
66	SVR机器学习上证综指择时中小板财务选股策略	快乐一生
67	回答wywy1995策略的一些问题	wywy1995
68	业绩预告小工具--已更新	Dobell
69	人工智能早晨十字星模型（反转形态预测）	MarioC
70	致敬市场(12)——赚钱的策略	Gyro^.^
71	小试多因子策略，年化65%	乘风万里
72	桥水 全天候策略 增加一致性度量ES 风险控制	aiyquant
73	稳健趋势策略，回测小，无未来函数	语桐
74	形态识别_单边上涨v3	伺底而动
75	【复现】个股动量效应的识别及球队硬币因子	Hugo2046
76	基金溢价（模拟效果好！）	大山深处
77	【复现】筹码分布因子	Hugo2046
78	国盛证券多因子	璇天凤舞
79	SVR选股的小市值策略	南澳
80	【深度解析 一】经典小市值模型	加百力
81	方正证券-多因子选股系列研究之四：“球队硬币”	小镇做题家MiniTAT
82	实盘模拟的小盘股策略，低频，带基本面优化以及个股风控	强势投资
83	回测“一创模拟盘2号”，最终优化，年化39%	P5张大猫
84	【社区研究】连板龙头策略-wywy：复现与研究	求索量化
85	【深度解析 七】BBI指标-大盘涨跌-宽基轮动 模型	加百力
86	根据连续亏损预判st股票	wywy1995
87	山寨红利	开心果
88	韶华研究之十六，指数增强	韶华不负
89	【复现·水】市场情绪指标专题:行业指数顶部和底部信号	Hugo2046
90	【复现】凸显度因子	Hugo2046
91	【复现】多任务时序动量策略	Hugo2046
92	【机器学习研究】动态多因子选股策略研究	南澳
93	【复现】国信投资者情绪指数择时模型	Hugo2046
94	大市值小市值今年的一些分析，然后尝试预测小市值行情	MarioC
95	基金的最爱与最恶	Gyro^.^
96	大盘预测器v1.1--引入世界数据	MarioC
97	大周期顶底判断：FED指标+格雷厄姆指数一次搞定	归零韭菜
98	FScore9因子模型改进——RFScore7因子	北门
99	波动率交易的前言——期权定价	Avi433
100	全市场归因	maomao