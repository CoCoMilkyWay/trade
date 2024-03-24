#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


from ...basicops import MovingAverageBase


class ACD(MovingAverageBase): 
    '''
    升降线(M=20) Accumulation/Distribution Indicator

    对比ACD和均线获得金叉死叉

    当前收盘价高于前期收盘价时,就会产生买进压力。收盘价与实际低点之间的离差加在ACD上。
    实际低点指当期最低价和前期收盘价两者中较低的一个。
    买卖压指数最普遍的用途为: 显示牛背离和熊背离。
    ACD和价格之间的背离表明上升或下降走势开始减弱。价格的高点比前一次的高点高
    但ACD的高点却比前一次的高点低时为熊背离,暗示上升走势开始减弱。

    LC:=REF(CLOSE,1);
    DIF:=CLOSE-IF(CLOSE>LC,MIN(LOW,LC),MAX(HIGH,LC));
    ACD:SUM(IF(CLOSE=LC,0,DIF),0);
    MAACD:EXPMEMA(ACD,M);

    LC赋值:1日前的收盘价
    DIF赋值:收盘价-如果收盘价>LC,返回最低价和LC的较小值,否则返回最高价和LC的较大值
    输出升降线:如果收盘价=LC,返回0,否则返回DIF的历史累和
    输出MAACD:ACD的M日指数平滑移动平均
    '''
    pass

class BBI(MovingAverageBase):
    '''
    多空均线(M1=3,M2=6,M3=12,M4=24) (假设个股周期为30天) BullBearIndex
    
    适合于中长线,提供逃顶保障,当价格跌破BBI,短期也出现抛压,视为平仓信号
    短期内噪声大,效果不佳,容易与股价羁绊

    BBI:(MA(CLOSE,M1)+MA(CLOSE,M2)+MA(CLOSE,M3)+MA(CLOSE,M4))/4;
    输出多空均线:(收盘价的M1日简单移动平均+收盘价的M2日简单移动平均+收盘价的
    M3日简单移动平均+收盘价的M4日简单移动平均)/4
    '''
    pass

class MCST(MovingAverageBase):
    '''
    市场平均成本价格 market cost
    1.MCST是市场平均成本价格;
    2.MCST上升表明市场筹码平均持有成本上升;
    3.MCST下降表明市场筹码平均持有成本下降。
    4.当股价在MCST曲线下方翻红时应关注,若向上突破MCST曲线应买入
    ;当MCST曲线的下降趋势持续超过30日时,若股价在MCST曲线上方翻红应买入。
    5.该指标应结合布林线使用。
    
    市场成本=以成交量(手)/当前流通股本(手)为权重成交额(元)/(100*成交量(手))的动态移动平均；
    MCST:DMA(AMOUNT/(100*VOL),VOL/CAPITAL);

    '''
    pass

class BBIBOLL(MovingAverageBase):
    '''
    上界,下界=多空布林线(N=11,N=6) BullBearIndexBollinger

    两者结合,提供了多种行情下的开平策略

    1.轨道收敛:
    说明行情即将变盘,向上或向下突破.
    2.轨道发散:
    表明将向上或向下扩大趋势.
    3.轨道极度发散:
    趋势向上，上轨线远离股价时为卖出信号;趋势向下，下轨线远离股价时为买入信号。

    1.为BBI与BOLL的迭加;
    2.高价区收盘价跌破BBI线,卖出信号;
    3.低价区收盘价突破BBI线,买入信号;
    4.BBI线向上,股价在BBI线之上,多头势强;
    5.BBI线向下,股价在BBI线之下,空头势强。

    CV:=CLOSE;
    BBIBOLL:(MA(CV,3)+MA(CV,6)+MA(CV,12)+MA(CV,24))/4;
    UPR:BBIBOLL+M*STD(BBIBOLL,N);
    DWN:BBIBOLL-M*STD(BBIBOLL,N);
    CV赋值:收盘价
    输出多空布林线:(CV的3日简单移动平均+CV的6日简单移动平均+CV的12日简单移动平均+CV的24日简单移动平均)/4
    输出UPR:BBIBOLL+M*BBIBOLL的N日估算标准差
    输出DWN:BBIBOLL-M*BBIBOLL的N日估算标准差
    '''
    pass

class ALLIGAT(MovingAverageBase):
    '''
    鳄鱼线(alligator)

    原理：分形几何+非线性动力系统

    多头市场：上唇>牙齿>下颚
    空头市场：下颚>牙齿>上唇

    NN:=(H+L)/2;
    上唇:REF(MA(NN,5),3),COLOR40FF40;
    牙齿:REF(MA(NN,8),5),COLOR0000C0;
    下颚:REF(MA(NN,13),8),COLORFF4040;

    NN赋值:(最高价+最低价)/2
    输出上唇:3日前的NN的5日简单移动平均,COLOR40FF40
    输出牙齿:5日前的NN的8日简单移动平均,COLOR0000C0
    输出下颚:8日前的NN的13日简单移动平均,COLORFF4040
    '''
    pass

class GMMA(MovingAverageBase):
    '''
    顾比均线() Guppy Multiple Moving Average

    顾比均线由两组均线构成：
    短期组:3、5、8、10、12、15
    长期组:30、35、40、45、50、60
    1、长期组从聚拢点向上扩散,发散状态与短期组一致,可做多
    2、长期组从聚拢点向下扩散,发散状态与短期组一致,可做空
    3、长期组从扩散状态向上且急剧向上聚拢,代表趋势反转,短期组呈扩散状态向上穿越长期组,可做多
    4、长期组从扩散状态向下且急剧向下聚拢,代表趋势反转,短期组呈扩散状态向下穿越长期组,可做空

    '''
    pass












