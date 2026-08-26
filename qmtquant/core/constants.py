"""核心枚举常量定义。

命名与 vn.py 保持一致，便于将来迁移或接入 vnpy 生态组件。
"""
from enum import Enum


class Direction(Enum):
    """买卖方向"""
    LONG = "买入"
    SHORT = "卖出"


class Offset(Enum):
    """开平方向（A 股股票只用 NONE，两融/期货才区分开平）"""
    NONE = ""
    OPEN = "开仓"
    CLOSE = "平仓"


class Status(Enum):
    """订单状态机"""
    SUBMITTING = "提交中"
    NOTTRADED = "未成交"
    PARTTRADED = "部分成交"
    ALLTRADED = "全部成交"
    CANCELLED = "已撤销"
    REJECTED = "拒单"


#: 处于活动状态的订单（还可能继续成交或被撤销）
ACTIVE_STATUSES = {Status.SUBMITTING, Status.NOTTRADED, Status.PARTTRADED}


class OrderType(Enum):
    """委托价格类型"""
    LIMIT = "限价"
    MARKET = "市价"
    BEST = "对手价"


class Exchange(Enum):
    """交易所"""
    SSE = "SSE"      # 上交所
    SZSE = "SZSE"    # 深交所
    BSE = "BSE"      # 北交所


class Interval(Enum):
    """K 线周期"""
    MINUTE = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR = "1h"
    DAILY = "1d"
    TICK = "tick"


class Product(Enum):
    """标的类型"""
    EQUITY = "股票"
    ETF = "ETF"
    INDEX = "指数"


class RejectReason(Enum):
    """风控拒单原因，用于告警与统计"""
    KILL_SWITCH = "全局急停已开启"
    ORDER_VALUE_LIMIT = "单笔委托金额超限"
    POSITION_RATIO_LIMIT = "单票持仓占比超限"
    TOTAL_POSITION_LIMIT = "总仓位超限"
    ORDER_COUNT_LIMIT = "当日下单笔数超限"
    TURNOVER_LIMIT = "当日成交金额超限"
    DAILY_LOSS_LIMIT = "当日亏损触及阈值，只平不开"
    BLACKLIST = "标的在黑名单中"
    INSUFFICIENT_CASH = "可用资金不足"
    INSUFFICIENT_POSITION = "可卖数量不足"
    PRICE_LIMIT = "委托价超出涨跌停区间"
    INVALID_VOLUME = "委托数量不合法"
    NOT_TRADING_TIME = "非交易时段"
