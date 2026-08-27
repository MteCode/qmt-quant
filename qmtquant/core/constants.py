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
    WEEKLY = "1w"
    MONTHLY = "1mon"
    TICK = "tick"


#: 分钟级周期，数据量大，下载与存储需特别注意
MINUTE_INTERVALS = {
    Interval.MINUTE, Interval.MINUTE_5, Interval.MINUTE_15,
    Interval.MINUTE_30, Interval.HOUR,
}


class Product(Enum):
    """标的类型"""
    EQUITY = "股票"
    ETF = "ETF"
    INDEX = "指数"


#: 各板块涨跌停幅度，按代码前缀判定。
#: 主板 10%，创业板(300/301)/科创板(688) 20%，北交所(4/8/92) 30%。
#: 放在 core 而非 datafeed，是因为**回测撮合与数据校验必须用同一份定义** ——
#: 曾经回测引擎硬编码全局 10%，而沪深300 中有 53 只（18%）是 20% 的创业板/科创板，
#: 导致这些标的涨跌超 10% 的交易日被误判为涨跌停而拒单。
PRICE_LIMIT_BY_PREFIX: dict[str, float] = {
    "688": 0.20,   # 科创板
    "300": 0.20,   # 创业板
    "301": 0.20,   # 创业板
    "92": 0.30,    # 北交所
    "8": 0.30,     # 北交所
    "4": 0.30,     # 北交所
}
DEFAULT_PRICE_LIMIT = 0.10

#: ST 股涨跌停 5%。历史 ST 状态无法从 QMT 取得，故不参与自动判定，
#: 仅作为已知常量供上层显式指定。
ST_PRICE_LIMIT = 0.05


def get_price_limit(symbol: str) -> float:
    """按代码前缀返回涨跌停幅度。

    :param symbol: 6 位代码或 vt_symbol，两者皆可

    注意：拿不到历史 ST 状态，ST 期间实际为 5%，本函数会高估。
    用于回测撮合时会略微放宽限制（把本该拒单的放行），
    用于数据校验时只作上界判断（超过它一定是数据错误）。
    """
    code = symbol.split(".")[0]
    # 先匹配长前缀，避免 "8" 抢先匹配到 "688"
    for prefix in sorted(PRICE_LIMIT_BY_PREFIX, key=len, reverse=True):
        if code.startswith(prefix):
            return PRICE_LIMIT_BY_PREFIX[prefix]
    return DEFAULT_PRICE_LIMIT


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
