"""回测引擎。

事件驱动的日线/分钟级回测，严格遵守 A 股规则：
- T+1：当日买入次日才可卖
- 100 股整数倍买入
- 涨跌停价不成交、停牌跳过
- **信号用 T 日收盘数据，成交发生在 T+1 开盘**，杜绝前视偏差
"""
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..config import CostConfig
from ..core.constants import (Direction, OrderType, ST_PRICE_LIMIT,
                              Status, get_price_limit)
from ..core.objects import BarData, OrderData, OrderRequest, TradeData
from ..gateway.sim_gateway import calc_cost
from ..risk.drawdown import DrawdownController, DrawdownLevel
from ..strategy.base import StrategyBase
from ..utils.symbol import normalize, split_vt_symbol
from .performance import PerformanceStats, calculate_stats

logger = logging.getLogger(__name__)

#: 历史 ST 判定器。模块级缓存 —— 回测里会调用几十万次，不能每次读文件
_ST_CHECKER = None
_ST_LOADED = False


def _load_st_checker():
    """加载历史 ST 判定器。缺数据时返回 None，调用方退回按板块判定。

    ST 涨跌停 5%%，不区分会让本该拒单的委托成交，高估可成交性。
    数据由 scripts/download_st_history.py 从 Tushare namechange 生成。
    """
    global _ST_CHECKER, _ST_LOADED
    if _ST_LOADED:
        return _ST_CHECKER
    _ST_LOADED = True
    try:
        from ..datafeed.st_history import get_checker, spans_by_symbol
        _ST_CHECKER = get_checker()
        if _ST_CHECKER is None:
            logger.info("无历史 ST 名单，涨跌停按板块判定"
                        "（ST 标的会被当作 10%% 处理）")
        else:
            by_sym = spans_by_symbol()
            logger.info("已加载历史 ST 名单：%d 只标的 %d 个区间",
                        len(by_sym), sum(len(v) for v in by_sym.values()))
    except Exception as e:
        logger.warning("加载历史 ST 名单失败: %s", e)
    return _ST_CHECKER



class BacktestEngine:
    """历史回测引擎"""

    #: 成交量的单位。A 股行情有的按股、有的按手（100 股）计。
    #: None = 还没判定；100 = 手；1 = 股。见 _resolve_volume_unit()
    _volume_unit: int | None = None

    def _resolve_volume_unit(self) -> int:
        """判定装载数据里 volume 的单位：股还是手。

        ## 为什么必须判定，又为什么能判定

        推复权因子需要真实价，真实价 = turnover / (volume * 单位)。
        单看一只股票，两种单位都说得通 —— 平安银行按股算因子 1.04、
        按手算 104，都不离谱。

        但**全市场一起看**就能判：后复权价按定义不低于真实价，
        且没有哪只 A 股复权了 500 倍。所以哪种单位算出的因子更多地
        落在 [0.8, 500]，就是哪种。实测 579 只：
            假设按股 -> 99.7% 的因子落在区间外（几乎全部 < 1）
            假设按手 ->  8.8% 落在区间外（因子≈1 的新股，
                                          收盘价与日均价的日内噪声）

        注意这**不是**逻辑上的决定性判据，只是统计上的：
        两种单位都能算出合理因子的数据集是存在的
        （adj=100 / real=10 / volume 按股，两种解释都自洽）。
        判不出来时会告警，不会静默猜。

        判不出来时返回 100（本仓库数据的实际约定），
        并留下告警 —— 静默猜错会让持仓规模差 100 倍。
        """
        if self._volume_unit is not None:
            return self._volume_unit

        # 判据：哪种单位算出来的复权因子更多地落在合理区间 [1, 500]。
        #
        # 「后复权价 >= 真实价」这条不变量只能**排除**不可能的解释，
        # 两种单位都自洽时选不出来（造一份 adj=100/real=10 的按股数据，
        # 两种解释都不违反）。所以要用完整的因子区间做判据：
        #   因子 < 1   -> 后复权价低于真实价，不可能
        #   因子 > 500 -> 没有哪只 A 股复权了 500 倍，数据有问题
        bad = {1: 0, 100: 0}
        n = 0
        for vt, (adj, turnover, volume) in self._last_bar_stats.items():
            if adj <= 0 or turnover <= 0 or volume <= 0:
                continue
            n += 1
            for unit in (1, 100):
                real = turnover / (volume * unit)
                f = adj / real if real > 0 else 0.0
                if not (0.8 <= f <= 500.0):
                    bad[unit] += 1
        if n < 20:
            # 样本太少，判据不可靠。用本仓库数据的约定，并说明。
            self._volume_unit = 100
            return 100

        self._volume_unit = 1 if bad[1] < bad[100] else 100
        if min(bad.values()) / n > 0.3:
            logger.warning(
                "成交量单位判定不可靠（违反率 股 %.1f%% / 手 %.1f%%），"
                "按 %d 处理；复权因子可能不准，整手取整会受影响",
                bad[1] / n * 100, bad[100] / n * 100, self._volume_unit)
        return self._volume_unit

    def _adj_factor(self, vt_symbol: str) -> float | None:
        """该标的的复权因子（后复权价 / 真实价）。

        本地存的是后复权价，且锚在 IPO，因子差异极大：
        工商银行 1.6、贵州茅台 6.2、盛屯矿业 9.7、平安银行 104。

        整手约束作用在**真实股数**上。用后复权价直接取整会把高因子的
        标的静默剔出标的池 —— 实测沪深300、10 万/只时剔掉 33 只（11%），
        2 万/只时剔掉 123 只（41%），而被剔掉的正是平安银行、云南白药、
        泸州老窖、格力、五粮液这类蓝筹。每个组合回测都因此偏向小盘股。

        ## 两条推导路径

        1. `data/1d_raw/`（不复权价，download_adj_factor.py 下载）—— 更准
        2. 没有 1d_raw 时，从 turnover / (volume * 单位) 推真实价

        路径 2 是必须的：`data/1d_raw` 在本仓库**并不存在**，
        路径 1 对所有标的恒返回 None，整个通道曾经是死的，
        而失效方式是静默降级 —— 回测照跑，只是少了一批蓝筹。

        取不到返回 None，调用方退回按后复权价取整并计数。
        """
        if vt_symbol in self._factor_cache:
            return self._factor_cache[vt_symbol]

        factor = None

        # ---- 路径 1：不复权价目录
        if self._raw_dir is not None:
            code, _, ex = vt_symbol.rpartition(".")
            p = self._raw_dir / ex / f"{code}.parquet"
            a = self._last_adj_close.get(vt_symbol)
            d = self._last_adj_date.get(vt_symbol)
            if p.exists() and a and a > 0 and d:
                try:
                    import pandas as pd
                    raw = pd.read_parquet(p, columns=["close"]).sort_index()
                    if not raw.empty:
                        # 取**与后复权末根同日**的不复权价，不能取 .iloc[-1]：
                        # 不复权文件常比 data/1d/ 新（今天仍在下，后复权滞后
                        # 一两天），直接取末根会跨日相除；两日之间若跨了除权日，
                        # 因子就错，而且错得无声。索引为 'YYYYMMDD' 字符串，
                        # searchsorted 取「不晚于 d 的最近一根」。
                        # 该日若停牌（close=0）则继续往前找 —— 停牌日的 0 价
                        # 会让相除得 0，进而被当作异常因子丢弃。
                        pos = raw.index.searchsorted(d, side="right") - 1
                        while pos >= 0 and float(raw["close"].iloc[pos]) <= 0:
                            pos -= 1
                        if pos >= 0:
                            r = float(raw["close"].iloc[pos])
                            factor = a / r
                except (OSError, ValueError, KeyError):
                    factor = None

        # ---- 路径 2：从成交额与成交量反推
        if factor is None:
            stats = self._last_bar_stats.get(vt_symbol)
            if stats:
                adj, turnover, volume = stats
                unit = self._resolve_volume_unit()
                if adj > 0 and turnover > 0 and volume > 0:
                    real = turnover / (volume * unit)
                    if real > 0:
                        f = adj / real
                        # 因子按定义 >= 1。略小于 1 是收盘价与日均价的
                        # 日内噪声（真实价用的是当日均价，不是收盘价），
                        # 夹到 1；显著偏离则说明数据有问题，不猜。
                        if 0.8 <= f < 1.0:
                            f = 1.0
                        if 1.0 <= f <= 500.0:
                            factor = f

        if factor is None:
            self.factor_fallbacks += 1
        self._factor_cache[vt_symbol] = factor
        return factor

    def __init__(self, initial_capital: float = 1_000_000,
                 cost: CostConfig | None = None,
                 price_limit_ratio: float | None = None,
                 lot_size: int = 100,
                 drawdown: "DrawdownController | None" = None) -> None:
        """
        :param price_limit_ratio: 强制统一的涨跌停幅度。
            默认 None = **按标的代码前缀自动判定**（主板 10%、
            创业板/科创板 20%、北交所 30%）。

            曾经这里硬编码全局 10%，而沪深300 中有 53 只（18%）是 20% 的
            创业板/科创板，它们涨跌超过 10% 的交易日会被误判为涨跌停而拒单，
            回测结果对这些标的系统性失真。仅在需要压制板块差异做对照实验时
            才显式传值。
        :param lot_size: 一手股数，A 股为 100。

            ⚠ 与后复权数据配合时需注意：后复权价高于真实价（茅台约 6 倍），
            一手的**名义成本**因此被放大，会造成本不该有的资金闲置甚至完全买不进。
            实测 100 万本金 / 10 只持仓时，沪深300 中有 36 只取整后为 0 股。
            研究阶段可临时设为 1 以消除该假象，但那样得到的成交量不可实盘复现。
            根本解法是按真实价取整（需复权因子），见 docs/TASKS.md。
        :param drawdown: 回撤控制器。传入后回测与实盘走同一套判定逻辑，
            否则回测会高估策略表现 —— 实盘被回撤控制拦下的仓位，
            回测里却照买不误。
        """
        self.initial_capital = initial_capital
        self.cost = cost or CostConfig()
        self.price_limit_ratio = price_limit_ratio
        self.lot_size = lot_size
        self.drawdown = drawdown

        self.cash: float = initial_capital
        #: vt_symbol -> {"volume": 总量, "available": 可卖, "price": 成本}
        self.positions: dict[str, dict] = {}

        self.strategy: StrategyBase | None = None
        self.history: dict[datetime, dict[str, BarData]] = {}
        self.pending_orders: list[OrderRequest] = []
        #: 标的池。为 None 时策略自行决定交易哪些标的
        self.universe = None

        self.trades: list[TradeData] = []
        self.orders: list[OrderData] = []
        self.equity_curve: dict[datetime, float] = {}

        self._order_count = 0
        self._trade_count = 0
        #: 因取整后不足一手而未能下出的委托，按标的计数。
        #: 后复权价被抬高后这类情况会激增（实测 100 万/10 只时有 36 只完全买不进），
        #: 静默丢弃等于把这些标的悄悄剔出标的池，必须统计出来告警
        self.undersized_orders: dict[str, int] = {}
        #: 被回撤控制拦下的买单数
        self.drawdown_blocked: int = 0

        #: 不复权价目录，用于推导复权因子按真实股数取整。
        #: 不存在则退回按后复权价取整，并在报告中提示
        from ..config import get_config
        try:
            _raw = Path(get_config().data.store_dir) / "1d_raw"
            self._raw_dir = _raw if _raw.exists() else None
        except Exception:
            self._raw_dir = None
        if self._raw_dir is None:
            # 以前这里静默退回，回测照跑、只是少了一批蓝筹 ——
            # 与「一切正常」长得一样。必须让缺失本身可见。
            logger.warning(
                "无 data/1d_raw（不复权价）—— 复权因子只能从成交额/成交量反推，"
                "而成交量单位判定是统计判据、非决定性。低因子/高因子标的的"
                "整手取整可能被静默剔出标的池或算错股数。补数据："
                "python scripts/download_adj_factor.py --sector 中证1000")
        self._factor_cache: dict[str, float | None] = {}
        #: 历史 ST 判定器。有则按当日状态用 5% 涨跌停，
        #: 无则退回按板块判定 —— 会高估 ST 标的的可成交性
        self._is_st = _load_st_checker()
        #: 各标的最后一根 Bar 的后复权收盘价，算因子用
        self._last_adj_close: dict[str, float] = {}
        #: 与之配套的日期（'YYYYMMDD'）。不复权文件比 data/1d/ 新时，
        #: 必须按日期对齐取值，否则跨日相除会算错因子（见 _adj_factor）
        self._last_adj_date: dict[str, str] = {}
        #: {vt_symbol: (后复权收盘价, 成交额, 成交量)}，反推真实价用
        self._last_bar_stats: dict[str, tuple[float, float, float]] = {}
        #: 复权因子取不到、退回按后复权价取整的标的数。
        #: 不为零说明部分标的的整手取整用的是后复权口径，
        #: 高因子的会被静默剔除。
        self.factor_fallbacks: int = 0
        #: 回撤控制强制发出的减仓委托笔数
        self.risk_exit_orders: int = 0
        #: 上一次执行过强制减仓的档位，防止同一档位反复卖出
        self._last_enforced_level = DrawdownLevel.NORMAL
        self._current_bars: dict[str, BarData] = {}
        self._prev_bars: dict[str, BarData] = {}
        self._current_dt: datetime | None = None

    # ------------------------------------------------------------ 数据装载

    def load_data(self, bars: list[BarData]) -> None:
        """装载历史 K 线，按时间戳聚合成截面"""
        grouped: dict[datetime, dict[str, BarData]] = defaultdict(dict)
        for bar in bars:
            grouped[bar.datetime][bar.vt_symbol] = bar
            # 记末根收盘价，与不复权价相除得到复权因子（见 _adj_factor）
            if bar.close_price > 0:
                self._last_adj_close[bar.vt_symbol] = bar.close_price
                self._last_adj_date[bar.vt_symbol] = bar.datetime.strftime("%Y%m%d")
                # 成交额与成交量用于反推真实价（见 _adj_factor 路径 2）
                self._last_bar_stats[bar.vt_symbol] = (
                    bar.close_price, bar.turnover, bar.volume)
        self.history = dict(sorted(grouped.items()))
        n_dates = len({dt.date() for dt in grouped})
        if n_dates < len(grouped):
            logger.warning(
                "装载了日内级别数据（%d 个截面 / %d 个自然日）。"
                "T+1 结算按截面触发，日内数据会导致当日买入立刻可卖 —— "
                "此引擎仅支持日频回测",
                len(grouped), n_dates)
        logger.info("已装载 %d 个时间截面，标的数 %d",
                    len(self.history), len({b.vt_symbol for b in bars}))

    def add_strategy(self, strategy_class: type[StrategyBase],
                     vt_symbols: list[str], setting: dict | None = None) -> None:
        # 必须归一化：策略普遍用 vt_symbol 作字典 key，而 BarData.vt_symbol
        # 永远是 `600519.SSE` 格式。若调用方传入 `600519.SH`，key 对不上，
        # 策略会静默地一根 Bar 都处理不到 —— 不报错，只是永远没有信号。
        vt_symbols = [normalize(s) for s in vt_symbols]
        self.strategy = strategy_class(self, strategy_class.__name__, vt_symbols, setting)

    def set_universe(self, provider) -> None:
        """设置标的池。选股型策略通过 self.engine.get_universe(dt) 取当日可选标的。"""
        self.universe = provider
        report = provider.describe_bias()
        logger.info("标的池已设置，%d 个标的", report.size)
        if not report.is_clean:
            logger.warning("标的池存在偏差，回测收益会被系统性高估：\n%s", report.summary())

    def get_universe(self, dt=None) -> list[str]:
        """取指定日期（默认当前回测时点）的可交易标的"""
        if self.universe is None:
            return list(self.strategy.vt_symbols) if self.strategy else []
        return self.universe.get_universe(dt or self._current_dt)

    # ------------------------------------------------------------ 主循环

    def run(self) -> PerformanceStats:
        if not self.strategy:
            raise RuntimeError("未添加策略，请先调用 add_strategy()")
        if not self.history:
            raise RuntimeError("未装载数据，请先调用 load_data()")

        self.strategy.on_init()
        self.strategy.inited = True
        self.strategy.on_start()
        self.strategy.trading = True

        for dt, bars in self.history.items():
            self._current_dt = dt
            self._prev_bars = self._current_bars
            self._current_bars = bars

            # 1) 新交易日开始：释放 T+1 冻结
            self._settle_t1()
            # 2) 用开盘价撮合上一根 Bar 产生的委托，避免前视偏差
            self._match_pending(bars)
            # 3) 推送行情给策略，策略在此产生新信号
            self._push_bars(bars)
            # 4) 按收盘价记录净值，并推给回撤控制器
            equity = self._calc_equity(bars)
            self.equity_curve[dt] = equity
            if self.drawdown is not None:
                self.drawdown.update(equity)
                # 5) 触发减仓/清仓档位时**真的卖出**。
                # 只挡住开新仓是压不住回撤的：持仓不动的话，
                # 行情继续跌，回撤照样往下走
                self._enforce_drawdown(bars)

        self.strategy.trading = False
        self.strategy.on_stop()

        equity = pd.Series(self.equity_curve).sort_index()
        stats = calculate_stats(equity, self.trades, self.initial_capital)
        return stats

    def _push_bars(self, bars: dict[str, BarData]) -> None:
        try:
            self.strategy.on_bars(bars)
            for bar in bars.values():
                self.strategy.on_bar(bar)
        except Exception:
            logger.exception("策略处理 Bar 异常 dt=%s", self._current_dt)

    def _settle_t1(self) -> None:
        """新交易日：昨日买入的股票解冻可卖"""
        for pos in self.positions.values():
            pos["available"] = pos["volume"]

    # ------------------------------------------------------------ 撮合

    def _match_pending(self, bars: dict[str, BarData]) -> None:
        """用当根 Bar 的开盘价撮合挂起委托；不可成交的直接作废（不留隔日单）"""
        pending, self.pending_orders = self.pending_orders, []

        # 卖单必须排在买单之前：调仓时先卖出释放资金，买单才有钱成交。
        # 否则同一批调仓指令会因「资金不足」大面积拒单，回测结果严重失真。
        pending.sort(key=lambda r: 0 if r.direction == Direction.SHORT else 1)

        for req in pending:
            bar = bars.get(req.vt_symbol)
            if bar is None or bar.suspended:
                self._reject(req, "标的停牌或无行情")
                continue

            prev = self._prev_bars.get(req.vt_symbol)
            pre_close = prev.close_price if prev else bar.open_price
            ratio = self._limit_ratio(req.vt_symbol, bar.datetime)
            limit_up = round(pre_close * (1 + ratio), 2)
            limit_down = round(pre_close * (1 - ratio), 2)

            # 一字涨停买不进，一字跌停卖不出
            if req.direction == Direction.LONG and bar.open_price >= limit_up:
                self._reject(req, "开盘涨停，无法买入")
                continue
            if req.direction == Direction.SHORT and bar.open_price <= limit_down:
                self._reject(req, "开盘跌停，无法卖出")
                continue

            # 限价单需价格可达
            if req.order_type == OrderType.LIMIT:
                if req.direction == Direction.LONG and req.price < bar.open_price:
                    self._reject(req, "限价低于开盘价，未成交")
                    continue
                if req.direction == Direction.SHORT and req.price > bar.open_price:
                    self._reject(req, "限价高于开盘价，未成交")
                    continue

            # 成交价 = 开盘价 + 滑点（买入向上、卖出向下）。
            #
            # 用**相对**滑点而非绝对 tick：本地行情是后复权价，
            # 复权因子从 1.6 到 104 不等，把 0.01 元加到后复权价上，
            # 对平安银行只滑了真实价的百分之一个 tick，
            # 且低估倍数因股而异 —— 不同标的的回测成本互相不可比。
            # 相对值施加在后复权价和真实价上效果相同，绕开了这个问题。
            slip = bar.open_price * self.cost.slippage_rate
            price = bar.open_price + (slip if req.direction == Direction.LONG else -slip)
            price = max(min(price, limit_up), limit_down)
            self._fill(req, price)

    def _limit_ratio(self, vt_symbol: str, dt=None) -> float:
        """该标的的涨跌停幅度。

        优先级：显式指定 > 当日 ST 状态（5%）> 按板块判定（10/20/30%）。

        ST 必须按**当日**状态判定，不能用当前状态：某只股票 2023 年被 ST、
        2025 年摘帽，按当前状态回测 2023 年会用 10% 撮合本该 5% 的标的，
        让本该拒单的委托成交，系统性高估可成交性。
        """
        if self.price_limit_ratio is not None:
            return self.price_limit_ratio
        if dt is not None and self._is_st is not None:
            try:
                if self._is_st(vt_symbol, dt):
                    return ST_PRICE_LIMIT
            except Exception:
                pass
        return get_price_limit(vt_symbol)

    def _fill(self, req: OrderRequest, price: float) -> None:
        volume = req.volume
        fee = calc_cost(price, volume, req.direction, self.cost)

        if req.direction == Direction.LONG:
            need = price * volume + fee
            if need > self.cash:
                self._reject(req, "资金不足")
                return
            self.cash -= need
            pos = self.positions.setdefault(
                req.vt_symbol, {"volume": 0.0, "available": 0.0, "price": 0.0})
            pos["price"] = (pos["price"] * pos["volume"] + price * volume) / (pos["volume"] + volume)
            pos["volume"] += volume
            # T+1：当日买入不可卖，available 不增加
        else:
            pos = self.positions.get(req.vt_symbol)
            if not pos or pos["available"] < volume:
                self._reject(req, "可卖数量不足")
                return
            self.cash += price * volume - fee
            pos["volume"] -= volume
            pos["available"] -= volume
            if pos["volume"] <= 0:
                self.positions.pop(req.vt_symbol, None)

        self._trade_count += 1
        symbol, exchange = split_vt_symbol(req.vt_symbol)
        trade = TradeData(
            symbol=symbol, exchange=exchange,
            orderid=f"BT{self._order_count:08d}", tradeid=f"BT{self._trade_count:08d}",
            direction=req.direction, price=price, volume=volume, commission=fee,
            datetime=self._current_dt, reference=req.reference, gateway_name="BACKTEST",
        )
        self.trades.append(trade)
        try:
            self.strategy.on_trade(trade)
        except Exception:
            logger.exception("策略处理成交回报异常")

    #: 风控强制平仓的委托标记，用于区分策略主动交易与被动减仓
    RISK_REFERENCE = "__RISK_DRAWDOWN__"
    #: 风控减仓的限价缓冲。给得深是有意的 ——
    #: 风控要出场就必须出得去，挂不上单的止损等于没有止损。
    #: 撮合价取开盘价而非限价，放宽不恶化成交价
    RISK_EXIT_BUFFER = 0.09

    def _enforce_drawdown(self, bars: dict[str, BarData]) -> None:
        """按回撤档位强制削减持仓。

        这一段之前是缺失的：控制器有 CLOSE_ONLY/REDUCE/FLAT 三档，
        但引擎只调用了 ``allow_open()`` —— 后两档从未被执行。
        结果是回撤中「不开新仓但也不卖」，行情继续跌则回撤继续深，
        20% 的上限根本压不住。

        委托挂到下一根 Bar 开盘撮合，与策略下单走同一条路径，
        不引入前视偏差。限价给足缓冲 ——
        **风控要出场就必须出得去**，挂不上单的止损等于没有止损。
        """
        level = self.drawdown.level
        # **只在档位升高的那一根 Bar 执行一次减仓。**
        #
        # 真 bug：原本每根 Bar 都重算「卖到 volume * ratio」，
        # 而 volume 已经是上次卖完之后的值 —— 于是指数衰减地反复卖，
        # 实测在低换手策略上制造出 6556 笔成交、胜率 2.25%
        # （正常应为 610 笔、49.66%），全是被反复切割出来的微亏平仓。
        #
        # 「削减到该比例」是一次性动作，不是每根 Bar 的目标。
        if level <= self._last_enforced_level:
            self._last_enforced_level = level   # 档位下降时同步，允许再次触发
            return
        self._last_enforced_level = level

        ratio = self.drawdown.target_position_ratio()
        if ratio >= 1.0:
            return

        for vt_symbol, pos in list(self.positions.items()):
            volume = pos["volume"]
            if volume <= 0:
                continue
            # T+1：当日买入的部分卖不掉，只能减 available 那部分
            sellable = min(pos.get("available", volume), volume)
            if sellable <= 0:
                continue

            target = volume * ratio
            excess = volume - target
            if excess <= 0:
                continue
            sell_volume = min(excess, sellable)

            # 整手取整。正常下单路径做了这件事，这条强平路径原先没做 ——
            # 会发出 137 股这种委托，实盘会被券商拒掉，回测却照单成交，
            # 于是回测里的「减仓能力」强于实际。
            #
            # A 股的规则是：部分卖出须为 100 的整数倍，但**清空整个可卖
            # 持仓**时零股可以一次性卖掉。所以只在不是清仓时取整。
            #
            # 与买入路径同样，整手约束作用在真实股数上，要按复权因子换算
            # （见 _adj_factor）。
            if sell_volume < sellable:
                factor = self._adj_factor(vt_symbol) or 1.0
                real = sell_volume * factor
                real = int(real // self.lot_size) * self.lot_size
                sell_volume = real / factor
            if sell_volume <= 0:
                continue

            bar = bars.get(vt_symbol)
            if bar is None or bar.suspended or bar.close_price <= 0:
                continue   # 停牌卖不掉，下一根 Bar 再试

            self._order_count += 1
            symbol, exchange = split_vt_symbol(vt_symbol)
            self.pending_orders.append(OrderRequest(
                symbol=symbol, exchange=exchange, direction=Direction.SHORT,
                order_type=OrderType.LIMIT,
                # 深度限价：等价于实盘「跌停价挂单」，成交价仍取开盘价，
                # 放宽缓冲不恶化成交价，只提高成交概率
                price=bar.close_price * (1 - self.RISK_EXIT_BUFFER),
                volume=sell_volume, reference=self.RISK_REFERENCE))
            self.risk_exit_orders += 1
            logger.debug("回撤 %s：强制减仓 %s %.0f 股（目标比例 %.0f%%）",
                         self.drawdown.level.label, vt_symbol, sell_volume,
                         ratio * 100)

    def _reject(self, req: OrderRequest, msg: str) -> None:
        symbol, exchange = split_vt_symbol(req.vt_symbol)
        order = OrderData(
            symbol=symbol, exchange=exchange, orderid=f"BT{self._order_count:08d}",
            direction=req.direction, price=req.price, volume=req.volume,
            status=Status.REJECTED, message=msg, datetime=self._current_dt,
            reference=req.reference, gateway_name="BACKTEST",
        )
        self.orders.append(order)
        logger.debug("回测拒单 %s %s: %s", self._current_dt, req.vt_symbol, msg)

    # ------------------------------------------------------------ 策略调用的接口

    def send_order(self, strategy_name: str, vt_symbol: str, direction: Direction,
                   price: float, volume: float,
                   order_type: OrderType = OrderType.LIMIT) -> str:
        """策略下单：不立即成交，挂到下一根 Bar 开盘撮合"""
        if volume <= 0:
            return ""
        # 回撤控制：达到任一档位即停止开新仓，卖出始终放行
        if (direction == Direction.LONG and self.drawdown is not None
                and not self.drawdown.allow_open()):
            self.drawdown_blocked += 1
            return ""

        # 买入向下取整到一手。
        #
        # 取整必须按**真实价**而非后复权价：后复权价被抬高（茅台约 5.4 倍），
        # 一手的名义成本同比例放大，会把高价股静默剔出标的池 ——
        # 实测 100 万/10 只时沪深300 有 36 只取整后为 0 股，
        # 而它们往往正是大盘蓝筹。
        #
        # 有复权因子时按真实价算股数，再换算回后复权口径的金额；
        # 没有则退回旧行为并计数，便于评估影响面。
        if direction == Direction.LONG:
            factor = self._adj_factor(vt_symbol)
            if factor and factor > 0:
                # 后复权价 = 真实价 x 复权因子，故同样的钱能买到的
                # **真实股数** 是后复权口径的 factor 倍。整手约束作用在
                # 真实股数上，取整后再换算回后复权口径
                real_shares = int(volume * factor // self.lot_size) * self.lot_size
                volume = real_shares / factor
            else:
                volume = int(volume // self.lot_size) * self.lot_size
            if volume <= 0:
                # 不足一手买不了。计数以便在报告中告警，
                # 而不是无声地少一个候选标的。
                self.undersized_orders[vt_symbol] = (
                    self.undersized_orders.get(vt_symbol, 0) + 1)
                return ""

        self._order_count += 1
        symbol, exchange = split_vt_symbol(vt_symbol)
        req = OrderRequest(symbol=symbol, exchange=exchange, direction=direction,
                           order_type=order_type, price=price, volume=volume,
                           reference=strategy_name)
        self.pending_orders.append(req)
        return f"BACKTEST.BT{self._order_count:08d}"

    def cancel_order(self, vt_orderid: str) -> None:
        """回测中委托只存活一根 Bar，撤单无实际意义"""

    def cancel_all(self, strategy_name: str) -> None:
        self.pending_orders = [
            r for r in self.pending_orders if r.reference != strategy_name
        ]

    def get_cash(self) -> float:
        return self.cash

    def get_pos(self, vt_symbol: str) -> float:
        return self.positions.get(vt_symbol, {}).get("volume", 0)

    def load_bars(self, strategy, days: int, interval: str = "1d") -> None:
        """回测数据已在 load_data 中一次性装载，此处无需额外操作"""

    # ------------------------------------------------------------ 净值

    def _calc_equity(self, bars: dict[str, BarData]) -> float:
        market_value = 0.0
        for vt_symbol, pos in self.positions.items():
            bar = bars.get(vt_symbol)
            price = bar.close_price if bar else pos["price"]
            market_value += pos["volume"] * price
        return self.cash + market_value

    def get_equity_df(self) -> pd.DataFrame:
        s = pd.Series(self.equity_curve).sort_index()
        df = pd.DataFrame({"equity": s})
        df["returns"] = df["equity"].pct_change().fillna(0)
        df["drawdown"] = df["equity"] / df["equity"].cummax() - 1
        return df

    def get_trades_df(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "datetime": t.datetime, "symbol": t.vt_symbol,
            "direction": t.direction.value, "price": t.price,
            "volume": t.volume, "commission": round(t.commission, 2),
            "amount": round(t.price * t.volume, 2), "strategy": t.reference,
        } for t in self.trades])
