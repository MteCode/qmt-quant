"""日内做 T —— 底仓 + 相对 VWAP 的分档高抛低吸。

## 做 T 是什么

持有底仓，日内价格偏高时卖掉一部分**昨仓**、偏低时买回来，净持仓回到
底仓水平。赚的是日内波动差价，不改变隔夜暴露。

这是 A 股 T+1 下唯一合法的日内交易方式：当日买入的不能卖，但**昨仓可以**。
所以做 T 必须先有底仓 —— 没有底仓就没有可卖的量，一切无从谈起。

## 为什么用 VWAP 作基准

做 T 要判断「现在算高还是算低」，需要一个基准。几种选择：

- **昨收/今开**：固定值。行情单边走时会一路偏离，要么早早卖光再也买不回，
  要么一直不触发
- **VWAP（成交量加权均价）**：随日内成交动态移动，自带趋势跟随。价格
  高于 VWAP 说明当前买盘占优、相对偏贵，这正是做 T 想卖的时刻

VWAP 还有个好处：它是机构成交成本的通用基准，偏离 VWAP 的幅度有实际含义，
不像「偏离今开 1%」那样随行情波动率变化而失真。

## 分档而不是一次性

一次性卖光的问题是：卖在 +0.8%，结果涨到 +3%，剩下的波段全错过，而且
买不回来（价格再没回到基准下方）。分档则是涨 0.8% 卖一档、涨 1.6% 再卖
一档，逐步兑现，把单点判断变成一条曲线。

档位数由 max_layers 控制。实测候选池里的票日均有十几个 0.8% 级别的波段，
分 3-4 档能覆盖大部分日内摆动。

## 硬性约束

- **卖出量必须 ≤ 可卖量**：当日买回的部分 T+1 冻结，不能再卖。实盘上
  忽略这一点的后果是每分钟发单、每分钟被「可卖数量不足」拒
- **净持仓不超过底仓**：做 T 不是加仓，买回只是补回卖掉的部分
- **涨跌停不交易**：涨停买不进、跌停卖不掉，挂上去只是占额度
"""
from __future__ import annotations

import logging
from typing import Any

from qmtquant.core.constants import OrderType
from qmtquant.core.objects import BarData
from qmtquant.strategy.base import StrategyBase

logger = logging.getLogger(__name__)


class IntradayT0Strategy(StrategyBase):
    """底仓做 T：相对日内 VWAP 分档高抛低吸。"""

    parameters = [
        "base_value",          # 每只标的的底仓金额
        "trade_value",         # 每只标的可用于回转的金额
        "grid",                # 单档偏离阈值
        "max_layers",          # 最多分几档
        "entry_time",          # 开始做 T 的时间
        "exit_time",           # 停止做 T、回补底仓的时间
        "max_intraday_loss",   # 单票日内亏损上限，触发后当日停做
        "avoid_limit",         # 涨跌停不交易
    ]

    variables = StrategyBase.variables + ["layers"]

    base_value: float = 40000.0
    trade_value: float = 40000.0
    #: 单档偏离。和筛选脚本的 zigzag 阈值对齐 —— 筛选用 0.8% 衡量「有多少
    #: 个可做的波段」，执行就该用同一尺度去吃，否则筛出来的机会抓不到
    grid: float = 0.008
    max_layers: int = 3
    entry_time: str = "09:35"
    exit_time: str = "14:45"
    max_intraday_loss: float = 0.02
    avoid_limit: bool = True

    def __init__(self, engine: Any, strategy_name: str,
                 vt_symbols: list[str], setting: dict | None = None):
        #: vt_symbol -> 当前已卖出的档数（正数=已卖，需要买回）
        self.layers: dict[str, int] = {}

        super().__init__(engine, strategy_name, vt_symbols, setting)

        #: 日内累计成交额与成交量，用于算 VWAP
        self._cum_amt: dict[str, float] = {}
        self._cum_vol: dict[str, float] = {}
        self._day: str = ""
        #: 当日已停做的标的（触发止损）
        self._halted: set[str] = set()
        #: 涨跌停价，由启动脚本传入
        self.limit_prices: dict[str, tuple[float, float]] = {}
        lp = (setting or {}).get("limit_prices")
        if isinstance(lp, dict):
            self.limit_prices = {k: (float(v[0]), float(v[1]))
                                 for k, v in lp.items() if v}

    # ------------------------------------------------------------ 生命周期

    def on_start(self) -> None:
        self._reset_day("")

    def _reset_day(self, day: str) -> None:
        """换日重置日内状态。

        layers 也要清零 —— 它记录的是「今天卖出了几档」，隔夜后昨仓构成
        已变，沿用会导致买回数量算错。
        """
        self._day = day
        self._cum_amt.clear()
        self._cum_vol.clear()
        self._halted.clear()
        self.layers = {}

    # ------------------------------------------------------------ 工具

    def _vwap(self, vt: str, bar: BarData) -> float:
        """日内累计 VWAP，用 close 加权而非 amount/volume。

        ## 为什么不用 amount / volume

        xtdata 分钟线的 volume 单位是「手」而 amount 是「元」，两者差 100 倍；
        清洗层又是复权价，实测 amount/volume 得到 1396 而收盘价是 87，
        **差 16 倍**。项目里 features/intraday.py 的 vwap_gap 也踩了同一个坑，
        注释写着「绝对值不准但偏离方向正确」—— 对那个特征够用，对这里不够：
        分档做 T 靠的是 0.8% 这种绝对阈值，基准错 16 倍就永远触发不了。
        实测按 amount/volume 跑 40 个交易日，操作次数为 0。

        改用 Σ(close × volume) / Σvolume：volume 的单位无论是股还是手，
        在分子分母上同时出现、直接约掉，得到的是真正的成交量加权均价。
        代价是用收盘价代替每根 bar 内的真实均价，日内 1 分钟粒度下误差可忽略。
        """
        vol = float(getattr(bar, "volume", 0) or 0)
        px = float(bar.close_price)
        if px <= 0:
            return 0.0
        if vol <= 0:
            # 无成交的 bar 不该改变基准，返回已有 VWAP
            cv = self._cum_vol.get(vt, 0.0)
            return self._cum_amt.get(vt, 0.0) / cv if cv > 0 else px
        self._cum_amt[vt] = self._cum_amt.get(vt, 0.0) + px * vol
        self._cum_vol[vt] = self._cum_vol.get(vt, 0.0) + vol
        cv = self._cum_vol[vt]
        return self._cum_amt[vt] / cv if cv > 0 else px

    def _blocked_by_limit(self, vt: str, price: float, is_buy: bool) -> bool:
        """涨停不买、跌停不卖 —— 挂上去也成交不了，只是占额度。"""
        if not self.avoid_limit:
            return False
        lim = self.limit_prices.get(vt)
        if not lim:
            return False
        up, down = lim
        if is_buy and up > 0 and price >= up - 1e-6:
            return True
        if not is_buy and down > 0 and price <= down + 1e-6:
            return True
        return False

    def _layer_volume(self, price: float) -> float:
        """单档股数，向下取整到 100 股。"""
        if price <= 0 or self.max_layers <= 0:
            return 0.0
        v = self.trade_value / self.max_layers / price
        return float(int(v // 100) * 100)

    # ------------------------------------------------------------ 主循环

    def on_bars(self, bars: dict[str, BarData]) -> None:
        if not bars:
            return
        first = next(iter(bars.values()))
        day = first.datetime.strftime("%Y%m%d")
        if day != self._day:
            self._reset_day(day)
        t = first.datetime.strftime("%H:%M")

        if not self.trading or t < self.entry_time:
            # 仍要累计 VWAP，否则开盘前的成交不计入基准
            for vt, bar in bars.items():
                self._vwap(vt, bar)
            return

        closing = t >= self.exit_time

        for vt, bar in bars.items():
            price = float(bar.close_price)
            vwap = self._vwap(vt, bar)
            if price <= 0 or vwap <= 0:
                continue

            cur = self.layers.get(vt, 0)

            # 尾盘：回到 0 档，恢复底仓。两个方向都要平 ——
            # 卖出的要买回，低位加买的要卖掉
            if closing:
                if cur > 0:
                    self._buy_layers(vt, price, cur)
                elif cur < 0:
                    self._sell_layers(vt, price, -cur)
                continue

            if vt in self._halted:
                continue

            dev = (price - vwap) / vwap

            # 熔断：档位已打满仍在继续单边下跌，说明这天不是震荡行情。
            # 再买下去就是在接飞刀 —— 网格策略最典型的死法
            if cur <= -self.max_layers and dev <= -self.max_intraday_loss:
                self._halted.add(vt)
                self.write_log(f"做T停做 {vt} 偏离 {dev:.2%}，档位已满仍在单边下跌")
                continue

            # 目标档位，**对称**：
            #   高于 VWAP 越多 -> 卖出越多档（正）
            #   低于 VWAP 越多 -> 买入越多档（负）
            # 压成非负是之前的设计错误：只做「高抛-回补」而不做「低吸-高抛」，
            # 结果是单边上涨日靠底仓错配赚钱、震荡日反复付成本，
            # 损益与当日涨跌幅正相关 0.40 —— 做 T 本该与方向无关
            target = int(dev / self.grid)
            target = max(-self.max_layers, min(target, self.max_layers))

            if target > cur:
                self._sell_layers(vt, price, target - cur)
            elif target < cur:
                self._buy_layers(vt, price, cur - target)

    # ------------------------------------------------------------ 下单

    def _sell_layers(self, vt: str, price: float, n: int) -> None:
        """卖出 n 档。

        T+1 下卖出量受**可卖量**约束，而可卖量来自昨仓（底仓）。低位加买
        的部分当日冻结，所以「买低-卖高」这条路径卖的其实是底仓的股份 ——
        股份同质，差价照样吃到，只要当日累计卖出不超过底仓即可。
        """
        if n <= 0 or self._blocked_by_limit(vt, price, is_buy=False):
            return
        want = self._layer_volume(price) * n
        if want <= 0:
            return
        avail = self.get_available(vt)
        vol = float(int(min(want, avail) // 100) * 100)
        if vol <= 0:
            return
        self.sell(vt, price, vol, OrderType.LIMIT)
        # 实际成交量可能少于想卖的量（可卖不足），档位按实际比例推进，
        # 否则档位与持仓脱节，尾盘会平错数量
        done = n if vol >= want else max(1, int(round(n * vol / want)))
        self.layers[vt] = self.layers.get(vt, 0) + done
        self.write_log(f"做T卖出 {vt} {vol:.0f} 股 @{price:.3f} "
                       f"档位 {self.layers[vt]:+d}")

    def _buy_layers(self, vt: str, price: float, n: int) -> None:
        """买入 n 档 —— 既用于回补卖出的档位，也用于低位加买。"""
        if n <= 0 or self._blocked_by_limit(vt, price, is_buy=True):
            return
        want = self._layer_volume(price) * n
        if want <= 0:
            return
        cash = self.get_cash()
        vol = want
        if cash < vol * price:
            vol = float(int((cash / price) // 100) * 100)
        vol = float(int(vol // 100) * 100)
        if vol <= 0:
            return
        self.buy(vt, price, vol, OrderType.LIMIT)
        done = n if vol >= want else max(1, int(round(n * vol / want)))
        self.layers[vt] = self.layers.get(vt, 0) - done
        self.write_log(f"做T买入 {vt} {vol:.0f} 股 @{price:.3f} "
                       f"档位 {self.layers[vt]:+d}")
