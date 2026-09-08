"""信号文件策略 —— 把离线选股的产出接进实盘引擎。

## 为什么需要它

项目里长期存在两个平行世界：

- **引擎世界**：run_live.py + LiveEngine + Gateway + RiskManager，
  事件驱动、常驻、订单状态机与对账齐全 —— 但只跑了一个演示用的均线策略。
- **脚本世界**：generate_signal.py 产出 target_latest.csv，
  paper_trade.py 读它下单 —— 真正的策略全在这边，但每个脚本自己搭一套
  网关和风控，跑完就退，没有订单生命周期管理。

两边不通的技术原因之一是 `PortfolioStrategy.on_bars` 依赖
`engine.get_universe()`，而这是回测引擎才有的方法 —— 直接拿它跑实盘会
AttributeError。

这个类是缺失的那座桥：读信号文件，按引擎的方式下单，从此离线选股的
产出也能享受引擎的风控、订单状态机、对账与持久化。

## 与 PortfolioStrategy 的区别

| | PortfolioStrategy | 本类 |
|---|---|---|
| 调仓时机 | 每 N 根 bar | **信号文件更新时** |
| 目标来源 | 子类 select() 现算 | 读 CSV |
| 权重 | 等权 | **信号里的 weight 列** |
| 标的池 | engine.get_universe() | 信号文件本身 |

## 信号文件格式

    vt_symbol,score,weight,target_value
    001267.SZSE,0.322668,0.046,23003.15

`weight` 是占总资产的比例。没有 weight 列时退化为等权。

## 安全约束

- **信号新鲜度**：超过 max_signal_age_days 天的信号不执行。
  隔了三天的选股结果拿来今天下单，等于用过期的判断做决策。
- **每日只调一次**：同一份信号不重复执行，避免挂单未成交时
  重复计算差异导致重复下单。
- 停牌标的跳过，下个信号日再试。
"""
from __future__ import annotations

import csv
import logging
from datetime import datetime, timedelta
from pathlib import Path

from ..core.objects import BarData
from .portfolio import PortfolioStrategy

logger = logging.getLogger(__name__)


class SignalFileStrategy(PortfolioStrategy):
    """按信号文件调仓。"""

    parameters = PortfolioStrategy.parameters + [
        "signal_file", "max_signal_age_days", "min_order_value",
    ]

    variables = PortfolioStrategy.variables + ["last_signal_key"]

    #: 信号文件路径（绝对路径或相对项目根）
    signal_file: str = ""
    #: 信号超过这个天数就不执行 —— 过期的判断不该驱动今天的交易
    max_signal_age_days: int = 3
    #: 单笔委托金额下限，低于此值不下单（避免几百块的碎单）
    min_order_value: float = 5000.0

    def __init__(self, engine, strategy_name, vt_symbols, setting=None):
        #: 上次执行的信号标识（文件修改时间），用于避免重复调仓
        self.last_signal_key: str = ""
        super().__init__(engine, strategy_name, vt_symbols, setting)

    # ------------------------------------------------------------ 信号读取

    def _signal_path(self) -> Path | None:
        if not self.signal_file:
            return None
        p = Path(self.signal_file)
        if not p.is_absolute():
            p = Path(__file__).resolve().parents[2] / p
        return p if p.exists() else None

    def load_signal(self) -> tuple[dict[str, float], str] | None:
        """读取信号，返回 ({vt_symbol: weight}, 信号标识)。

        读不到、格式不对、或已过期时返回 None 并写日志 ——
        静默跳过会让人以为策略在跑，实际什么都没做。
        """
        p = self._signal_path()
        if p is None:
            self.write_log(f"信号文件不存在: {self.signal_file}")
            return None

        try:
            mtime = datetime.fromtimestamp(p.stat().st_mtime)
        except OSError as e:
            self.write_log(f"读取信号文件状态失败: {e}")
            return None

        age = datetime.now() - mtime
        if age > timedelta(days=self.max_signal_age_days):
            self.write_log(
                f"信号已过期 {age.days} 天（上限 {self.max_signal_age_days}），"
                f"不执行调仓。请先跑信号生成。")
            return None

        try:
            with p.open(encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
        except (OSError, csv.Error) as e:
            self.write_log(f"信号文件读取失败: {e}")
            return None

        if not rows or "vt_symbol" not in rows[0]:
            self.write_log("信号文件为空或缺少 vt_symbol 列")
            return None

        weights: dict[str, float] = {}
        has_weight = "weight" in rows[0]
        for r in rows:
            sym = (r.get("vt_symbol") or "").strip()
            if not sym:
                continue
            try:
                w = float(r.get("weight") or 0) if has_weight else 0.0
            except ValueError:
                w = 0.0
            weights[sym] = w

        if not weights:
            self.write_log("信号文件没有有效标的")
            return None

        # 没有 weight 列（或全为 0）时退化为等权
        if sum(weights.values()) <= 0:
            eq = 1.0 / len(weights)
            weights = {s: eq for s in weights}

        return weights, f"{p.name}@{mtime:%Y-%m-%d %H:%M:%S}"

    # ------------------------------------------------------------ 主循环

    def select(self, bars: dict[str, BarData], candidates: list[str]) -> list[str]:
        """满足基类契约。实际调仓走 on_bars，这里只回传信号标的。"""
        sig = self.load_signal()
        return list(sig[0]) if sig else []

    def on_bars(self, bars: dict[str, BarData]) -> None:
        """信号文件更新时调仓。

        不用基类的「每 N 根 bar」计时 —— 那套依赖 engine.get_universe()，
        实盘引擎没有这个方法；而且信号是离线算好的，按 bar 计数调仓
        会在信号没变时反复下单。
        """
        self.update_indicators(bars)
        if not self.trading:
            return

        sig = self.load_signal()
        if sig is None:
            return
        weights, key = sig

        # 同一份信号只执行一次：挂单未成交时持仓不变，重复执行会
        # 算出同样的差异再下一遍，直接双倍建仓
        if key == self.last_signal_key:
            return

        self.write_log(f"信号更新 [{key}]，{len(weights)} 只标的，开始调仓")
        self.last_signal_key = key
        self.last_selection = list(weights)
        self.rebalance_by_weight(weights, bars)

    # ------------------------------------------------------------ 调仓

    def rebalance_by_weight(self, weights: dict[str, float],
                            bars: dict[str, BarData]) -> None:
        """按信号权重调仓。先卖后买 —— 卖出释放的资金供买入使用。"""
        total_value = self._estimate_total_value(bars)
        if total_value <= 0:
            self.write_log("总资产为 0，跳过调仓")
            return

        target_set = set(weights)
        held = {s for s, v in self.pos.items() if v > 0}

        # ---- 卖出不在目标里的 ----
        n_sell = 0
        for vt_symbol in sorted(held - target_set):
            bar = bars.get(vt_symbol)
            if bar is None or bar.suspended:
                self.write_log(f"  {vt_symbol} 停牌，本次不卖")
                continue
            volume = self.get_pos(vt_symbol)
            if volume > 0:
                self.sell(vt_symbol,
                          bar.close_price * (1 - self.price_buffer), volume)
                n_sell += 1

        # ---- 买入 / 补足目标仓位 ----
        n_buy = 0
        investable = total_value * (1 - self.cash_buffer)
        for vt_symbol, w in sorted(weights.items(), key=lambda x: -x[1]):
            bar = bars.get(vt_symbol)
            if bar is None or bar.suspended or bar.close_price <= 0:
                continue
            target_value = investable * w
            held_value = self.get_pos(vt_symbol) * bar.close_price
            gap = target_value - held_value
            if gap < self.min_order_value:
                continue        # 已达标或差额太小，不值得付一次交易成本
            volume = gap / bar.close_price
            self.buy(vt_symbol,
                     bar.close_price * (1 + self.price_buffer), volume)
            n_buy += 1

        self.write_log(f"调仓完成：卖出 {n_sell} 只，买入 {n_buy} 只")
