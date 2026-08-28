# 写策略 → 回测 → 验证

从零写一个策略，到判断它能不能上实盘，全流程。

---

## 一、写策略

放哪都行，只要 Python 能 import 到。建议放 `qmtquant/strategy/`。

### 择时类（每只标的独立判断买卖）

继承 `StrategyBase`，实现 `on_bar`：

```python
from qmtquant.core.objects import BarData
from qmtquant.strategy.base import StrategyBase


class MyStrategy(StrategyBase):
    """我的策略"""

    # 必须声明，否则 --set 传进来的参数会被静默丢弃
    parameters = ["window", "threshold", "price_buffer", "exit_price_buffer"]
    variables = ["inited", "trading", "pos", "trade_count"]

    window: int = 20
    threshold: float = 0.05
    price_buffer: float = 0.03
    exit_price_buffer: float = 0.08

    def __init__(self, engine, strategy_name, vt_symbols, setting=None):
        super().__init__(engine, strategy_name, vt_symbols, setting)
        self._coerce_types()
        self._validate()
        self.closes: dict[str, list[float]] = {}
        self.trade_count = 0

    def _coerce_types(self):
        self.window = int(self.window)
        self.threshold = float(self.threshold)

    def _validate(self):
        if self.window < 2:
            raise ValueError(f"window 至少为 2，实际 {self.window}")
        if self.exit_price_buffer < self.price_buffer:
            raise ValueError("卖出缓冲必须不小于买入缓冲")

    def on_bar(self, bar: BarData):
        if bar.suspended or bar.close_price <= 0:
            return

        closes = self.closes.setdefault(bar.vt_symbol, [])
        closes.append(bar.close_price)
        if len(closes) > self.window + 5:
            del closes[0]           # 必须限长，否则长期运行内存持续增长
        if len(closes) < self.window:
            return

        ma = sum(closes[-self.window:]) / self.window
        pos = self.get_pos(bar.vt_symbol)

        if bar.close_price > ma * (1 + self.threshold) and pos == 0:
            volume = self.get_cash() * 0.95 / bar.close_price   # 整手交给引擎
            if self.buy(bar.vt_symbol,
                        bar.close_price * (1 + self.price_buffer), volume):
                self.trade_count += 1
        elif bar.close_price < ma and pos > 0:
            if self.sell(bar.vt_symbol,
                         bar.close_price * (1 - self.exit_price_buffer), pos):
                self.trade_count += 1
```

### 选股类（每期从一堆股票里挑几只）

继承 `PortfolioStrategy`，只需实现 `select()`，调仓/下单/等权分配基类已处理：

```python
from qmtquant.strategy.portfolio import PortfolioStrategy


class MyPicker(PortfolioStrategy):
    parameters = PortfolioStrategy.parameters + ["lookback"]
    lookback: int = 60

    def update_indicators(self, bars):
        """每根 Bar 都调用，维护你的指标窗口"""
        ...

    def select(self, bars, candidates: list[str]) -> list[str]:
        """返回按优先级排序的标的，基类取前 max_holdings 只等权买入"""
        scored = [(self.score(s), s) for s in candidates]
        scored.sort(reverse=True)
        return [s for _, s in scored]
```

### 五条容易踩的

| 要点 | 为什么 |
|---|---|
| `parameters` 必须声明每个参数 | 未声明的会被**静默丢弃**，不报错 |
| `__init__` 里做类型转换 | 参数寻优传进来的是 `float64`，`deque(maxlen=20.0)` 会炸，且异常常被吞掉 |
| **卖出缓冲要比买入宽** | 错过买入只是少赚，错过卖出是实亏。对称缓冲在暴跌跳空时卖不掉 |
| 别自己把手数取整 | 引擎统一处理；自己取整会让「预算不足一手」被静默丢弃，报告里看不到 |
| 收盘价列表要限长 | 否则长期运行内存持续增长 |

---

## 二、跑回测

**不需要写脚本。** 用通用入口：

```bash
cd /e/qmt && ./.venv/Scripts/python.exe scripts/run_strategy.py --strategy my_pkg.my_mod.MyStrategy --describe
```

`--describe` 先确认参数名和默认值对不对。然后正式跑：

```bash
cd /e/qmt && ./.venv/Scripts/python.exe scripts/run_strategy.py --strategy my_pkg.my_mod.MyStrategy --universe-csv data/universe/index_weight_000300.SH.csv --start 2016-01-01 --end 2026-08-27 --set window=60 --set threshold=0.03 --report reports/mine
```

### 标的池三选一 —— 这一项决定结论可不可信

| 选项 | 偏差 |
|---|---|
| `--symbols 510300.SH` | 自己指定，适合择时/单 ETF |
| `--sector 沪深300` | 当前成分快照 + 上市日/纳入日过滤 |
| `--universe-csv ...` | **历史成分股，无幸存者偏差与成分股前视** |

选股策略**必须**用第三种。实测同一组均值回归参数，换标的池后总收益从 **+320.33% 变成 -4.49%** —— 320 个百分点全是偏差。

回测末尾会打印「标的池偏差说明」，上面写着 `幸存者偏差: 存在 ⚠` 的话，那个收益数字不能信。

### 常用开关

```
--interval 1d|1w|1m     周期
--capital 1000000       初始资金
--drawdown              启用回撤控制（三档：只平不开/强制减仓/全平）
--benchmark 000300.SH   对标指数，空字符串则不对标
--report reports/mine   输出目录，不同策略务必用不同目录，否则互相覆盖
```

### 看结果

`--report` 目录下三个文件：

- `<类名>_report.html` —— 双击打开，净值/回撤曲线、KPI、成交明细，plotly 已内联无需联网
- `<类名>_equity.csv` —— 每日净值
- `<类名>_trades.csv` —— 逐笔成交

单标的时还会自动跟**买入持有**对比。这比跟指数比诚实 —— 择时策略要证明的是「进出场比一直拿着强」。

---

## 三、验证（决定能不能上实盘的一步）

**回测赚钱不等于策略有效。** 参数网格扫一遍总能找出好看的组合，那是拟合噪声。

```bash
cd /e/qmt && ./.venv/Scripts/python.exe scripts/validate_strategy.py --strategy my_pkg.my_mod.MyStrategy --grid '{"window": [20, 40, 60, 120], "threshold": [0.01, 0.03, 0.05]}' --universe-csv data/universe/index_weight_000300.SH.csv --start 2016-01-01 --end 2026-08-27 --split 2022-01-01
```

### 四道检验

| 检验 | 问题 | 不通过意味着 |
|---|---|---|
| **参数平原** | 最优点周围的参数也还行吗？ | 孤峰 = 拟合噪声 |
| **样本外** | 样本内选的参数，没见过的区间还灵吗？ | 只是记住了历史 |
| **Walk-forward** | 滚动地「用过去选参数、在未来验证」 | 时灵时不灵 |
| **成本敏感性** | 手续费翻几倍会打平？ | 靠低成本假设撑着 |

输出长这样：

```
  [未通过] 参数平原          邻域保留 0%
  [未通过] 样本外           衰减 -84%
  [未通过] Walk-forward  盈利窗口 41%
  [通过]   成本承受          5 倍时转负
  1/4 项通过
  存在未通过项，不建议上实盘。
```

**四项全过才考虑实盘。** 项目里六个策略至今没有一个全过。

### 选股类策略先做 IC，别急着回测

一次 IC 分析几秒，一次回测几分钟；而且回测结果掺杂了仓位、成本、调仓频率一堆与因子无关的东西。**因子没有预测力的话，回测出来的盈亏只是噪声的形状。**

```bash
cd /e/qmt && ./.venv/Scripts/python.exe scripts/analyze_factors.py --index 000300.SH --start 2016-01-01
```

看 `IC_t(NW)` 那一列（Newey-West 修正过的）。`|t| < 2` 就别往下做了。

---

## 四、上实盘前

1. 四道检验全过
2. `--drawdown` 开着再跑一遍，确认回撤控制不会把策略锁死
3. 模拟盘跑通下单/撤单/成交（`scripts/test_order_flow.py`）
4. 小资金实盘

配置写进 `config/config.yaml` 的 `strategies` 段，然后 `scripts/run_live.py`。

---

## 附：内置策略短名

`mean_reversion` / `index_timing` / `momentum` / `trend_ma` / `ma_cross` / `intraday_vwap`

自己的策略用完整路径 `包.模块.类名`，不用改任何脚本。
