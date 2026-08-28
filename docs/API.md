# qmtquant 接口手册

写策略时需要知道的全部接口。按「数据 → 策略 → 回测 → 验证 → 实盘」的顺序。

---

# 一、数据

## 1.1 本地已有什么

| 目录 | 内容 | 覆盖 | 大小 |
|---|---|---|---|
| `data/1d/` | 日线 | **641 只**，2015-01 ~ 2026-08 | 77 MB |
| `data/1w/` | 周线 | 300 只，2020-01 ~ 2026-08 | 7 MB |
| `data/1m/` | 分钟线 | 300 只，**仅 2025-08 ~ 2026-08** | 582 MB |
| `data/index/` | 指数日线 | 5 个基准 | 1 MB |
| `data/financial/` | 三大报表+每股指标 | 300 只 × 5 表 | 106 MB |
| `data/factor/daily_basic/` | 逐日估值因子 | **全市场 5803 只**，1100 万行 | 685 MB |
| `data/universe/` | 历史成分股 | 沪深300/中证500/中证1000 | 10 MB |

**分钟线只有 1 年**（券商限制）。日线有 11 年。做样本外检验时这个差别是决定性的 —— 1 年数据切两半，每段只剩半年，任何结论都不稳健。

## 1.2 数据来源分工

| 来源 | 给什么 | 要求 |
|---|---|---|
| **QMT / xtdata** | 行情（日/周/分钟）、财报、当前成分 | 客户端已登录 |
| **Tushare Pro** | **历史**成分股、逐日估值因子 | 2000 积分（200 元/年） |
| akshare | 退市股行情（备用） | 免费 |

QMT 给不了历史成分股 —— 只有当前快照。这是必须买 Tushare 的唯一理由。

## 1.3 下载命令

```bash
cd /e/qmt && ./.venv/Scripts/python.exe scripts/download_data.py --sector 沪深300 --intervals 1d,1w --start 2015-01-01
```

| 脚本 | 拉什么 | 依赖 |
|---|---|---|
| `download_data.py` | 行情 K 线 | QMT 已登录 |
| `download_index.py` | 基准指数 | QMT |
| `download_financial.py` | 财务报表 | QMT |
| `build_universe.py` | 当前成分 + 上市日/纳入日 | QMT |
| `download_index_weight.py` | **历史成分股** | Tushare |
| `download_daily_basic.py` | **逐日估值因子** | Tushare |
| `check_data.py` | 数据质量体检 | — |

常用参数：`--intervals 1m,5m,15m,30m,1h,1d,1w,1mon` / `--start` / `--end` / `--resume`（断点续传）/ `--summary`（只看库存）

## 1.4 复权

全局 `back`（后复权），在 `config.yaml` 的 `data.dividend_type`。

**不要改成 front**：前复权对高分红股会算出负价格（实测 601919 最低 -5.14），7 只股票 851 根 Bar 受影响。

## 1.5 策略能拿到的字段

`BarData`：

```python
symbol, exchange, datetime, interval,
open_price, high_price, low_price, close_price,
volume,      # 股（datafeed 层已把 手→股 转好）
turnover,    # 成交额（元）⚠ 未复权，见下
suspended    # 是否停牌
```

⚠ **`turnover` 是未复权的**，而 `close_price` 是后复权的。两者相除得到的均价落在另一个价格空间 —— 实测茅台收盘价 8137、`turnover/volume` 只有 1300，比值恒为 6.26（复权因子）。需要日内均价请用 `IntradayVwap`，它累加 `价格 × 成交量`。

## 1.6 估值因子（不在 BarData 里）

PE/PB/PS/股息率/换手率/市值等要用 `FactorStore`：

```python
from qmtquant.datafeed.factor_store import FactorStore

store = FactorStore(cfg.data.store_dir, ["turnover_rate_f", "pb"],
                    start="2016-01-01", end="2026-08-27")

store.get_cross_section(dt, "pb")           # {vt_symbol: 值}，asof 语义
store.rolling_mean(dt, "turnover_rate_f", 60)  # 过去 60 日均值
```

只载入用到的列 —— 全量 18 列进内存要几个 GB。

可用列：`close, turnover_rate, turnover_rate_f, volume_ratio, pe, pe_ttm, pb, ps, ps_ttm, dv_ratio, dv_ttm, total_share, float_share, free_share, total_mv, circ_mv`

---

# 二、写策略

## 2.1 两种基类

| 基类 | 用于 | 要实现 |
|---|---|---|
| `StrategyBase` | 择时（每只标的独立判断） | `on_bar(bar)` |
| `PortfolioStrategy` | 选股（每期挑几只） | `select(bars, candidates)` |

### 择时型

```python
from qmtquant.core.objects import BarData
from qmtquant.strategy.base import StrategyBase


class MyStrategy(StrategyBase):
    parameters = ["window", "price_buffer", "exit_price_buffer"]
    variables = ["inited", "trading", "pos", "trade_count"]

    window: int = 20
    price_buffer: float = 0.03
    exit_price_buffer: float = 0.08

    def __init__(self, engine, strategy_name, vt_symbols, setting=None):
        super().__init__(engine, strategy_name, vt_symbols, setting)
        self._coerce_types()
        self._validate()
        self.trade_count = 0

    def on_bar(self, bar: BarData):
        if bar.suspended or bar.close_price <= 0:
            return
        ...
        self.buy(bar.vt_symbol, price, volume)
        self.sell(bar.vt_symbol, price, volume)
```

### 选股型

```python
from qmtquant.strategy.portfolio import PortfolioStrategy


class MyPicker(PortfolioStrategy):
    parameters = ["lookback", "max_holdings", "rebalance_days"]
    lookback: int = 60

    def update_indicators(self, bars):
        """每根 Bar 都调用，维护指标窗口"""

    def select(self, bars, candidates: list[str]) -> list[str]:
        """返回按优先级排序的标的。基类取前 max_holdings 只等权买入，
        调仓/下单/资金分配都已处理"""
        return sorted(candidates, key=self.score)
```

## 2.2 可用的调用

| 方法 | 说明 |
|---|---|
| `self.buy(vt_symbol, price, volume)` | 限价买。返回订单号，被拦截时返回 `""` |
| `self.sell(vt_symbol, price, volume)` | 限价卖 |
| `self.cancel_order(vt_orderid)` / `self.cancel_all()` | 撤单 |
| `self.get_pos(vt_symbol)` | 策略视角持仓 |
| `self.get_cash()` | 可用资金 |
| `self.load_bars(days)` | 预热历史（实盘有效，回测是空实现） |
| `self.write_log(msg)` | 带策略名的日志 |
| `self.engine.get_universe()` | 当日可交易标的（选股型用） |

生命周期回调：`on_init` / `on_start` / `on_stop` / `on_bar` / `on_bars` / `on_tick` / `on_order` / `on_trade`

**同一份策略代码在回测与实盘下行为完全一致** —— 策略只调 `self.buy/sell`，由注入的 engine 决定落到模拟撮合还是真实券商。

## 2.3 七条硬规矩

| 规矩 | 不遵守的后果 |
|---|---|
| `parameters` 必须声明每个参数 | 未声明的被**静默丢弃**，不报错 |
| `__init__` 里做类型转换 | 寻优传 `float64`，`deque(maxlen=20.0)` 会炸且异常常被吞 |
| **卖出缓冲 ≥ 买入缓冲** | 暴跌跳空时卖单限价高于开盘价，被拒，出不去 |
| 别自己取整手数 | 引擎统一处理；自己取整会让「不足一手」被静默丢弃 |
| 收盘价列表要限长 | 否则长期运行内存持续增长 |
| 停牌 Bar 要跳过 | 停牌价格无意义，会污染指标窗口 |
| **参数总数 ≤ 4** | 自由度预算只有 4.3 个，超了得到的是记忆不是规律 |

## 2.4 复用组件

```python
from qmtquant.strategy.indicators import (
    MovingAverage, CrossDetector, IntradayVwap,
    AverageTrueRange, Donchian,
)
from qmtquant.strategy.trade_manager import TradeManager   # 止损/止盈/移动止损/仓位
```

`TradeManager` 提供逐仓风控：

```python
mgr = TradeManager(risk_per_trade=0.01,      # 每笔最多亏总资产的 1%
                   trailing_ratio=0.15,      # 移动止损
                   trailing_start_r=1.0)     # 盈利 1R 才启动

volume = mgr.position_size(total_value, entry, stop)  # 止损距离反推仓位
mgr.open(symbol, entry, volume, stop, bar_index=i)
reason = mgr.check(symbol, high, low, close, i)       # 返回离场原因或 None
```

---

# 三、回测

## 3.1 通用入口（不用写脚本）

```bash
cd /e/qmt && ./.venv/Scripts/python.exe scripts/run_strategy.py --strategy my_pkg.my_mod.MyStrategy --describe
```

```bash
cd /e/qmt && ./.venv/Scripts/python.exe scripts/run_strategy.py --strategy my_pkg.my_mod.MyStrategy --universe-csv data/universe/index_weight_000852.SH.csv --start 2016-01-01 --end 2026-08-27 --set lookback=60 --report reports/mine
```

## 3.2 标的池三选一 —— 这一项决定结论可不可信

| 选项 | 偏差 | 用于 |
|---|---|---|
| `--symbols 510300.SH` | 无（先验指定） | 择时、单 ETF |
| `--sector 沪深300` | 幸存者 + 成分前视 ⚠ | 快速试验 |
| `--universe-csv ...` | **全部消除** | **选股策略必用** |

实测同一组均值回归参数，换标的池后总收益从 **-4.49% 变成 +320.33%** —— 320 个百分点全是偏差。

可用的历史成分 CSV：

```
data/universe/index_weight_000300.SH.csv   沪深300   949 只 / 2005-2026
data/universe/index_weight_000905.SH.csv   中证500  1926 只 / 2005-2026
data/universe/index_weight_000852.SH.csv   中证1000 2839 只 / 2014-2026
```

⚠ 用中证500/1000 前要先补行情：本地日线只有 641 只，主要覆盖沪深300 历史成分。

## 3.3 其他开关

```
--interval 1d|1w|1m       周期。1m 只有 2025-08 起的数据
--start / --end           区间
--capital 1000000         初始资金
--no-drawdown             关闭回撤控制（默认开启）
--max-drawdown 0.20       回撤上限，超过判不合格
--benchmark 000300.SH     对标指数
--report reports/mine     输出目录，**不同策略务必分开，否则互相覆盖**
```

## 3.4 输出

| 文件 | 内容 |
|---|---|
| `<类名>_report.html` | 双击打开。净值/回撤曲线、KPI、成交明细，plotly 内联无需联网 |
| `<类名>_equity.csv` | 每日净值 |
| `<类名>_trades.csv` | 逐笔成交 |

报告末尾必看两块：**标的池偏差说明**（写着「存在 ⚠」则收益不可信）和**回撤合规**（✓/✗）。

## 3.5 引擎内置的约束

回测引擎默认执行这些，不需要策略自己处理：

- **T+1**：当日买入当日不可卖
- **涨跌停**：主板 10% / 创业板科创板 20% / 北交所 30%，按代码前缀判定
- **整手**：买入向下取整到 100 股，不足一手计入 `undersized_orders`
- **次日开盘撮合**：T 日收盘产生的信号在 T+1 开盘成交，避免前视
- **交易成本**：佣金万 2.5（最低 5 元）+ 印花税千 1（仅卖出）+ 过户费万 0.1 + 滑点
- **回撤控制**：三档（只平不开 6% / 强制减仓 9% / 全部平仓 12%）

## 3.6 直接用引擎（写自己的脚本时）

```python
from qmtquant.engine.backtest_engine import BacktestEngine
from qmtquant.datafeed.xt_feed import XtDataFeed
from qmtquant.universe.providers import HistoricalUniverse
from qmtquant.core.constants import Interval

feed = XtDataFeed(cfg.data.store_dir, cfg.data.dividend_type)
universe = HistoricalUniverse("data/universe/index_weight_000852.SH.csv")
bars = feed.load_bars(universe.all_symbols(), "2016-01-01", "2026-08-27",
                      Interval.DAILY)

engine = BacktestEngine(initial_capital=1_000_000, cost=cfg.cost,
                        drawdown=controller)
engine.load_data(bars)
engine.set_universe(universe)
engine.add_strategy(MyStrategy, symbols, {"lookback": 60})
stats = engine.run()

print(stats.summary())
print(universe.describe_bias().summary())
```

---

# 四、验证（决定能不能上实盘）

**回测赚钱不等于策略有效。** 参数网格扫一遍总能找出好看的组合。

```bash
cd /e/qmt && ./.venv/Scripts/python.exe scripts/validate_strategy.py --strategy my_pkg.my_mod.MyStrategy --grid '{"lookback": [20, 40, 60, 120], "max_holdings": [10, 20, 30]}' --universe-csv data/universe/index_weight_000852.SH.csv --start 2016-01-01 --end 2026-08-27 --split 2022-01-01
```

## 五道检验

| 检验 | 问题 | 不通过意味着 |
|---|---|---|
| **自由度** | 数据够不够拟合这么多参数 | **前置否决项** —— 后面几项都不能采信 |
| 参数平原 | 最优点周围也还行吗 | 孤峰 = 拟合噪声 |
| 样本外 | 没见过的区间还灵吗 | 只是记住了历史 |
| Walk-forward | 滚动地用过去选参数、未来验证 | 参数在时间上不稳定 |
| 成本敏感性 | 手续费翻几倍打平 | 靠低成本假设撑着 |

默认行为（都可以关，但不建议）：

- **选平原中心而非峰值**（`--pick-peak` 关闭）—— 在 N 组里挑最高分，即使全是噪声也会很好看
- **回撤约束在寻优前生效**（`--max-drawdown`）—— 违规组合直接剔除，不是选完再看
- **带回撤控制**（`--no-drawdown` 关闭）—— 不带的话验的是另一套系统

## 因子研究（选股策略先做这个）

一次 IC 分析几秒，一次回测几分钟，而且 IC 不掺杂仓位和成本：

```bash
cd /e/qmt && ./.venv/Scripts/python.exe scripts/analyze_factors.py --index 000852.SH --start 2016-01-01
```

看 `IC_t(NW)` 列（Newey-West 修正过的）。`|t| < 2` 就别往下做了。**必须同时看两段样本** —— 单段显著而另一段不显著的因子，实测在沪深300 上占多数。

---

# 五、实盘 / 模拟盘

## 5.1 配置

`config/config.yaml`（已在 `.gitignore`，不入库）：

```yaml
gateway:
  name: miniqmt              # sim / miniqmt
  qmt_path: "D:/qmtApp/userdata_mini"
  account_id: "62221162"
  account_type: STOCK

strategies:
  - name: MyStrategyLive
    class: my_pkg.my_mod.MyStrategy
    vt_symbols:
      - "000001.SZSE"
    setting:
      lookback: 60
```

## 5.2 启动

```bash
cd /e/qmt && ./.venv/Scripts/python.exe scripts/run_live.py --gateway sim
```

| 参数 | 说明 |
|---|---|
| `--gateway sim` | 本地模拟撮合，不接券商，**先用这个联调** |
| `--gateway miniqmt` | 接 miniQMT，需客户端已登录 |
| `--dry-run` | 启动即开急停，只跑行情不下单 |
| `--no-store` | 不持久化状态 |

`Ctrl+C` 优雅退出：撤单 → 停策略 → 断开网关。

## 5.3 实盘专有的机制

| 机制 | 说明 |
|---|---|
| `reconcile()` | 启动时与券商对账：持仓、活动委托 |
| `RiskManager` | 11 项盘前检查（单笔金额、单票占比、总仓位、当日笔数、当日成交额、日亏损线、ST、黑名单、回撤档位…） |
| 急停 | `risk_manager.activate_kill_switch(reason)`，立即停止所有开仓 |
| `StateStore` | 策略状态、成交流水落 SQLite（`data/state.db`）。**不持久化持仓** —— 券商是唯一真相 |
| 断线重连 | 指数退避，默认最多 10 次 |
| `daily_settle()` | 收盘结算，重置当日计数 |

## 5.4 上实盘前的顺序

1. 五道检验全过
2. `--no-drawdown` 关掉再跑一遍，确认回撤控制没把策略锁死
3. `scripts/test_order_flow.py` 跑通下单/撤单/成交
4. `--gateway sim` 全链路联调
5. `--gateway miniqmt --dry-run` 只看不下单，观察一天
6. 小资金实盘

---

# 六、内置策略短名

回测/验证时可以直接用短名代替完整路径：

| 短名 | 类 | 类型 |
|---|---|---|
| `mean_reversion` | MeanReversionStrategy | 选股 |
| `momentum` | MomentumRotationStrategy | 选股 |
| `low_turnover` | LowTurnoverStrategy | 选股 |
| `breakout` | BreakoutStrategy | 择时 |
| `index_timing` | IndexTimingStrategy | 择时（单 ETF） |
| `trend_ma` | TrendMaStrategy | 择时 |
| `ma_cross` | MaCrossStrategy | 择时 |
| `intraday_vwap` | IntradayVwapStrategy | 日内 |

自己的策略用完整路径 `包.模块.类名`，不需要改任何脚本。

⚠ 这八个**没有一个通过五道检验**，留着是当对照基准。详见 `docs/HOWTO_STRATEGY.md`。
