# 架构设计文档

项目：**qmtquant** — A 股个人量化交易系统
版本：v0.1

---

## 1. 设计原则

1. **事件驱动**：所有跨模块通信走 EventEngine，模块之间零直接依赖。
2. **网关抽象**：策略只面向 `BaseGateway` 接口，miniQMT / AMT / 模拟盘可插拔。
3. **回测实盘同构**：`BacktestEngine` 与 `LiveEngine` 向策略暴露完全相同的 API，策略代码不改一行即可切换。
4. **风控前置且不可绕过**：所有下单请求必须经过 `RiskManager` 校验，策略无法直连网关。
5. **配置外置**：无硬编码参数，全部走 `config/config.yaml`。
6. **纵深防御**：本地风控 + 网关侧校验 + Kill Switch 三层。

---

## 2. 总体架构

```
┌───────────────────────────────────────────────────────────────┐
│                        应用层 / 脚本入口                       │
│   run_live.py    run_backtest.py    run_daily_job.py          │
└───────────────────────┬───────────────────────────────────────┘
                        │
┌───────────────────────▼───────────────────────────────────────┐
│                         策略层  strategy/                      │
│   StrategyBase ── on_bar / on_tick / on_order / on_trade      │
│   examples: ma_cross, factor_rotation                          │
└───────────────────────┬───────────────────────────────────────┘
                        │  buy() / sell() / cancel()
┌───────────────────────▼───────────────────────────────────────┐
│                     引擎层  engine/                            │
│   ┌──────────────────┐        ┌──────────────────────────┐    │
│   │  LiveEngine      │        │  BacktestEngine          │    │
│   │  (实盘/模拟盘)    │        │  (历史回放 + 撮合模拟)    │    │
│   └────────┬─────────┘        └───────────┬──────────────┘    │
└────────────┼──────────────────────────────┼───────────────────┘
             │                              │
┌────────────▼──────────────┐   ┌───────────▼───────────────────┐
│      风控层 risk/          │   │   回测撮合 + 成本模型          │
│  RiskManager (前置校验)    │   │   A股规则: T+1/涨跌停/整手     │
│  KillSwitch                │   └───────────────────────────────┘
└────────────┬──────────────┘
             │ send_order / cancel_order
┌────────────▼──────────────────────────────────────────────────┐
│                      网关层  gateway/                          │
│  BaseGateway (抽象)                                            │
│   ├─ MiniQmtGateway   (xtquant.xttrader + xtdata)   ← 主通道   │
│   ├─ AmtGateway       (国金 AMT 算法执行)           ← 备用     │
│   └─ SimGateway       (本地撮合，联调用)                       │
└────────────┬──────────────────────────────────────────────────┘
             │ 回报: OrderData / TradeData / PositionData / AccountData
┌────────────▼──────────────────────────────────────────────────┐
│                    事件引擎  event/EventEngine                 │
│   EVENT_TICK  EVENT_BAR  EVENT_ORDER  EVENT_TRADE             │
│   EVENT_POSITION  EVENT_ACCOUNT  EVENT_LOG  EVENT_TIMER       │
└────────────┬──────────────────────────────────────────────────┘
             │
┌────────────▼──────────────────────────────────────────────────┐
│         数据层 datafeed/          持久层 store/                │
│  XtDataFeed (xtdata)             SQLite / Parquet             │
│  CsvDataFeed (离线)               BarRepository / TradeLog     │
│  行情订阅 / 历史下载 / 复权        交易日历 / 标的池            │
└───────────────────────────────────────────────────────────────┘
```

---

## 3. 模块职责

### 3.1 `core/` — 领域对象与常量
借鉴 vn.py 对象模型，保证将来可迁移。

| 对象 | 说明 |
|------|------|
| `TickData` | 最新价、买卖五档、成交量额、涨跌停价 |
| `BarData` | OHLCV + `Interval` |
| `OrderRequest` / `CancelRequest` | 下单/撤单请求 |
| `OrderData` | 订单状态（`SUBMITTING/NOTTRADED/PARTTRADED/ALLTRADED/CANCELLED/REJECTED`） |
| `TradeData` | 成交回报 |
| `PositionData` | 持仓量、可用量（T+1 冻结）、成本价、浮盈 |
| `AccountData` | 总资产、可用资金、市值 |
| `ContractData` | 标的信息、最小变动价位、涨跌停 |

`vt_symbol` 统一格式：`000001.SZSE` / `600000.SSE`，在网关层与 xtquant 的 `000001.SZ` 互转。

### 3.2 `event/` — 事件引擎
- 单线程事件循环 + `queue.Queue`，独立定时器线程发 `EVENT_TIMER`（1s）。
- `register(type, handler)` / `unregister` / `put(event)`。
- **Handler 内异常必须捕获并记日志，绝不允许冒泡终止事件循环**。
- 按事件类型分发 + 通用监听器（`register_general`）。

### 3.3 `datafeed/` — 数据
- `BaseDataFeed`：`download_history()` / `load_bars()` / `subscribe()`。
- `XtDataFeed`：封装 `xtdata.download_history_data2` / `get_market_data_ex` / `subscribe_quote`。
- 复权：统一在 DataFeed 层完成（`dividend_type = front/back/none`）。
- 落库：Parquet（按 `symbol/period/year` 分区）+ SQLite 存元数据与交易日历。

### 3.4 `gateway/` — 交易网关

```python
class BaseGateway(ABC):
    def connect(self, setting: dict) -> None
    def subscribe(self, req: SubscribeRequest) -> None
    def send_order(self, req: OrderRequest) -> str      # 返回 vt_orderid
    def cancel_order(self, req: CancelRequest) -> None
    def query_account(self) -> None
    def query_position(self) -> None
    def close(self) -> None
    # 回调统一 put 事件
    def on_tick / on_order / on_trade / on_position / on_account
```

**MiniQmtGateway 要点**
- `XtQuantTrader(path, session_id)` + `XtQuantTraderCallback` 异步回报。
- `session_id` 每次启动用时间戳生成，避免冲突。
- 启动流程：`start()` → `connect()` → `subscribe(account)` → 查资产/持仓/委托做**本地对账**。
- 断线：`on_disconnected` 回调触发重连（指数退避），重连后全量对账，**期间禁止新开仓**。
- 本地维护 `orderid → 券商 order_id` 映射，撤单靠映射表。

**SimGateway 要点**
- 订阅行情后，用下一根 Bar 的开盘价 + 滑点撮合，模拟 T+1 与涨跌停。

### 3.5 `risk/` — 风控
下单链路：`Strategy.buy() → Engine.send_order() → RiskManager.check() → Gateway.send_order()`

校验项（任一不过则拒单并记日志+告警）：
1. Kill Switch 是否开启
2. 单笔金额 / 单票市值占比 / 总仓位上限
3. 当日下单笔数、当日成交金额上限
4. 当日浮亏是否触及阈值（触及则只允许卖出）
5. 黑名单（ST / 停牌 / 退市整理 / 新股上市首日）
6. 可用资金、可卖数量（T+1 冻结）
7. 价格是否在涨跌停区间内、数量是否 100 股整数倍（卖出可零股）

### 3.6 `engine/`
- `LiveEngine`：装配 EventEngine + Gateway + RiskManager + 策略集合；管理策略生命周期、行情订阅、回报路由（按 `vt_orderid` 找到归属策略）。
- `BacktestEngine`：历史 Bar 回放 → 调用策略 `on_bar` → 收集订单 → 下一 Bar 撮合 → 更新组合 → 出报告。

### 3.7 `store/` — 持久化
SQLite 表：`bar_meta` / `trade_log` / `order_log` / `daily_pnl` / `strategy_state` / `calendar`。
策略状态（持仓、变量）每次变更落库，进程重启可恢复。

---

## 4. 关键流程

### 4.1 实盘启动
```
加载 config → 初始化 EventEngine → 初始化 Gateway 并 connect
  → 查询账户/持仓/委托，与本地状态对账
  → 初始化 RiskManager（读入当日额度）
  → 加载并 init 各策略 → 订阅所需标的行情
  → 等待 09:15 集合竞价数据 → 09:30 start 策略
  → 15:00 停止策略 → 盘后结算 → 生成报告 → 退出
```

### 4.2 下单
```
Strategy.buy(symbol, price, volume)
  → LiveEngine 生成 OrderRequest（带 strategy_name）
  → RiskManager.check()  ──拒绝──> 记日志 + 告警 + 返回空
  → Gateway.send_order() → 券商
  → 回报 EVENT_ORDER/EVENT_TRADE → LiveEngine 按 vt_orderid 路由
  → Strategy.on_order/on_trade → 更新策略持仓 → 落库
```

### 4.3 异常与恢复
| 场景 | 处理 |
|------|------|
| 网关断线 | 重连 + 全量对账 + 期间禁止开仓 |
| 进程崩溃 | 重启后从 `strategy_state` 恢复 + 与券商持仓对账，不一致则告警并进入只读模式 |
| 行情停推 > 60s（交易时段） | 告警，重新订阅 |
| 风控触发日内亏损线 | Kill Switch 半开：撤全部买单，只允许卖出 |

---

## 5. 目录结构

```
E:\qmt\
├── docs/                    # 架构 / 需求 / 实现 / 选型
├── config/
│   ├── config.example.yaml  # 模板（入库）
│   └── config.yaml          # 真实配置（.gitignore）
├── qmtquant/
│   ├── core/       objects.py  constants.py
│   ├── event/      engine.py
│   ├── datafeed/   base.py  xt_feed.py  csv_feed.py
│   ├── gateway/    base.py  miniqmt_gateway.py  sim_gateway.py  amt_gateway.py
│   ├── risk/       risk_manager.py
│   ├── engine/     live_engine.py  backtest_engine.py  performance.py
│   ├── strategy/   base.py  examples/ma_cross.py
│   ├── store/      database.py
│   ├── utils/      logger.py  notifier.py  calendar.py
│   └── config.py
├── scripts/        run_backtest.py  run_live.py  download_data.py
├── tests/
├── data/  logs/    # .gitignore
└── requirements.txt
```

---

## 6. 技术选型

| 领域 | 选择 | 理由 |
|------|------|------|
| 语言 | Python 3.11 | xtquant 生态；3.11 性能提升明显 |
| 行情/交易 | xtquant (miniQMT) | 官方 SDK，稳定 |
| 数据处理 | pandas + numpy + pyarrow | 生态成熟 |
| 存储 | SQLite + Parquet | 单机零运维；Tick 量大后换 DuckDB |
| 配置 | PyYAML + dataclass | 类型安全 |
| 日志 | logging + RotatingFileHandler | 标准库够用 |
| 调度 | APScheduler | 盘前/盘中/盘后任务 |
| 回测绘图 | matplotlib / plotly | 生成 HTML 报告 |
| 测试 | pytest | — |
| 因子研究（后期） | Qlib | 独立进程，产出信号文件给实盘消费 |

---

## 7. 演进路线

- **v0.1** 内核 + SimGateway + 回测 + 双均线示例
- **v0.2** MiniQmtGateway + 风控 + 模拟盘实盘
- **v0.3** 监控告警 + 定时调度 + 盘后报告
- **v0.4** 多策略组合、资金分配、Qlib 因子接入
- **v0.5** AMT 算法执行通道、Tick 级回测、Web 监控面板
