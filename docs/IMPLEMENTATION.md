# 实现文档

项目：**qmtquant**
面向：负责搭建、接入、上线本系统的人（即你自己）

---

## 1. 环境搭建

### 1.1 基础环境

```bash
git clone git@github.com:MteCode/qmt-quant.git
cd qmt-quant
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

验证（不需要 QMT）：

```bash
pytest tests/ -q
```

```bash
python scripts/run_backtest.py --mock
```

### 1.2 接入 xtquant

`xtquant` 是迅投官方的 Python 接口包。取得方式有两种，**取决于你的客户端有没有打包它**：

**方式 A：pip 安装（多数情况用这个）**

PyPI 上的 `xtquant` 是迅投官方发布的，版本号是发布日期（如 `250807.1.2`）：

```bash
pip install xtquant
```

**方式 B：从客户端目录复制**

有些券商定制版会把包放在 `<QMT安装目录>\bin.x64\Lib\site-packages\xtquant`，
若存在则优先用它（与客户端版本严格一致）：

```bash
cp -r "D:/qmtApp/bin.x64/Lib/site-packages/xtquant" .venv/Lib/site-packages/
```

> 注意：并非所有客户端都打包了 Python 端。实测国金 miniQMT（`D:\qmtApp`）的
> `bin.x64` 下**没有** `Lib` 目录，只有 `XtQuantServer.dll` 服务端，此时必须走方式 A。

验证：

```bash
python -c "from xtquant import xtdata; print(xtdata.__file__); import xtquant; print(xtquant.__version__)"
```

连通性验证（客户端需已登录）：

```bash
python -c "from xtquant import xtdata; print(len(xtdata.get_trading_dates('SH','20240101','20240110')))"
```

看到 `xtdata连接成功` 和交易日数量即为正常。

> 客户端大版本升级后建议 `pip install -U xtquant`，接口不匹配可能导致
> 连接成功但下单静默失败。

### 1.3 位数必须匹配

miniQMT 是 64 位，Python 也必须是 64 位。混用会在 `import xtquant` 时报 DLL 加载失败。

```bash
python -c "import platform; print(platform.architecture())"
```

---

## 2. 三条通道怎么用

| 通道 | 定位 | 什么时候用 |
|------|------|-----------|
| **miniQMT** | 主通道，行情 + 交易 | 日常策略自动执行，策略跑在自己的 Python 进程里 |
| **大 QMT** | 人工界面 | 盯盘、看持仓、程序出问题时手工撤单/平仓兜底 |
| **国金 AMT** | 算法执行 | 单笔量大需要拆单（TWAP/VWAP）、多账户同步下单 |

**关键点**：miniQMT 和大 QMT 用的是同一套底层，**同一账号同一时间只能被一个客户端登录**。
日常跑策略用 miniQMT（轻量、常驻），需要人工干预时再切大 QMT。

### 2.1 miniQMT 启动前置条件

1. 客户端已启动并**完成登录**（未登录时 `connect()` 返回非 0）
2. 券商已开通「量化交易权限」——需要联系客户经理开通，不是默认有的
3. 配置里的 `qmt_path` 指向 **`userdata_mini` 目录**，不是安装根目录

```yaml
gateway:
  name: miniqmt
  qmt_path: "D:\\国金QMT交易端模拟\\userdata_mini"
  account_id: "你的资金账号"
```

### 2.2 session_id 的坑

`XtQuantTrader(path, session_id)` 的 `session_id` **每次启动必须不同**，复用会导致连接被拒或回报错乱。
本项目用 `int(time.time())` 生成，见 `gateway/miniqmt_gateway.py:connect()`。

---

## 3. 数据接入

### 3.1 下载历史数据

```bash
python scripts/download_data.py --symbols 000001.SZ,600519.SH --start 2020-01-01
```

数据落到 `data/{周期}/{交易所}/{代码}.parquet`，之后回测**不再依赖 QMT 客户端**。

### 3.1.1 各周期的历史深度（实测）

在国金 miniQMT + 迅投数据源上实测（2026-08）：

| 周期 | 可回溯范围 | 说明 |
|------|-----------|------|
| 日线 / 周线 / 月线 | **到标的上市日** | 如 600519 可取到 2001-08-27 |
| 1m / 5m / 15m / 30m / 1h | **仅最近约 1 年** | 无论请求多早的起始日期，都只返回近 1 年 |

分钟级的限制是**券商/数据源侧**的，不是本项目的 bug。请求 2020 年起的 1 分钟线，
实际只会拿到最近 1 年。需要更长分钟历史，得向券商申请更高数据权限或购买迅投数据包。

**这对策略设计的影响**：分钟级策略的样本外验证窗口很短（1 年内），
过拟合风险显著高于日线策略。建议分钟级策略先在日线上验证逻辑成立，再下沉到分钟级。

实测磁盘占用（沪深300 全量，Parquet 压缩后）：

| 周期 | 占用 | 下载耗时 |
|------|------|---------|
| 日线（6.7 年） | 25.6 MB | ~25 秒 |
| 周线（合成） | 5.9 MB | ~3 秒 |
| 1 分钟线（1 年） | 545.7 MB | ~9 分钟 |

### 3.1.2 板块数据为空

全新安装的客户端，`get_stock_list_in_sector("沪深300")` 会返回 **0 只** ——
板块数据是本地缓存，需要先拉一次：

```python
from xtquant import xtdata
xtdata.download_sector_data()
```

本项目的 `get_sector_stocks()` 已内置这一步，取到空列表时会自动下载后重试。

### 3.2 复权处理

统一在 `XtDataFeed` 层完成，由 `data.dividend_type` 控制：

| 取值 | 说明 | 适用 |
|------|------|------|
| `front` | 前复权 | **回测默认用这个**，价格连续，指标计算不失真 |
| `back` | 后复权 | 长周期收益率计算 |
| `none` | 不复权 | 实盘下单价格必须用不复权的真实价 |

> **注意**：回测用前复权价算信号没问题，但实盘下单时 `bar.close_price` 必须是真实价格。
> 实盘的行情来自 tick 推送，天然是不复权的，不会有这个问题。

### 3.3 停牌识别

xtdata 停牌日的 `volume` 为 0，`XtDataFeed` 据此设置 `BarData.suspended`，回测撮合时跳过。

---

## 4. 回测：怎么保证结果可信

### 4.1 前视偏差的防范机制

这是回测最容易骗自己的地方。本项目的做法：

```
T 日 Bar 推给策略 → 策略调 buy() → 请求进入 pending_orders（不成交）
                                          ↓
T+1 日 Bar 到达 → 用 T+1 的【开盘价】撮合 pending_orders → 再推 T+1 Bar 给策略
```

代码见 `engine/backtest_engine.py:run()` 的第 2、3 步顺序，**这个顺序不能调换**。
对应测试：`tests/test_backtest.py::test_no_lookahead_fill_on_next_open`。

### 4.2 A 股规则的实现位置

| 规则 | 实现 |
|------|------|
| T+1 | `_settle_t1()` 每个新交易日把 `available` 重置为 `volume`；买入时只加 `volume` 不加 `available` |
| 100 股整数倍 | `send_order()` 中买入向下取整；不足 100 股直接不下单 |
| 涨跌停 | `_match_pending()` 用**前一根 Bar 的收盘价**算涨跌停，开盘价触及则拒单 |
| 停牌 | `bar.suspended` 为真时跳过撮合 |
| 交易成本 | `calc_cost()`：佣金（有最低 5 元）+ 印花税（仅卖出）+ 过户费 |

**已知限制**：涨跌停幅度目前是全局 10%（`price_limit_ratio`），
创业板/科创板 20%、ST 股 5%、北交所 30% 需要按标的区分 —— 这是 v0.2 待办。

### 4.3 选股回测的三种偏差（实测数据）

选股回测最大的陷阱不在撮合，而在**标的池**。`vt_symbols = get_sector_stocks("沪深300")`
这一行看着人畜无害，实际同时引入三种偏差：

| 偏差 | 说明 | 本项目能否消除 |
|------|------|--------------|
| **上市日前视** | 用了 2025 年才上市的股票去跑 2021 年的回测 | ✅ `PointInTimeUniverse` |
| **成分股前视** | 2021 年就用上了 2026 年才确定的成分名单 | ❌ 需外部历史成分数据 |
| **幸存者偏差** | 这些年退市/被调出指数的股票完全缺失 | ❌ QMT 数据源中退市股不存在 |

**实测影响**（沪深300 动量轮动，2021-01 ~ 2026-08，持仓 10 只，20 日调仓）：

| | 总收益 | 年化 | Sharpe |
|---|---|---|---|
| 关闭上市日过滤 | 64.30% | 9.17% | 0.368 |
| 启用上市日过滤 | 38.38% | 5.91% | 0.287 |

**仅上市日过滤这一项，就让收益从 64% 降到 38%** —— 被虚增的 26 个百分点，
纯粹来自「买了当时还不存在的股票」。而幸存者偏差和成分股前视在两个版本里都还在，
真实衰减只会更大。

#### 为什么 QMT 消除不了幸存者偏差

实测确认（2026-08）：

```python
xtdata.get_instrument_detail("300104.SZ")  # 乐视网，2020-07 退市
# -> None，没有任何元数据
"300104.SZ" in xtdata.get_stock_list_in_sector("沪深A股")  # -> False
```

退市股在数据源里**完全不存在**，不是「数据缺失」而是「标的缺失」。
`get_stock_list_in_sector(sector, real_timetag)` 虽然有日期参数，
但实测传 2020/2022 的时间戳返回的成分与当前**完全相同**（差异 0 只），
说明本地板块缓存只有当前快照。

#### 怎么办

1. **短期**：用 `PointInTimeUniverse` 至少修掉上市日前视，并在报告里
   显式打印 `BiasReport`，让每次看回测结果的人都知道还剩哪些偏差。
2. **正解**：取一份含退市标的的历史成分股 CSV，用 `HistoricalUniverse` 加载。
   数据来源：akshare 的 `index_stock_cons_csindex`、Tushare Pro 的 `index_weight`、
   中证指数官网历史成分文件。CSV 格式见 `providers.py:HistoricalUniverse` 文档。
3. **心理准备**：即使标的池干净了，有偏差的回测收益通常会再打对折。
   把回测当作「排除烂策略的筛子」，而不是「预测收益的工具」。

### 4.4 避免过拟合的检查清单

- [ ] 参数在邻域内是否稳定？（参数平原 vs 参数尖峰 —— 尖峰基本就是过拟合）
- [ ] 样本外（最近 1-2 年不参与调参）表现如何？
- [ ] 滚动前推验证（walk-forward）是否一致？
- [ ] 换手率是否高到手续费吃掉大部分收益？看 `total_commission / 初始资金`
- [ ] 成交笔数是否太少？少于 30 笔的统计结论没有意义
- [ ] 把滑点调大一倍，策略是否还赚钱？不赚说明利润全在噪音里

---

## 5. 风控：为什么它必须独立于策略

策略是会写错的 —— 一个符号写反就可能满仓反向操作。
所以风控做成**策略无法绕过的前置拦截**：策略只能调 `self.buy()`，它内部走 engine，
engine 必过 `RiskManager.check()`，策略拿不到 gateway 的引用。

### 5.1 校验顺序

```
急停 → 只平不开状态 → 数量合法性 → 黑名单 → 当日额度
     → 单笔金额 → 可用资金 → 单票占比 → 总仓位   （买入）
     → 可卖数量（T+1 冻结后的 available）        （卖出）
```

### 5.2 首次实盘的建议参数

**把额度调到"就算完全错了也不心疼"的水平**：

```yaml
risk:
  max_order_value: 5000           # 单笔最多 5000 元
  max_position_ratio: 0.05        # 单票不超过 5%
  max_order_count_per_day: 20     # 一天最多 20 笔
  max_turnover_per_day: 50000
  daily_loss_limit_ratio: 0.01    # 亏 1% 就停手
```

跑满一个月、对账无误后再逐步放大。

### 5.3 急停

```python
risk_manager.activate_kill_switch("行情异常")
```

触发后拒绝一切下单。实盘引擎应同时撤销所有活动委托。

---

## 6. 上线检查清单

### 上线前

- [ ] `pytest tests/ -q` 全绿
- [ ] `config/config.yaml` 已配置且**不在 git 追踪中**（`git status` 看不到它）
- [ ] 券商量化权限已开通
- [ ] 用**模拟账户**跑通完整一天：开盘订阅 → 下单 → 成交回报 → 盘后对账
- [ ] 风控额度已调到最小
- [ ] 日志目录可写，`logs/trade.log` 有内容

### 每日盘前（09:00）

- [ ] QMT 客户端已登录
- [ ] 增量下载昨日数据
- [ ] 检查持仓与券商是否一致
- [ ] 风控计数已重置（`RiskManager.new_day()`）

### 每日盘后（15:30）

- [ ] 本地成交记录与券商对账单核对，笔数与金额是否一致
- [ ] 检查是否有未撤销的挂单
- [ ] 检查风控拒单日志，确认拒单原因是否符合预期
- [ ] 生成当日盈亏报告

---

## 7. 常见问题

| 现象 | 原因 | 处理 |
|------|------|------|
| `connect()` 返回非 0 | 客户端未登录 / 路径错 | 确认已登录，`qmt_path` 指向 `userdata_mini` |
| `import xtquant` 报 DLL 错误 | Python 是 32 位 | 换 64 位 Python |
| 下单返回负数 | 无量化权限 / 账号类型错 | 联系客户经理开通；确认 `account_type` |
| 连接成功但下单无反应 | xtquant 版本与客户端不匹配 | 重新从客户端目录复制 xtquant |
| 收不到行情回调 | 未调 `subscribe_quote` / 非交易时段 | 检查订阅、检查时间 |
| 回测收益好得离谱 | 大概率前视偏差或过拟合 | 对照 4.1、4.3 逐条排查 |
| 终端中文乱码 | Windows 控制台代码页是 GBK | `chcp 65001`，或直接看 `logs/` 里的文件 |
| K 线日期比实际早一天 | xtdata 时间戳时区处理错误 | 见下方「时区陷阱」 |
| 分钟线只有 1 年 | 券商侧限制，非 bug | 见 3.1.1 |
| 沪深300 成分股取到 0 只 | 板块缓存为空 | 见 3.1.2 |
| `config.yaml` 报 `unknown escape character` | YAML 双引号里 `\q` `\u` 被当转义符 | 路径改用正斜杠 `D:/qmtApp/userdata_mini` |

### 7.1 时区陷阱（严重，会静默污染回测）

`xtdata` 返回的 `time` 字段是 **UTC 毫秒时间戳，但它表示的是北京时间的那一刻**。

```python
# 错误 —— 得到 UTC
pd.to_datetime(df["time"], unit="ms")
# 2020-01-02 的日线 → 2020-01-01（元旦，休市日）
# 09:31 的分钟线 → 01:31

# 正确
(pd.to_datetime(df["time"], unit="ms", utc=True)
   .dt.tz_convert("Asia/Shanghai").dt.tz_localize(None))
```

危害在于**它不报错**：日线整体前移一天，周线归错周，回测用错日期却看不出异常。
本项目已在 `XtDataFeed._normalize_df()` 中修正，并有回归测试
（`tests/test_datafeed.py::TestTimezone`）锁住。

自查方法：日线的第一根不应落在节假日，且所有日期的 `weekday() < 5`。

---

## 8. 二次开发指引

### 加一个新策略

1. 在 `qmtquant/strategy/` 下新建文件，继承 `StrategyBase`
2. `parameters` 声明可调参数，`variables` 声明需持久化的变量
3. 实现 `on_bar`（择时型）或 `on_bars`（选股型）
4. 在 `config.yaml` 的 `strategies` 里注册

### 加一个新网关

1. 继承 `gateway/base.py:BaseGateway`
2. 实现 `connect / close / subscribe / send_order / cancel_order / query_account / query_position`
3. 所有回报通过 `self.on_order()` / `self.on_trade()` 等推入事件引擎
4. **把该券商 SDK 的所有调用都关在这个文件里** —— 这样 SDK 升级只需改一处

### 加一个风控规则

在 `risk/risk_manager.py:_do_check()` 中增加分支，返回对应的 `RejectReason`；
同时在 `core/constants.py:RejectReason` 里加枚举值，并补一条单测。
