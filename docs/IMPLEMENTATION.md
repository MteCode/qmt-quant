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

### 1.2 接入 xtquant（关键，最容易踩坑）

`xtquant` **不能用 pip 安装** —— PyPI 上同名的第三方包不是券商官方的，版本对不上会导致下单行为异常。

正确做法，二选一：

**方式 A：复制到虚拟环境（推荐）**

```
从：<QMT安装目录>\bin.x64\Lib\site-packages\xtquant
到：<项目>\.venv\Lib\site-packages\xtquant
```

**方式 B：加 PYTHONPATH**

```bash
set PYTHONPATH=D:\国金QMT交易端模拟\bin.x64\Lib\site-packages;%PYTHONPATH%
```

验证：

```bash
python -c "from xtquant import xtdata; print(xtdata.__file__)"
```

> **每次 QMT 客户端升级后都要重新复制一遍**，否则会出现连接成功但下单静默失败的情况。

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

### 4.3 避免过拟合的检查清单

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
