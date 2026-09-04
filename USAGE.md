# 系统使用说明

一人操作、50 万本金、A 股日频。所有操作都能从管理台点，也都能用命令行跑。

## 启动管理台

```bash
cd E:\qmt
.\.venv\Scripts\python.exe -m webui.app
```

浏览器打开 `http://127.0.0.1:8800`。八个页面：

| 页面 | 看什么 |
|------|--------|
| 总览 | 净值与回撤曲线、模型产物、风控档位、关键指标 |
| 策略选股 | 回测各调仓日买了卖了什么、行业分布 |
| 实盘执行 | 账户实际持仓、委托记录、与信号的差异 |
| 数据核对 | 二进制数据可视化、自动查错、一致性扫描 |
| 实验结果 | 多种子方差、集成收敛、参数网格、跨期稳定性 |
| 运行任务 | 一键触发训练/回测/下单 |
| 定时任务 | 盘前盘后调度 |
| 运行记录 | 历史任务与实时日志 |

管理台只监听本机。可执行的动作走白名单，下单类需二次确认。

---

## 每日操作

### 交易日时间线

| 时间 | 动作 | 命令 |
|------|------|------|
| 08:40 | 盘前补齐行情 | 定时自动 |
| 09:15 | 生成信号 | `generate_signal.py` |
| 09:31 | 预览 → 下单 | `paper_trade.py --dry-run` → 去掉 `--dry-run` |
| 15:20 | 盘后更新行情 | 定时自动 |
| 15:30 | 记净值 + 对账 | `track_equity.py`、`reconcile.py` |

行情更新已配成定时任务（管理台「定时任务」页可改时间）。
信号与下单**刻意不做定时** —— 下单产生真实委托，应人工确认。

### 命令行版本

```bash
cd E:\qmt

# 1 更新行情（下载 -> 清洗 -> 导出 Qlib -> 建库）
.\.venv\Scripts\python.exe scripts\update_market_data.py --mode post

# 2 生成信号
.\.venv\Scripts\python.exe strategies\alstm_ppo_csi1000\generate_signal.py

# 3 预览（跑完整风控，只是不提交委托）
.\.venv\Scripts\python.exe strategies\alstm_ppo_csi1000\paper_trade.py --dry-run

# 4 确认无误后执行
.\.venv\Scripts\python.exe strategies\alstm_ppo_csi1000\paper_trade.py

# 5 收盘后：记净值、对账
.\.venv\Scripts\python.exe strategies\alstm_ppo_csi1000\track_equity.py
.\.venv\Scripts\python.exe strategies\alstm_ppo_csi1000\reconcile.py
```

第 5 步不能省。**净值漏记补不回来** —— 券商查不到历史序列；
不对账则不知道成交率与滑点，实盘落后回测时找不到原因。

---

## 数据

```
data/1d/          原始下载，不可变，可回溯
      │  scripts/clean_data.py     唯一清洗出口
data/clean/       清洗层，唯一真相源
      ├──> data/qlib_data/   训练与回测（二进制，快 25 倍）
      └──> data/market.db    查询与核对（SQL）
```

### 核对数据是否干净

```bash
sqlite3 data\market.db < scripts\check_clean.sql
```

或在管理台「数据核对」页看。七组查询：脏数据残留、清洗动作汇总、
被洗最多的标的、交叉核对、数据新鲜度、因子滞后、随机抽样。

**验证的关键是两侧对得上**：清洗日志说清掉了 58,342 行非正价格，
库里查一条都没有 —— 只看日志不算数。

### 手动跑数据链路

```bash
.\.venv\Scripts\python.exe scripts\clean_data.py          # 增量清洗
.\.venv\Scripts\python.exe scripts\clean_data.py --report # 只看报告
.\.venv\Scripts\python.exe scripts\build_database.py      # 增量建库
```

---

## 研究：判断一个策略有没有 alpha

这套流程两小时内能否掉一个策略，比任何单次回测都重要。

```bash
S=strategies\alstm_ppo_csi1000

# 1 多种子 —— 换个种子结论是否翻转
.\.venv\Scripts\python.exe $S\seed_experiment.py --seeds 8

# 2 集成规模 —— 方差是否随规模收敛（不收敛=模型高度相关）
.\.venv\Scripts\python.exe $S\ensemble_scaling.py

# 3 参数网格 —— 是连片区域还是孤立尖刺
.\.venv\Scripts\python.exe $S\sweep_portfolio.py

# 4 分段检验 —— 前后两段符号是否一致
.\.venv\Scripts\python.exe $S\sweep_portfolio.py --start 2022-01-01 --end 2024-02-29 --tag _h1
.\.venv\Scripts\python.exe $S\sweep_portfolio.py --start 2024-03-01 --end 2026-08-27 --tag _h2
```

结果在管理台「实验结果」页可视化。

### 判读标准

| 现象 | 结论 |
|------|------|
| Sharpe 分布横跨零轴 | 单次数字不可信 |
| 箱体不随规模收窄 | 模型间高度相关，集成无效 |
| 网格里孤立亮格 | 参数挖掘，换个参数就崩 |
| 分段符号翻转 | 不具跨期稳定性 |

### 因子评估

```bash
.\.venv\Scripts\python.exe $S\eval_factors.py
```

三关：IC 显著（Newey-West 修正）、分段不翻转、与现有信号正交。
外加两道硬门槛：**可获得滞后**（防前视）、**数据完整度**（≥60% 交易日有数据）。

---

## 训练

```bash
# 选股（LightGBM 集成，90 秒/种子）
.\.venv\Scripts\python.exe strategies\lgb_agents_ppo\train_screen.py --seeds 5

# 择时（PPO，16 分钟/种子）
.\.venv\Scripts\python.exe $S\train_ppo.py --seed 0 --tag _s0
```

改了选股分数**必须重跑 PPO**，否则两者不匹配。

---

## 当前状态

### 能用的

- 数据链路：下载 → 清洗 → 双出口存储 → SQL 核对，全部打通
- 回测：T+1、涨跌停、按真实价整手、停牌、成本、回撤三档
- 风控：下单前置 + 日内日亏线 + 跨日回撤三档 + 盘中常驻，状态跨进程共享
- 实盘：miniQMT 网关、先卖后买、持仓快照、净值跟踪、成交对账
- 研究：四类检验 + 因子评估框架

### 已知不可靠的

**PPO 择时层方差极大**，正在量化：

| 条件 | 平均仓位 | 最大回撤 | Sharpe |
|------|---------|---------|--------|
| ALSTM 分数 / 种子 42 | 30.0% | 13.96% | 0.899 |
| LightGBM 分数 / 种子 42 | 70.2% | 33.29% | 0.418 |
| ALSTM 分数 / 种子 0 | **0.0%** | — | 0.000 |

同一套代码，仅换种子或分数源，学出的策略天差地别 ——
种子 0 直接学成永远空仓。**在多种子实验做完之前，不应把 PPO 当可用组件。**

### 策略本身：尚无可上实盘的

两个模型族（ALSTM 神经网络、LightGBM 树模型）IC 质量差 40%、
稳定性差 17 倍，但组合层面撞在同一堵墙上：

- 参数网格中位数为负，最优点是孤立尖刺
- 分段检验 0/12 配置在两段都为正
- 2022-01~2024-02（中证 1000 深度熊市）所有配置全部亏钱

**瓶颈确定不在模型**。满仓做多小盘股在熊市里无解，
这是策略风格问题，需要择时或换风格来解决。

---

## 出问题时

| 症状 | 查什么 |
|------|--------|
| 训练报 KeyError | 管理台「数据核对」看一致性，多半是数据撕裂 |
| 信号为空 | 检查 `models/*_scores.parquet` 是否存在且日期够新 |
| 下单全被拒 | 看风控日志，可能触发了回撤档位或日亏线 |
| 实盘落后回测 | 跑 `reconcile.py` 看成交率与滑点 |
| 买不进高价股 | 检查 `data/1d_raw/` 是否有不复权价（整手按真实价取整需要） |

更详细的架构与缺口清单见 [ARCHITECTURE.md](ARCHITECTURE.md)。
