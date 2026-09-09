# 从训练模型到实盘下单

这份文档写的是**这套系统实际怎么走完一整条链路**，不是理想流程。
每一步都标了「怎么确认它真的成了」——因为这套系统里最常见的失败
不是报错，是静默地什么都没做，或者做了但用的是错的数据。

> 全文出现的所有数字与路径都可以在仓库里核对。凡是我不确定的地方，
> 都写明了不确定，而不是给一个看起来确定的说法。

---

## 0. 先明确一件事：链路能跑通 ≠ 策略能赚钱

这套系统里目前**没有一个策略被验证有正 alpha**。做 T 方向的研究
（22000+ 组参数、样本外 0/20 存活、毛 edge 中位数约 0.00%）结论是否定的，
管理台上那几个「研究」类条目的 `caveat` 里写着具体数字。

所以本文档教的是**怎么安全地把一条链路接通并观察它**，
不是「怎么开始赚钱」。真金白银上之前，先把 §7 的检查清单走一遍。

---

## 1. 全景

```
数据         模型            信号                执行              监控
────         ────            ────                ────              ────
下载 →  训练 →  回测 →  生成 target_latest.csv →  引擎读信号 →  风控 →  网关 →  券商
                                                                    ↓
                                                              成交回报 → 状态库
                                                                    ↓
                                                            盘后对账 / 归因
```

三个**必须分开理解**的东西：

| | 是什么 | 在哪 |
|---|---|---|
| **任务（Task）** | 一次性的脚本运行（训练、回测、生成信号） | `webui/registry.py` |
| **服务（Service）** | 长期运行的进程（调度器、实盘引擎） | `webui/services.py` |
| **策略（Strategy）** | 引擎里跑的对象，消费行情产生委托 | `config.yaml` 的 `strategies` |

管理台上「跑了个回测」和「策略在实盘跑着」是两回事。
前者是任务，后者要求 `live_qmt` 服务在运行**且**策略登记在 `config.yaml` 里。

---

## 2. 数据

```bash
python scripts/download_data.py
```

**怎么确认它真的成了**：

```bash
python scripts/check_data.py --sector 沪深300 --interval 1d
```

这个检查会报 OHLC 逻辑错误、异常跳空、重复、乱序，以及**数据停更**。

停更那条是跨标的判断：基准取全体标的里最新的日期，而不是今天。
原因是拿今天比会产生三种假警报——退市股停在 2017 年是正确的、
今天可能是周末、所有标的都停在同一天时那天就是最后一个交易日。

- 落后一年以内 → `错误`，下载可能挂了，**要处理**
- 落后一年以上 → `提示`，多为退市，正常

⚠️ 训练用的标的池已按规范排除**北交所、科创板（688xxx）、
1 手买入超过 5 万的高价股**。改这条规则前先想清楚为什么。

---

## 3. 训练

以全市场日内 GBM 为例：

```bash
python scripts/train_intraday_gbm.py
```

**怎么确认**：`models/intraday_gbm/metrics.json` 里的 `test_auc`。
样本外 AUC 接近 0.5 就是没学到东西，不要因为「训练完成」就往下走。

训练完模型只是个打分器，它**不会自己下单**。

---

## 4. 回测

```bash
python scripts/backtest_intraday_gbm.py
```

### 回测的成本口径（改过，注意）

成本的唯一事实源是 `qmtquant/core/costs.py`。当前口径：

| 项 | 值 | 备注 |
|---|---|---|
| 佣金 | 万 0.854 | **最低 5 元/笔** |
| 印花税 | 0.05% | 仅卖出，2023-08-28 起减半，法定不可谈 |
| 过户费 | 万 0.1 | 双边 |
| 滑点 | 万 5 | 单边，相对值 |

**最低佣金是做 T 类策略的隐形杀手**：单笔低于
`5 / 0.0000854 = 58,548` 元就触发下限，实际费率高于名义——
单笔 2.5 万时往返 0.192%，1 万时 0.252%，而名义只有 0.169%。
做 T 天然把资金拆成小单，几乎全落在这一段。

滑点在回测引擎里是**相对值**而不是「几个 tick」。因为本地行情是
后复权价且锚在 IPO，复权因子从 1.6（工行）到 104（平安银行）不等，
把 0.01 元加到后复权价上，对平安银行只滑了真实价的百分之一个 tick。
模拟撮合网关（`sim_gateway`）仍按 tick，因为它拿到的是实时真实价。

### 看回测结果时最容易搞反的两处

1. **`annual_return` 含底仓 beta**。判断策略有没有 alpha 只能看
   `t_annual` / `t_contribution` 这类拆出来的字段。看总收益会把
   「股票自己涨了」记成策略业绩。
2. **搜索类结果没有单一的「年化」**。7776 组里挑最好的一组，
   既是全域最大值又含 beta，拿它当代表值是误导。管理台上这类条目
   的年化显示为「—」，代表值用全域均值。

---

## 5. 生成信号

引擎不直接调用模型，它读**信号文件**。这个解耦是有意的：模型可以
用任何语言、任何频率生成信号，引擎只认文件。

```bash
python strategies/alstm_ppo_csi1000/generate_signal.py
```

产出 `strategies/<策略>/signals/target_latest.csv`（目标持仓）。

`SignalFileStrategy` 消费它，有两道保护：

- `max_signal_age_days`（默认 3）：信号太旧就不执行。
  **过期的判断不该驱动今天的交易。**
- `last_signal_key`：按文件修改时间去重，避免重复调仓。

---

## 6. 接上引擎

### 6.1 先跑模拟撮合，不接券商

```bash
python scripts/run_live.py --gateway sim --replay --replay-speed 60
```

`--replay` 是必须的：`SimGateway` 只撮合，**不产生行情**。
不加这个参数一根 bar 都不会来，进程活着但什么都不发生——
这是最容易误判成「跑起来了」的情形。

### 6.2 策略要登记在 config.yaml

```yaml
strategies:
  - name: MaCrossDemo
    class: qmtquant.strategy.ma_cross.MaCrossStrategy
    vt_symbols: ["000001.SZSE"]
    setting:
      fast_window: 5
      slow_window: 20
      position_ratio: 0.95
```

**没登记在这里的策略不会跑**，不管它在管理台上显示得多完整。
管理台的策略列表展示的是回测产物，和实盘登记是两回事。

### 6.3 接 miniQMT

前置条件：

1. QMT 客户端已启动并登录
2. `config.yaml` 填了 `gateway.qmt_path` 和 `account_id`
3. **`xtquant` 必须来自 QMT 客户端安装目录**，不能 `pip install`
   （PyPI 上的同名包不是券商发行的那个）

```bash
python scripts/run_live.py --gateway miniqmt --dry-run
```

`--dry-run` 启动后立即开启急停：跑行情、跑策略、走风控，
但**不会真的下单**。用它确认整条链路的信号和委托意图是对的，
再去掉这个参数。

---

## 7. 上真钱之前的检查清单

按顺序，每条都要有**具体证据**，不能凭印象：

- [ ] `python -m pytest tests/ -q` 全绿
- [ ] `python scripts/check_data.py` 无「数据停更」级错误
- [ ] 回测的 `cost_model` 与当前一致（管理台顶部若有「旧成本模型」
      红色横幅，说明那些数字是旧口径跑的，**不能与新结果比较**）
- [ ] 用 `--gateway sim --replay` 跑通，`state.db` 里有成交记录
- [ ] 用 `--gateway miniqmt --dry-run` 跑通，日志里有委托意图但无真实委托
- [ ] `config.yaml` 的 `risk` 段按你的实际承受能力设过，不是默认值
- [ ] 告警通道打开并**收到过测试消息**（见 §9）
- [ ] 想清楚急停怎么按（见 §8）

---

## 8. 出事的时候

### 急停

风控层的 `kill_switch` 一开，所有新委托直接拒掉。管理台上有按钮；
命令行可以直接杀进程（引擎捕获 Ctrl+C 会撤单 → 停策略 → 断网关）。

⚠️ Windows 上 `os.kill(pid, SIGTERM)` 等价于 `TerminateProcess`，
**不会触发优雅退出**。要用 `CTRL_BREAK_EVENT`（进程需以
`CREATE_NEW_PROCESS_GROUP` 启动）。管理台的服务停止按钮已经这么做了。

### 风控的硬约束

`config.yaml` 的 `risk` 段，全部是硬约束不是建议值：

| 参数 | 含义 |
|---|---|
| `max_order_value` | 单笔委托金额上限 |
| `max_position_ratio` | 单票市值占总资产上限 |
| `max_total_position_ratio` | 总仓位上限 |
| `daily_loss_limit_ratio` | 当日亏损达此比例后只平不开 |
| `drawdown_*` | 回撤三档：停开仓 / 强制减仓 / 全平 |

⚠️ **回撤档位收紧不是单调变好的。** 实测（突破策略·沪深300·2016-2026）：

| 一档/二档/清仓 | 最大回撤 | 总收益 |
|---|---|---|
| 8/11/15 | −22.67% | +95.38% |
| 6/9/12 | −18.35% | +22.91% ← 当前默认 |
| 5/8/11 | −32.42% | −22.12% ← **收得更紧反而更差** |

阈值低于策略常态波动时会被反复触发，在局部低点被迫卖出，
峰值重置后再吃一轮完整回撤。改这几个数之前务必重跑扫描。

---

## 9. 盘后

### 对账

```bash
python strategies/alstm_ppo_csi1000/reconcile.py --date 2026-09-08
```

对账的核心是 `missing`——**「我下了单，但券商侧查无此单」**。
委托丢了却没人知道是实盘最坏的失败模式之一。

> 这条检查曾经是死的：`load_intent()` 找 `.json` 而 `paper_trade.py`
> 写 `.csv`，恒返回空，`missing` 恒为空列表，对账永远报「一切正常」。
> 已修，并加了测试。如果哪天 `executions/` 下的文件格式又变了，
> 现在会**直接抛错**而不是静默通过。

### 告警

`config.yaml`：

```yaml
notify:
  enabled: true
  channel: wecom          # wecom / dingtalk
  webhook: "https://..."  # 属于 secret，勿提交
```

打开后，ERROR 及以上的日志会推到群机器人：网关断开、报单被拒、
行情停推、重连达上限进入只读、对账查不到资金。

只在实盘入口（`run_live.py`）生效。下载与训练脚本不推——
那时人就在跟前，推了反而会淹掉真正要紧的告警。

同一条消息 5 分钟内只发一次，每分钟最多 10 条。

**打开后一定要确认收到过消息。** 配好了收不到，和没配是一样的，
而且更糟——你以为有人在看着。

---

## 10. 已知的坑

| 坑 | 表现 | 怎么办 |
|---|---|---|
| `SimGateway` 不产生行情 | 进程活着，一根 bar 都不来 | `--replay` |
| 策略没登记在 `config.yaml` | 管理台显示正常，实盘不跑 | 检查 `strategies` 段 |
| `xtquant` 装错来源 | 连接失败或行为诡异 | 只从 QMT 安装目录取 |
| Windows 优雅退出 | 撤单逻辑没执行 | 用 `CTRL_BREAK_EVENT` |
| 控制台 GBK 编码 | 脚本崩在 `print` 上 | 脚本已加 `errors="replace"` |
| 回测结果是旧成本口径 | 数字偏悲观且不可比 | 看管理台顶部横幅，重跑 |
| `models/` 有产物但管理台不显示 | 页面少一块，无报错 | `unregistered_outputs()` 会报，测试会红 |

---

## 11. 相关文档

- [ARCHITECTURE.md](ARCHITECTURE.md) — 事件驱动架构与模块划分
- [HOWTO_STRATEGY.md](HOWTO_STRATEGY.md) — 怎么写一个新策略
- [API.md](API.md) — 核心对象与接口
- [IMPLEMENTATION.md](IMPLEMENTATION.md) — 实现细节与依赖安装
