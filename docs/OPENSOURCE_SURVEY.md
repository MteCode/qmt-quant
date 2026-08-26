# 开源量化框架选型调研

> 目标：Windows 本地、A 股、已有国金 AMT 账号 + miniQMT / 大 QMT，选一个能二次开发的开源底座。

## 1. 候选框架对比

| 框架 | 语言 | 类型 | A股实盘 | QMT/miniQMT 对接 | 二开难度 | 结论 |
|------|------|------|---------|------------------|----------|------|
| **vn.py** | Python | 全栈（CTA/回测/实盘/GUI） | ✅ 成熟 | 社区有 `vnpy_xt`（miniQMT 网关） | 中，架构清晰 | **强烈推荐做底座** |
| **Qlib**（微软） | Python | AI 因子研究 / 模型训练 | ❌ 无实盘 | 无 | 中 | 推荐做**因子/选股研究层** |
| **backtrader** | Python | 回测为主 | 弱 | 无 | 低 | 只适合快速验证策略逻辑 |
| **RQAlpha** | Python | 回测 + 简单实盘 | 一般 | 无 | 中 | 米筐生态绑定重，不推荐 |
| **Hikyuu** | C++/Python | 高性能回测 | ❌ | 无 | 高 | 需要极致回测速度时再考虑 |
| **WonderTrader** | C++/Python | 全栈、低延迟 | ✅ | 无（重心在期货 CTP） | 高 | 股票场景性价比低 |
| **khQuant** | Python | 专为 miniQMT 封装 | ✅ | 原生 | 低 | 可作为**对接参考实现** |
| **easytrader** | Python | 券商客户端自动化 | 模拟点击 | 无 | 低 | 不稳定，不建议生产用 |

## 2. 账号 / 通道能力对比

| 通道 | 行情 | 交易 | 部署形态 | 适用 |
|------|------|------|----------|------|
| **miniQMT**（`xtquant`：`xtdata` + `xttrader`） | L1/L2 全推、历史 K 线、Tick | 股票/两融/ETF 下单 | 极简客户端 + Python 进程 | **首选：策略跑在自己的 Python 进程里** |
| **大 QMT** | 同上 | 同上 | 完整客户端，内置 Python 策略编辑器 | 图形化调试、可视化下单 |
| **国金 AMT** | 平台内行情 | 平台托管执行 | 云端/服务端托管 | 算法母单拆单、T0 执行、多账户 |

**结论**：以 **miniQMT 作为主通道**（行情 + 交易），AMT 作为**备用/大单算法执行通道**，大 QMT 用于人工盯盘与应急手工干预。

## 3. 最终选型

采取 **"借鉴 vn.py 架构 + 自建轻量内核"** 路线，而非直接 fork vn.py：

**理由**
1. vn.py 体量大（GUI/期货/期权/CTP 一大堆），A 股单市场场景 80% 用不上，维护负担重。
2. `vnpy_xt` 网关属社区维护，版本跟随 xtquant 更新有滞后，出问题时仍需自己改。
3. 本项目内核只需 4 层：事件引擎 / 网关抽象 / 策略引擎 / 风控，代码量 2k 行以内可控。

**但保留 vn.py 的核心设计**：事件驱动（EventEngine）、Gateway 抽象、`TickData/BarData/OrderData/TradeData` 对象模型 —— 这样将来若要迁回 vn.py 或接入 `vnpy_xt`，成本极低。

**分层复用开源**
- 因子/模型研究：Qlib
- 快速策略验证：本项目自带向量化回测；复杂场景导出到 backtrader
- 数据落地：SQLite（起步）→ ClickHouse / DuckDB（Tick 级）

## 4. 参考链接

- vn.py: https://github.com/vnpy/vnpy
- vnpy_xt: https://github.com/vnpy/vnpy_xt
- Qlib: https://github.com/microsoft/qlib
- khQuant: https://github.com/KHQuant/khQuant
- xtquant 官方文档: http://docs.thinktrader.net/
