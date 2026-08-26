# 开源量化框架选型调研

> 目标：Windows 本地、A 股、已有国金 AMT 账号 + miniQMT / 大 QMT，选一个能二次开发的开源底座。

## 0. GitHub 星标实况（2026-08-26 经 GitHub API 实测）

| Stars | 项目 | 许可证 | 最近提交 | 语言 | 能否 A股实盘 |
|------:|------|--------|---------|------|-------------|
| 47,944 | [microsoft/qlib](https://github.com/microsoft/qlib) | MIT | 2026-07-23 | Python | ❌ 只做研究 |
| 44,775 | [vnpy/vnpy](https://github.com/vnpy/vnpy) | MIT | 2026-08-10 | Python | ⚠ 期货强，miniQMT 需魔改 |
| 22,969 | [mementum/backtrader](https://github.com/mementum/backtrader) | GPL-3.0 | **2024-08-19** | Python | ❌ 已停更 2 年 |
| 22,243 | [akfamily/akshare](https://github.com/akfamily/akshare) | MIT | 2026-08-26 | Python | — 数据源 |
| 21,355 | [QuantConnect/Lean](https://github.com/QuantConnect/Lean) | Apache-2.0 | 2026-08-25 | C# | ❌ 不支持 A股券商 |
| 20,067 | [quantopian/zipline](https://github.com/quantopian/zipline) | Apache-2.0 | **2024-02-13** | Python | ❌ 项目已死 |
| 18,391 | [UFund-Me/Qbot](https://github.com/UFund-Me/Qbot) | MIT | 2026-03-11 | Notebook | ⚠ 偏演示 |
| 18,232 | [bbfamily/abu](https://github.com/bbfamily/abu) | GPL-3.0 | 2026-01-24 | Python | ❌ 无实盘通道 |
| 16,105 | [AI4Finance/FinRL](https://github.com/AI4Finance-Foundation/FinRL) | MIT | 2026-07-13 | Notebook | ❌ 强化学习研究 |
| 15,367 | [waditu/tushare](https://github.com/waditu/tushare) | BSD-3 | **2024-03-13** | Python | — 数据源 |
| 10,643 | [StockSharp](https://github.com/StockSharp/StockSharp) | 自定义 | 2026-08-25 | C# | ❌ 不含 A股 |
| 6,722 | [ricequant/rqalpha](https://github.com/ricequant/rqalpha) | 自定义 | 2026-08-24 | Python | ⚠ 绑定米筐生态 |
| 6,297 | [wondertrader](https://github.com/wondertrader/wondertrader) | MIT | 2025-09-30 | C++ | ⚠ 重心在 CTP 期货 |
| 3,468 | [fasiondog/hikyuu](https://github.com/fasiondog/hikyuu) | Apache-2.0 | 2026-08-26 | C++ | ❌ 回测研究为主 |
| 1,537 | [khscience/OSkhQuant](https://github.com/khscience/OSkhQuant) | CC BY-NC | 2026-04-18 | Python | ❌ 官方明确不含实盘 |

**关键观察：星数与「能否解决本项目的问题」几乎无关。**

- 星数前两名（qlib 4.8w、vnpy 4.5w）都不能直接用 miniQMT 下 A 股单
- 22.9k 的 backtrader 和 20k 的 zipline **都已停止维护两年以上**
- 唯一 miniQMT 原生的 OSkhQuant 只有 1.5k 星，且明确不含实盘

## 1. 候选框架对比

| 框架 | 语言 | 类型 | A股实盘 | QMT/miniQMT 对接 | 二开难度 | 结论 |
|------|------|------|---------|------------------|----------|------|
| **vn.py** | Python | 全栈（CTA/回测/实盘/GUI） | ✅ 成熟（期货/CTP） | ⚠ 见下方更正 | 中，架构清晰 | 期货强，A股 miniQMT 弱 |
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

## 2.5 更正（2026-08-26，查证后）

本文档初版凭印象写就，有一处关键错误，更正如下：

### 更正 1：vnpy_xt 不是 miniQMT 交易网关

初版称「社区有 `vnpy_xt`（miniQMT 网关）」。**这是错的。**

[vnpy/vnpy_xt](https://github.com/vnpy/vnpy_xt) 官方定位是**迅投研数据服务接口（datafeed）**，
用于获取历史量价数据。VeighNa 社区中资深用户的明确回复是：
vnpy_xt 支持的是「迅投研」数据服务，**并未支持 miniQMT 交易**。

包内虽然含 gateway 文件，但社区实践显示需要自行改造（注释掉 QMT 路径后缀等）
才能实盘，属于**非官方支持的自行魔改**，且框架升级后改动会丢失。
社区里「连接XT 报错：服务器端只支持用户模式」之类的问题至今存在。

**结论**：截至 2026-08，**不存在成熟且官方维护的开源 miniQMT 交易网关**。

### 更正 2：khQuant 只做回测，不含实盘

[khscience/OSkhQuant](https://github.com/khscience/OSkhQuant)（约 1.5k stars）是真实存在
且质量不错的 miniQMT 原生框架：PyQt5 图形界面、完整撮合引擎、DuckDB 数据源、
回测报告齐全，全部开源。

但有两个硬约束：

1. **明确不含实盘交易**。官方文档原话：核心功能是历史数据验证，
   官方版本不包含任何直接执行实盘交易的功能。改造实盘属自行承担。
2. **许可证是 CC BY-NC 4.0**（署名-非商业性使用）。这是内容许可证而非软件许可证，
   用于代码时权责边界模糊，且 NC 条款限制商业用途。

## 3. 最终选型

采取 **"借鉴 vn.py 架构 + 自建轻量内核"** 路线。

**查证后的理由**（原理由中关于 vnpy_xt 的部分已作废，见 2.5）

1. **交易网关层：没有可用的开源替代品。** 这是决定性因素。
   vnpy_xt 是数据接口不是交易网关，OSkhQuant 明确不含实盘。
   要用 miniQMT 实盘，网关这层无论如何都得自己写。
2. **回测层：有替代品（OSkhQuant），但迁移成本 > 收益。**
   本项目回测引擎已实现且有测试覆盖；换过去要重写全部策略，
   且受 CC BY-NC 许可证约束。
3. vn.py 体量大（GUI/期货/期权/CTP），A 股单市场场景 80% 用不上。

**自建的真实代价（诚实记录）**

miniQMT 的坑要自己一个个踩。已踩过的：

| 坑 | 后果 | 发现方式 |
|----|------|---------|
| 深市限价类型用了旧版 101 | **所有深市股票下不了单**，且失败无原因 | 实测报单返回 -1 |
| xtdata 时间戳时区 | 日线整体偏移一天，回测静默用错日期 | 检查首根 K 线落在元旦 |
| 板块缓存为空 | 沪深300 取到 0 只 | 实测 |
| 客户端未打包 xtquant | 按文档复制目录找不到文件 | 实测 |

这些在成熟框架里本该已经趟平。自建意味着这类问题会持续出现 ——
应对办法是**每个坑都补一条回归测试**，让它只坑一次。
本项目 `tests/test_miniqmt_gateway.py` 即为此而设，会校验常量映射与 SDK 一致，
SDK 升级后若枚举变动能立刻发现。

**但保留 vn.py 的核心设计**：事件驱动（EventEngine）、Gateway 抽象、`TickData/BarData/OrderData/TradeData` 对象模型 —— 这样将来若要迁回 vn.py 或接入 `vnpy_xt`，成本极低。

**分层复用开源**
- 因子/模型研究：Qlib
- 快速策略验证：本项目自带向量化回测；复杂场景导出到 backtrader
- 数据落地：SQLite（起步）→ ClickHouse / DuckDB（Tick 级）

## 4. 参考链接

- vn.py: https://github.com/vnpy/vnpy
- vnpy_xt（数据服务，非交易网关）: https://github.com/vnpy/vnpy_xt
- Qlib: https://github.com/microsoft/qlib
- OSkhQuant（miniQMT 原生回测，CC BY-NC）: https://github.com/khscience/OSkhQuant
- xtquant 官方文档: http://docs.thinktrader.net/
- VeighNa 社区（miniQMT 相关问题）: https://www.vnpy.com/forum/
