# qmtquant

> 基于 miniQMT / QMT 的 A 股个人量化交易系统 —— 事件驱动内核、回测实盘同构、前置硬风控。

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Windows-lightgrey)](#环境要求)

## 这是什么

把"选股逻辑 → 回测验证 → 自动下单 → 风控 → 复盘"整条链路自动化。
借鉴 vn.py 的事件驱动与网关抽象设计，但只保留 A 股场景需要的部分，代码量可控、便于二次开发。

**核心特性**

- **回测实盘同构** —— 同一份策略代码，切换 engine 即可在历史回测与真实账户间迁移，不改一行
- **严守 A 股规则** —— T+1 冻结、100 股整数倍、涨跌停不成交、停牌跳过
- **无前视偏差** —— 信号只用 T 日及之前数据，成交强制发生在 T+1 开盘
- **风控不可绕过** —— 所有下单必经 `RiskManager`，策略无法直连网关；含全局急停
- **网关可插拔** —— miniQMT / 国金 AMT / 本地模拟撮合，策略层无感知

## 快速开始

```bash
git clone git@github.com:MteCode/qmt-quant.git
cd qmt-quant
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
```

无需 QMT 环境，先用随机行情验证框架链路：

```bash
python scripts/run_backtest.py --mock
```

> 随机行情跑出的任何绩效都**没有参考意义**，只用来确认框架能跑通。

接入真实数据（需 QMT 客户端已登录，且 `xtquant` 可导入）：

```bash
python scripts/download_data.py --symbols 000001.SZ,600519.SH --start 2021-01-01
```

```bash
python scripts/run_backtest.py --symbols 000001.SZ --start 2021-01-01 --end 2024-12-31
```

跑测试：

```bash
pytest tests/ -q
```

## 配置

复制模板后填入真实值：

```bash
cp config/config.example.yaml config/config.yaml
```

> `config/config.yaml` 已在 `.gitignore` 中。**资金账号、webhook token 绝不要提交到仓库。**

关键项：

| 配置 | 说明 |
|------|------|
| `gateway.name` | `sim` 模拟撮合 / `miniqmt` 券商通道 / `amt` 国金算法通道 |
| `gateway.qmt_path` | miniQMT 的 `userdata_mini` 目录，注意不是安装根目录 |
| `cost.*` | 佣金/印花税/过户费/滑点，按你的实际费率改 |
| `risk.*` | 风控硬约束。**首次实盘务必把额度调到很小** |

## 项目结构

```
qmtquant/
├── core/       领域对象与枚举（TickData/BarData/OrderData/...）
├── event/      事件引擎，所有跨模块通信的总线
├── datafeed/   数据源：xtdata 下载落 Parquet / CSV 离线
├── gateway/    交易网关：BaseGateway 抽象 + miniQMT + 模拟撮合
├── risk/       前置风控与全局急停
├── engine/     回测引擎、实盘引擎、绩效计算
├── strategy/   策略基类与示例
├── store/      持久化
└── utils/      日志、代码转换、通知
```

## 写一个策略

```python
from qmtquant.strategy.base import StrategyBase

class MyStrategy(StrategyBase):
    parameters = ["window"]
    window = 20

    def on_bar(self, bar):
        if self.get_pos(bar.vt_symbol) == 0 and self.should_buy(bar):
            cash = self.get_cash() * 0.95
            volume = int(cash / bar.close_price // 100) * 100
            self.buy(bar.vt_symbol, bar.close_price * 1.02, volume)
```

完整示例见 [`qmtquant/strategy/examples/ma_cross.py`](qmtquant/strategy/examples/ma_cross.py)。

## 文档

| 文档 | 内容 |
|------|------|
| [架构设计](docs/ARCHITECTURE.md) | 分层设计、模块职责、关键流程、技术选型 |
| [需求文档](docs/REQUIREMENTS.md) | 功能需求清单、优先级、风险与里程碑 |
| [实现文档](docs/IMPLEMENTATION.md) | 环境搭建、接入步骤、上线检查清单 |
| [选型调研](docs/OPENSOURCE_SURVEY.md) | vn.py / Qlib / khQuant 等框架对比 |

## 环境要求

- Windows 10/11 x64（miniQMT 仅支持 Windows）
- Python 3.11
- miniQMT 或大 QMT 客户端，需常驻登录
- `xtquant`：**不要用 pip 装**，从 QMT 客户端安装目录 `bin.x64\Lib\site-packages` 复制，版本必须与客户端一致

## 开发路线

- [x] v0.1 事件驱动内核 + 模拟撮合 + 回测引擎 + 风控
- [ ] v0.2 miniQMT 网关实盘联调、实盘引擎
- [ ] v0.3 监控告警、定时调度、盘后报告
- [ ] v0.4 多策略组合与资金分配、Qlib 因子接入
- [ ] v0.5 AMT 算法执行通道、Tick 级回测、Web 监控面板

## 风险提示

本项目为个人学习与研究用途的交易框架。

- **仓库内的示例策略仅用于验证框架链路，不构成任何投资建议。**
- 回测绩效不代表未来收益，实盘存在滑点、冲击成本、成交不确定性等回测无法完全刻画的因素。
- 自动化交易可能因程序缺陷、网络中断、券商接口变更造成实际损失。
- 上线前请务必：先用模拟账户跑满一个月 → 再用小资金实盘验证 → 逐步放大额度。
- 使用本项目产生的一切后果由使用者自行承担。
