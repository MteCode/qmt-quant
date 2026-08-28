"""配置加载。

禁止在代码里硬编码可调参数，一律走 config.yaml。
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "config"
DATA_DIR = ROOT_DIR / "data"
LOG_DIR = ROOT_DIR / "logs"


@dataclass
class CostConfig:
    """交易成本模型（A 股）"""
    commission_rate: float = 0.00025    # 佣金万 2.5
    commission_min: float = 5.0         # 单笔最低 5 元
    stamp_tax_rate: float = 0.001       # 印花税千 1，仅卖出
    transfer_fee_rate: float = 0.00001  # 过户费万 0.1，双向
    slippage_tick: int = 1              # 滑点，单位为最小变动价位


@dataclass
class RiskConfig:
    """风控参数，全部为硬约束"""
    max_order_value: float = 50_000          # 单笔委托金额上限
    max_position_ratio: float = 0.20         # 单票市值占总资产上限
    max_total_position_ratio: float = 0.95   # 总仓位上限
    max_order_count_per_day: int = 200       # 当日下单笔数上限
    max_turnover_per_day: float = 1_000_000  # 当日成交金额上限
    daily_loss_limit_ratio: float = 0.03     # 当日亏损达 3% 则只平不开
    forbid_st: bool = True
    blacklist: list[str] = field(default_factory=list)

    # --- 回撤控制（从峰值算起的累计跌幅，覆盖「连续阴跌」盲区）
    #
    # 目标：把实际最大回撤压在 20% 以内（硬约束）。
    #
    # 三档不能直接照着 20% 设 —— 信号在收盘产生、次日开盘才成交，
    # 加上 T+1 当日买入不可卖，从触发到出清有滞后。
    # 清仓线设 12% 才能让实际回撤落在 18% 左右。
    #
    # ⚠ **收紧档位不是单调变好的。** 实测（突破策略 · 沪深300 · 2016-2026）：
    #
    #   一档/二档/清仓    最大回撤    总收益
    #      8/11/15       -22.67%    +95.38%
    #      6/ 9/12       -18.35%    +22.91%   ← 当前默认
    #      5/ 8/11       -32.42%    -22.12%   ← 收得更紧反而更差
    #
    # 阈值低于策略常态波动时会被反复触发，在局部低点被迫卖出，
    # 峰值重置后再吃一轮完整回撤。改这几个数之前务必重跑扫描。
    drawdown_enabled: bool = True
    drawdown_close_only: float = 0.06    # 一档：停止开新仓
    drawdown_reduce: float = 0.09        # 二档：强制减仓
    drawdown_reduce_keep: float = 0.3    # 二档保留的仓位比例
    drawdown_flat: float = 0.12          # 三档：全部平仓
    drawdown_recovery_ratio: float = 0.7  # 降档迟滞系数
    drawdown_min_observations: int = 20
    # 最长冻结期（观测点数），0=禁用。超时后重置峰值并恢复交易，
    # 避免单次深回撤把策略永久锁死。
    # 实测设 0（不解冻）时策略会停摆，成交从 3742 掉到 1184，
    # 回撤反而升到 -21.55%；120 是扫描出来的可行值
    drawdown_max_freeze: int = 120


@dataclass
class GatewayConfig:
    name: str = "sim"                        # sim / miniqmt / amt
    qmt_path: str = ""                       # 大 QMT/miniQMT 的 userdata_mini 路径
    account_id: str = ""
    account_type: str = "STOCK"
    reconnect_max_retry: int = 10
    reconnect_base_delay: float = 2.0        # 指数退避基数（秒）


@dataclass
class DataConfig:
    provider: str = "xt"                     # xt / csv
    store_dir: str = str(DATA_DIR)
    # 必须 back：前复权对高分红股会算出负价格（601919 最低 -5.14）
    dividend_type: str = "back"              # back / front / none
    default_interval: str = "1d"


@dataclass
class BacktestConfig:
    start: str = "2020-01-01"
    end: str = "2024-12-31"
    initial_capital: float = 1_000_000
    interval: str = "1d"
    benchmark: str = "000300.SSE"


@dataclass
class TushareConfig:
    """Tushare Pro 数据源。

    token 是付费凭证，**绝不能进 git**。优先从环境变量 ``TUSHARE_TOKEN``
    读取，其次才是 config.yaml（该文件已在 .gitignore 中）。
    """
    #: 留空则回退到环境变量 TUSHARE_TOKEN
    token: str = ""
    #: 每分钟调用上限。2000 积分档官方限制远高于此，
    #: 设保守值是为了避免批量下载时触发风控被临时封禁
    calls_per_minute: int = 200
    #: 单次请求失败后的重试次数。网络抖动和偶发限流都靠它兜住
    max_retry: int = 3
    #: 重试退避基数（秒），实际等待为 base * 2**(n-1)
    retry_base_delay: float = 2.0


@dataclass
class NotifyConfig:
    enabled: bool = False
    channel: str = "wecom"                   # wecom / dingtalk
    webhook: str = ""


@dataclass
class AppConfig:
    log_level: str = "INFO"
    gateway: GatewayConfig = field(default_factory=GatewayConfig)
    data: DataConfig = field(default_factory=DataConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    tushare: TushareConfig = field(default_factory=TushareConfig)
    notify: NotifyConfig = field(default_factory=NotifyConfig)
    strategies: list[dict[str, Any]] = field(default_factory=list)


def _fill(cls, data: dict | None):
    """用 dict 填充 dataclass，未知字段忽略，缺失字段用默认值"""
    if not data:
        return cls()
    valid = {f.name for f in cls.__dataclass_fields__.values()}
    return cls(**{k: v for k, v in data.items() if k in valid})


def load_config(path: str | Path | None = None) -> AppConfig:
    """加载配置文件；不存在时回退到 config.example.yaml，再回退到默认值"""
    if path is None:
        path = CONFIG_DIR / "config.yaml"
        if not Path(path).exists():
            path = CONFIG_DIR / "config.example.yaml"

    path = Path(path)
    raw: dict = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    return AppConfig(
        log_level=raw.get("log_level", "INFO"),
        gateway=_fill(GatewayConfig, raw.get("gateway")),
        data=_fill(DataConfig, raw.get("data")),
        cost=_fill(CostConfig, raw.get("cost")),
        risk=_fill(RiskConfig, raw.get("risk")),
        backtest=_fill(BacktestConfig, raw.get("backtest")),
        tushare=_fill(TushareConfig, raw.get("tushare")),
        notify=_fill(NotifyConfig, raw.get("notify")),
        strategies=raw.get("strategies", []),
    )


#: 全局配置单例，首次访问时惰性加载
_config: AppConfig | None = None


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(cfg: AppConfig) -> None:
    global _config
    _config = cfg
