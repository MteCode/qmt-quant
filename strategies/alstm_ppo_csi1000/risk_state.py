"""回撤状态的持久化 —— 盘中监控与下单脚本共用同一份。

## 为什么要共用

`DrawdownController` 的状态（峰值、档位、观测点数）必须跨进程存活：

- 峰值是从历史最高净值算起的，进程重启后从零开始等于抹掉回撤记忆
- `min_observations=20` 要求积累 20 个观测点才生效，每次新建控制器都会
  让这个计数归零，档位判定**永远不会触发**
- 盘中监控与下单脚本若各持一份状态，会出现「监控说只平不开、下单脚本
  照样开新仓」的分裂

因此两边都从这里读写同一个 JSON。

## 观测点的时间尺度

档位控制器每天只喂 **1 个**观测点。`min_observations=20` 与
`max_freeze_observations=120` 是按日线扫出来的参数（见 drawdown.py 的实测
表格），秒级喂会让「120 天冻结期」变成「1 小时」，保护直接失效。

日内的连续保护由 3% 日亏线负责，那是另一个时间尺度的指标。
"""
import json
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

STATE_FILE = paths.RISK_STATE


def load_state(controller, verbose: bool = True) -> dict:
    """把上次的回撤状态灌回控制器。

    :return: 原始状态字典（含 last_obs_date、day_start_balance 等）
    """
    if not STATE_FILE.exists():
        if verbose:
            print("  无历史风控状态，以本次为起点")
        return {}

    from qmtquant.risk.drawdown import DrawdownLevel

    data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    s = controller.state
    s.peak = data.get("peak", 0.0)
    s.current = data.get("current", 0.0)
    s.drawdown = data.get("drawdown", 0.0)
    s.level = DrawdownLevel(data.get("level", 0))
    s.observations = data.get("observations", 0)
    s.observations_at_level = data.get("observations_at_level", 0)
    s.peak_resets = data.get("peak_resets", 0)

    if verbose:
        print(f"  已加载风控状态: 峰值 {s.peak:,.0f}, 回撤 {s.drawdown:.2%}, "
              f"档位 {s.level.label}, 观测 {s.observations} 天")
    return data


def save_state(controller, last_obs_date: str | None = None,
               day_start_balance: float | None = None,
               day_start_date: str | None = None) -> None:
    """落盘。未提供的字段沿用文件里的旧值，避免互相覆盖。"""
    old = {}
    if STATE_FILE.exists():
        try:
            old = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            old = {}

    s = controller.state
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps({
        "peak": s.peak,
        "current": s.current,
        "drawdown": s.drawdown,
        "level": int(s.level),
        "observations": s.observations,
        "observations_at_level": s.observations_at_level,
        "peak_resets": s.peak_resets,
        "last_obs_date": (last_obs_date if last_obs_date is not None
                          else old.get("last_obs_date")),
        "day_start_balance": (day_start_balance
                              if day_start_balance is not None
                              else old.get("day_start_balance")),
        "day_start_date": (day_start_date if day_start_date is not None
                           else old.get("day_start_date")),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_day_start(state: dict, current_balance: float) -> float:
    """取当日起始总资产，用于 3% 日亏线。

    状态里记的若不是今天的，说明是新的一天，以当前资产为今日起点。
    这一步必须做：`RiskManager` 每次新建时会把 `_day_start_balance` 设成
    当前资产，于是在 14:00 启动的脚本会认为「今天还没亏」，
    哪怕早盘已经跌了 2.5%。
    """
    today = date.today().isoformat()
    if state.get("day_start_date") == today:
        saved = state.get("day_start_balance")
        if saved:
            return float(saved)
    return current_balance
