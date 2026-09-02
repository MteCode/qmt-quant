"""盘中风控守护 —— 持续跟踪回撤，触线自动减仓/清仓。

与回测 (BacktestEngine) 共用同一套 DrawdownController 判定逻辑，
配置也读同一份 config.yaml 的 risk 段，保证回测与实盘口径一致。

## 两条时间线（重要）

回撤档位控制器的 `min_observations` / `max_freeze_observations` 是按
**日线**扫描调出来的参数（见 drawdown.py 的实测表格）。若按秒级轮询
喂给它，120 个观测点会从「约 6 个月」退化成「1 小时」，参数语义失效。
因此：

- **档位控制器**：每个交易日只喂 1 个观测点（当日首次轮询时），
  与回测严格一致，驱动 6%/9%/12% 三档动作。
- **日内连续监控**：每次轮询都算实时回撤与当日盈亏，
  当日亏损触及 `daily_loss_limit_ratio`(3%) 立即进入只平不开。

## 状态持久化

回撤峰值必须跨重启保留 —— `DrawdownController.reset()` 的文档明确写了
「实盘不应调用，那等于抹掉回撤记忆」。状态存到本策略的 `state/risk_state.json`，
启动时加载，每次更新后落盘。每个策略各自一份，互不干扰。

用法::

    python strategies/alstm_ppo_csi1000/risk_monitor.py              # 守护（自动减仓）
    python strategies/alstm_ppo_csi1000/risk_monitor.py --dry-run    # 只告警不下单
    python strategies/alstm_ppo_csi1000/risk_monitor.py --interval 60
    python strategies/alstm_ppo_csi1000/risk_monitor.py --once       # 只检查一次
"""
import argparse
import json
import sys
import time
from datetime import date, datetime, time as dtime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

STATE_FILE = paths.RISK_STATE

MORNING = (dtime(9, 30), dtime(11, 30))
AFTERNOON = (dtime(13, 0), dtime(15, 0))


def in_trading_hours(now: datetime | None = None) -> bool:
    now = now or datetime.now()
    if now.weekday() >= 5:
        return False
    t = now.time()
    return (MORNING[0] <= t <= MORNING[1]) or (AFTERNOON[0] <= t <= AFTERNOON[1])


# ---------------------------------------------------------------- 状态持久化

def load_state(controller) -> str | None:
    """把上次的回撤状态灌回控制器，返回上次观测日期(YYYY-MM-DD)。"""
    if not STATE_FILE.exists():
        print("  无历史风控状态，以今日为起点")
        return None

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

    print(f"  已加载风控状态: 峰值 {s.peak:,.0f}, 回撤 {s.drawdown:.2%}, "
          f"档位 {s.level.label}, 观测 {s.observations} 天")
    return data.get("last_obs_date")


def save_state(controller, last_obs_date: str | None) -> None:
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
        "last_obs_date": last_obs_date,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- 账户查询

def query_equity(trader, account) -> tuple[float, float]:
    """返回 (总资产, 可用资金)。查询失败返回 (0, 0)。"""
    asset = trader.query_stock_asset(account)
    if not asset:
        return 0.0, 0.0
    return float(asset.total_asset), float(asset.cash)


def query_positions(trader, account) -> list[dict]:
    """当前持仓列表，只含 volume > 0 的。"""
    out = []
    for pos in trader.query_stock_positions(account) or []:
        if pos.volume <= 0:
            continue
        out.append({
            "code": pos.stock_code,          # xtquant 格式 600000.SH
            "volume": int(pos.volume),
            "available": int(pos.can_use_volume),
            "market_value": float(pos.market_value),
        })
    return out


# ---------------------------------------------------------------- 减仓执行

def sell_to_target(trader, account, positions, keep_ratio: float,
                   dry_run: bool) -> int:
    """把持仓削减到 keep_ratio。keep_ratio=0 即清仓。

    受 T+1 限制，只能卖 available（昨仓）部分。
    """
    action = "清仓" if keep_ratio <= 0 else f"减仓至 {keep_ratio:.0%}"
    print(f"\n  >>> 执行{action}（{len(positions)} 只持仓）")

    sent = 0
    blocked = 0
    for p in positions:
        target_vol = int(p["volume"] * keep_ratio)
        # 卖出数量向下取整到 100 股；清仓时允许卖零股
        raw_sell = p["volume"] - target_vol
        if keep_ratio <= 0:
            sell_vol = p["available"]
        else:
            sell_vol = min((raw_sell // 100) * 100, p["available"])

        if sell_vol <= 0:
            if p["available"] <= 0 and raw_sell > 0:
                blocked += 1
                print(f"      {p['code']:>12s}  需卖 {raw_sell:>6d} 股，"
                      f"但可卖 0（T+1 冻结）")
            continue

        print(f"      {p['code']:>12s}  卖出 {sell_vol:>6d} / 持有 {p['volume']:>6d}")
        sent += 1
        if not dry_run:
            trader.order_stock(
                account, p["code"], 24,      # STOCK_SELL
                int(sell_vol), 5, 0,         # price_type=5 最优五档, price=0
                strategy_name="RISK_MONITOR",
                order_remark=f"risk_{action}",
            )
            time.sleep(0.2)

    print(f"  >>> 已发出 {sent} 笔卖单" + (f"，{blocked} 只被 T+1 冻结" if blocked else ""))
    if dry_run and sent:
        print("      (DRY RUN，未实际下单)")
    return sent


# ---------------------------------------------------------------- 主循环

def main() -> int:
    p = argparse.ArgumentParser(description="盘中风控守护")
    p.add_argument("--dry-run", action="store_true",
                   help="只监控告警，触线不实际下单")
    p.add_argument("--interval", type=int, default=30,
                   help="轮询间隔秒数，默认 30")
    p.add_argument("--once", action="store_true",
                   help="只检查一次就退出（用于测试）")
    p.add_argument("--ignore-hours", action="store_true",
                   help="忽略交易时段判断（非交易时段测试用）")
    args = p.parse_args()

    from qmtquant.config import get_config
    from qmtquant.event.engine import EventEngine
    from qmtquant.gateway.xt_gateway import XtGateway
    from qmtquant.risk.drawdown import DrawdownConfig, DrawdownController, DrawdownLevel
    from xtquant.xttype import StockAccount

    cfg = get_config()
    r = cfg.risk

    print("=" * 56)
    print("  盘中风控守护  ALSTM + PPO")
    print(f"  模式: {'DRY RUN（不下单）' if args.dry_run else '实盘（会自动减仓）'}")
    print(f"  轮询: 每 {args.interval} 秒")
    print("=" * 56)
    print(f"\n  档位阈值（与回测一致）:")
    print(f"    一档 停止开仓 : 回撤 {r.drawdown_close_only:.0%}")
    print(f"    二档 强制减仓 : 回撤 {r.drawdown_reduce:.0%} → 保留 {r.drawdown_reduce_keep:.0%}")
    print(f"    三档 全部清仓 : 回撤 {r.drawdown_flat:.0%}")
    print(f"    日亏损线     : {r.daily_loss_limit_ratio:.0%} → 只平不开")

    controller = DrawdownController(DrawdownConfig(
        enabled=r.drawdown_enabled,
        close_only_threshold=r.drawdown_close_only,
        reduce_threshold=r.drawdown_reduce,
        reduce_keep_ratio=r.drawdown_reduce_keep,
        flat_threshold=r.drawdown_flat,
        recovery_ratio=r.drawdown_recovery_ratio,
        min_observations=r.drawdown_min_observations,
        max_freeze_observations=r.drawdown_max_freeze,
    ))

    print()
    last_obs_date = load_state(controller)

    # 连接 miniQMT
    print(f"\n  连接 miniQMT ({cfg.gateway.account_id}) ...")
    event_engine = EventEngine()
    event_engine.start()
    gateway = XtGateway(event_engine)
    if not gateway.connect({"qmt_path": cfg.gateway.qmt_path,
                            "account_id": cfg.gateway.account_id}):
        print("  连接失败，请确认 miniQMT 已启动并登录")
        event_engine.stop()
        return 1
    print("  连接成功")

    trader = gateway.trader
    account = StockAccount(cfg.gateway.account_id)

    day_start_equity = 0.0       # 当日起始总资产，算日内盈亏
    close_only = False           # 日亏触线标志
    acted_level = controller.level   # 已执行过动作的档位，避免重复下单
    tick = 0

    try:
        while True:
            tick += 1
            now = datetime.now()
            today = now.strftime("%Y-%m-%d")

            if not args.ignore_hours and not in_trading_hours(now):
                if args.once:
                    print(f"\n  [{now:%H:%M:%S}] 非交易时段，退出"
                          f"（测试可加 --ignore-hours）")
                    break
                print(f"\r  [{now:%H:%M:%S}] 非交易时段，等待中 ...  ",
                      end="", flush=True)
                time.sleep(min(args.interval, 60))
                continue

            equity, cash = query_equity(trader, account)
            if equity <= 0:
                print(f"\n  [{now:%H:%M:%S}] 查询账户失败，跳过本次")
                time.sleep(args.interval)
                continue

            # --- 当日基准（每天首次轮询时确立）
            if last_obs_date != today:
                day_start_equity = equity
                close_only = False
                # 档位控制器每天只喂 1 个观测点，与回测日线口径一致
                new_level = controller.update(equity)
                last_obs_date = today
                save_state(controller, last_obs_date)
                print(f"\n  [{now:%H:%M:%S}] === 新交易日 {today} ===")
                print(f"      日初总资产 {equity:,.0f}，"
                      f"峰值 {controller.state.peak:,.0f}，"
                      f"回撤 {controller.drawdown:.2%}，"
                      f"档位 {new_level.label}")
            elif not day_start_equity:
                day_start_equity = equity

            # --- 日内连续指标（不喂控制器，只用于告警与日亏线）
            peak = max(controller.state.peak, equity)
            live_dd = 1 - equity / peak if peak > 0 else 0.0
            day_pnl = (equity - day_start_equity) / day_start_equity if day_start_equity else 0.0

            level = controller.level
            positions = query_positions(trader, account)
            pos_value = sum(p["market_value"] for p in positions)

            print(f"\r  [{now:%H:%M:%S}] 资产 {equity:>10,.0f} | "
                  f"持仓 {len(positions):>2d} 只 {pos_value:>9,.0f} | "
                  f"日内 {day_pnl:>+6.2%} | 回撤 {live_dd:>6.2%} | "
                  f"{level.label}   ", end="", flush=True)

            # --- 日亏损线：日内连续判断
            if (not close_only
                    and day_pnl <= -r.daily_loss_limit_ratio):
                close_only = True
                print(f"\n  [{now:%H:%M:%S}] !! 当日亏损 {day_pnl:.2%} "
                      f"触及 {r.daily_loss_limit_ratio:.0%}，进入只平不开")

            # --- 档位动作：档位升高时执行一次
            if level > acted_level:
                print(f"\n  [{now:%H:%M:%S}] !! 回撤档位升至【{level.label}】"
                      f"（回撤 {controller.drawdown:.2%}）")
                if level == DrawdownLevel.FLAT and positions:
                    sell_to_target(trader, account, positions, 0.0, args.dry_run)
                elif level == DrawdownLevel.REDUCE and positions:
                    sell_to_target(trader, account, positions,
                                   r.drawdown_reduce_keep, args.dry_run)
                elif level == DrawdownLevel.CLOSE_ONLY:
                    print("      停止开新仓，存量持仓不动")
                acted_level = level
            elif level < acted_level:
                print(f"\n  [{now:%H:%M:%S}] 回撤档位降至【{level.label}】，恢复交易")
                acted_level = level

            if args.once:
                print("\n\n  单次检查完成")
                break

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\n  收到中断，正在退出 ...")
    finally:
        save_state(controller, last_obs_date)
        print(f"\n{controller.summary()}")
        print(f"\n  风控状态已保存: {STATE_FILE}")
        gateway.close()
        event_engine.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
