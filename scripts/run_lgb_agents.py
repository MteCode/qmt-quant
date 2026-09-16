"""LGB+Agents 四层选股实盘 —— 信号文件驱动，20 日调仓。

用法：
    # 预览模式（不实际下单）
    python scripts/run_lgb_agents.py --dry-run

    # 实盘执行
    python scripts/run_lgb_agents.py

    # 自定义参数
    python scripts/run_lgb_agents.py --capital 200000 --max-holdings 10

miniQMT 客户端必须已启动并登录。按 Ctrl+C 优雅退出。
"""
import argparse
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SIGNAL_FILE = ROOT / "strategies" / "lgb_agents_ppo" / "signals" / "target_latest.csv"


def load_symbols_from_signal() -> list[str]:
    """从信号文件 + 持仓文件加载标的池。"""
    import csv as _csv
    import json

    syms: set[str] = set()

    if SIGNAL_FILE.exists():
        try:
            with SIGNAL_FILE.open(encoding="utf-8-sig") as f:
                syms.update(
                    r["vt_symbol"].strip()
                    for r in _csv.DictReader(f)
                    if r.get("vt_symbol")
                )
        except (OSError, KeyError, _csv.Error) as e:
            print(f"  [WARN] 读取信号标的失败: {e}")
    else:
        print(f"  [WARN] 信号文件不存在: {SIGNAL_FILE}")

    pos_file = ROOT / "strategies" / "lgb_agents_ppo" / "state" / "positions.json"
    if pos_file.exists():
        try:
            snap = json.loads(pos_file.read_text(encoding="utf-8"))
            syms.update(
                h["vt_symbol"]
                for h in snap.get("holdings", [])
                if h.get("vt_symbol")
            )
        except (OSError, ValueError, KeyError):
            pass

    filtered = []
    for vt in sorted(syms):
        code = vt.split(".")[0]
        if code.startswith("688"):
            continue
        if code[:2] in ("43", "83", "87"):
            continue
        filtered.append(vt)

    return filtered


def main() -> int:
    parser = argparse.ArgumentParser(description="LGB+Agents 四层选股实盘")
    parser.add_argument("--dry-run", action="store_true",
                        help="启动后开启急停，只跑行情不下单")
    parser.add_argument("--capital", type=float, default=200000,
                        help="总资金（默认 200000）")
    parser.add_argument("--max-holdings", type=int, default=10,
                        help="最大持仓数（默认 10）")
    parser.add_argument("--rebalance-days", type=int, default=20,
                        help="调仓周期（默认 20 交易日）")
    args = parser.parse_args()

    from qmtquant.config import LOG_DIR, get_config
    from qmtquant.utils.logger import setup_logging

    cfg = get_config()
    setup_logging(LOG_DIR, cfg.log_level, cfg)

    if not cfg.gateway.account_id:
        print("config.yaml 中未配置 gateway.account_id")
        return 1

    print("=" * 50)
    print("LGB+Agents 四层选股实盘")
    print("=" * 50)
    print(f"  资金:     {args.capital:,.0f} 元")
    print(f"  最大持仓: {args.max_holdings} 只")
    print(f"  调仓周期: 每 {args.rebalance_days} 交易日")
    print(f"  信号文件: {SIGNAL_FILE.relative_to(ROOT)}")
    if args.dry_run:
        print("  ** DRY RUN — 只观察不下单 **")
    print()

    symbols = load_symbols_from_signal()
    print(f"  标的池: {len(symbols)} 只")
    if not symbols:
        print("  [WARN] 标的池为空，请先运行 generate_signal.py 生成信号")

    from qmtquant.event.engine import EventEngine
    from qmtquant.gateway.miniqmt_gateway import MiniQmtGateway
    from qmtquant.engine.live_engine import LiveEngine
    from qmtquant.risk.risk_manager import RiskManager

    event_engine = EventEngine()
    event_engine.start()

    gateway = MiniQmtGateway(event_engine)
    risk_manager = RiskManager(cfg.risk, event_engine)

    print(f"\n连接 miniQMT...")
    print(f"  路径: {cfg.gateway.qmt_path}")
    print(f"  账号: {cfg.gateway.account_id}")

    connected = gateway.connect({
        "qmt_path": cfg.gateway.qmt_path,
        "account_id": cfg.gateway.account_id,
        "account_type": cfg.gateway.account_type,
        "reconnect_max_retry": cfg.gateway.reconnect_max_retry,
        "reconnect_base_delay": cfg.gateway.reconnect_base_delay,
        "request_timeout": cfg.gateway.request_timeout,
    })
    if not connected:
        print("连接失败！请确认 miniQMT 已启动并登录")
        event_engine.stop()
        return 1
    print("  连接成功")

    engine = LiveEngine(event_engine, gateway, risk_manager)

    if args.dry_run:
        risk_manager.activate_kill_switch("dry-run 模式，只观察不下单")

    from qmtquant.strategy.signal_file import SignalFileStrategy

    setting = {
        "signal_file": str(SIGNAL_FILE),
        "max_signal_age_days": 3,
        "min_order_value": 5000,
        "capital": args.capital,
        "max_holdings": args.max_holdings,
        "rebalance_days": args.rebalance_days,
        "cash_buffer": 0.05,
        "price_buffer": 0.03,
    }

    engine.add_strategy(
        SignalFileStrategy,
        "LGB_AGENTS_PPO",
        symbols,
        setting,
    )

    engine.init_all()
    engine.start_all()

    print(f"\n引擎已启动（{'dry-run' if args.dry_run else '实盘'}），Ctrl+C 退出")
    print(f"  调仓周期: 每 {args.rebalance_days} 交易日")
    print(f"  信号新鲜度: 最多 3 天")

    running = True

    def _stop(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop)
    for _sig in ("SIGBREAK", "SIGTERM"):
        h = getattr(signal, _sig, None)
        if h is None:
            continue
        try:
            signal.signal(h, _stop)
        except (OSError, ValueError):
            pass

    from webui import services
    try:
        tick = 0
        while running:
            time.sleep(1)
            tick += 1
            if tick % 30 == 0 and event_engine.is_active():
                services.beat("live")
    finally:
        print("\n正在退出：撤单 → 停策略 → 断开网关 ...")
        engine.close()
        event_engine.stop()
        print("已安全退出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
