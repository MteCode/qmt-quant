"""日内高频剥头皮实盘 —— Tick 级 VWAP 回归 + 动量突破。

用法：
    python scripts/run_intraday_hft.py --dry-run
    python scripts/run_intraday_hft.py --symbols 600519.SSE 000858.SZSE
"""
import argparse
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="日内高频剥头皮实盘")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--symbols", nargs="+",
                        default=["600519.SSE"],
                        help="交易标的（默认 600519.SSE）")
    args = parser.parse_args()

    from qmtquant.config import LOG_DIR, get_config
    from qmtquant.utils.logger import setup_logging

    cfg = get_config()
    setup_logging(LOG_DIR, cfg.log_level, cfg)

    from qmtquant.event.engine import EventEngine
    from qmtquant.gateway.miniqmt_gateway import MiniQmtGateway
    from qmtquant.engine.live_engine import LiveEngine
    from qmtquant.risk.risk_manager import RiskManager

    event_engine = EventEngine()
    event_engine.start()

    gateway = MiniQmtGateway(event_engine)
    risk_manager = RiskManager(cfg.risk, event_engine)

    # 必须在 connect 之前 —— connect 内部 query_account 会发出 EVENT_ACCOUNT，
    # 而注册该处理器的正是 LiveEngine。顺序反了账户快照就丢了，
    # risk_manager.account 永远是 None，每笔买单都被判「可用资金不足」
    engine = LiveEngine(event_engine, gateway, risk_manager)

    print(f"连接 miniQMT... 账号: {cfg.gateway.account_id}")
    connected = gateway.connect({
        "qmt_path": cfg.gateway.qmt_path,
        "account_id": cfg.gateway.account_id,
        "account_type": cfg.gateway.account_type,
        "reconnect_max_retry": cfg.gateway.reconnect_max_retry,
        "reconnect_base_delay": cfg.gateway.reconnect_base_delay,
        "request_timeout": cfg.gateway.request_timeout,
    })
    if not connected:
        print("连接失败！")
        event_engine.stop()
        return 1

    if args.dry_run:
        risk_manager.activate_kill_switch("dry-run")

    from strategies.intraday_hft.strategy import IntradayHftStrategy
    engine.add_strategy(IntradayHftStrategy, "INTRADAY_HFT", args.symbols, {})
    engine.init_all()
    engine.start_all()

    print(f"日内高频已启动（{'dry-run' if args.dry_run else '实盘'}），"
          f"标的: {args.symbols}，Ctrl+C 退出")

    running = True
    def _stop(signum, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, _stop)
    for _sig in ("SIGBREAK", "SIGTERM"):
        h = getattr(signal, _sig, None)
        if h:
            try: signal.signal(h, _stop)
            except (OSError, ValueError): pass

    from webui import services
    try:
        tick = 0
        while running:
            time.sleep(1)
            tick += 1
            if tick % 30 == 0 and event_engine.is_active():
                services.beat("live")
    finally:
        print("\n退出中...")
        engine.close()
        event_engine.stop()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
