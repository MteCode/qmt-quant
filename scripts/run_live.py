"""实盘 / 模拟盘入口。

用法：
    # 用本地模拟撮合网关做全链路联调（不接券商，安全）
    python scripts/run_live.py --gateway sim

    # 接 miniQMT（先确认 config.yaml 已填账号，且客户端已登录）
    python scripts/run_live.py --gateway miniqmt

按 Ctrl+C 优雅退出：撤单 → 停策略 → 断开网关。
"""
import argparse
import importlib
import json
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qmtquant.config import LOG_DIR, get_config  # noqa: E402
from qmtquant.engine.live_engine import LiveEngine  # noqa: E402
from qmtquant.event.engine import EventEngine  # noqa: E402
from qmtquant.risk.risk_manager import RiskManager  # noqa: E402
from qmtquant.store.database import StateStore  # noqa: E402
from qmtquant.utils.logger import setup_logging  # noqa: E402


def build_gateway(name: str, event_engine: EventEngine, cfg):
    """按配置构造网关"""
    if name == "sim":
        from qmtquant.gateway.sim_gateway import SimGateway
        return SimGateway(event_engine, "SIM",
                          initial_capital=cfg.backtest.initial_capital, cost=cfg.cost)
    if name == "miniqmt":
        from qmtquant.gateway.miniqmt_gateway import MiniQmtGateway
        return MiniQmtGateway(event_engine)
    raise ValueError(f"未知网关: {name}（可选 sim / miniqmt）")


def load_strategy_class(dotted_path: str):
    """`a.b.module.ClassName` -> 类对象"""
    module_path, _, class_name = dotted_path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def resolve_symbols(item: dict) -> list[str]:
    """解析策略的标的列表。

    信号驱动的策略标的是变化的，在 config 里写死几百只既难维护、
    也会在信号更新后失配。`vt_symbols: from_signal` 表示从信号文件读取，
    并与当前持仓求并集 —— 持仓里有而信号里没有的票也要订阅行情，
    否则卖不掉（`rebalance` 拿不到 bar 就跳过）。
    """
    raw = item.get("vt_symbols", [])
    if raw != "from_signal" and not (
            isinstance(raw, list) and "from_signal" in raw):
        return list(raw)

    import csv as _csv

    syms: set[str] = set()
    sig = (item.get("setting") or {}).get("signal_file", "")
    if sig:
        p = Path(sig)
        if not p.is_absolute():
            p = ROOT / p
        if p.exists():
            try:
                with p.open(encoding="utf-8-sig") as f:
                    syms.update(r["vt_symbol"].strip()
                                for r in _csv.DictReader(f)
                                if r.get("vt_symbol"))
            except (OSError, KeyError, _csv.Error) as e:
                print(f"  [WARN] 读取信号标的失败: {e}")
        else:
            print(f"  [WARN] 信号文件不存在: {p}")

    # 并上当前持仓，否则要卖的票收不到行情
    pos_file = ROOT / "strategies" / "alstm_ppo_csi1000" / "state" / "positions.json"
    if pos_file.exists():
        try:
            snap = json.loads(pos_file.read_text(encoding="utf-8"))
            syms.update(h["vt_symbol"] for h in snap.get("holdings", [])
                        if h.get("vt_symbol"))
        except (OSError, ValueError, KeyError):
            pass

    # 显式列出的标的照样保留
    if isinstance(raw, list):
        syms.update(s for s in raw if s != "from_signal")
    return sorted(syms)


def main() -> int:
    parser = argparse.ArgumentParser(description="qmtquant 实盘引擎")
    parser.add_argument("--gateway", default=None, help="sim / miniqmt")
    parser.add_argument("--dry-run", action="store_true",
                        help="启动后立即开启急停，只跑行情不下单")
    parser.add_argument("--no-store", action="store_true",
                        help="不持久化状态（策略状态、成交流水不落库）")
    parser.add_argument("--replay", action="store_true",
                        help="回放本地历史行情驱动策略。SimGateway 不产生行情，"
                             "不开这个就一根 bar 都不会来")
    parser.add_argument("--replay-start", default=None, help="回放起始日")
    parser.add_argument("--replay-end", default=None, help="回放结束日")
    parser.add_argument("--replay-speed", type=float, default=0.0,
                        help="回放加速：0=尽快跑完，60=1分钟行情用1秒")
    parser.add_argument("--replay-interval", default="1m", help="1m / 1d")
    args = parser.parse_args()

    cfg = get_config()
    setup_logging(LOG_DIR, cfg.log_level)

    gateway_name = args.gateway or cfg.gateway.name
    if gateway_name == "miniqmt" and not cfg.gateway.account_id:
        print("config/config.yaml 中未配置 gateway.account_id，无法连接 miniQMT")
        return 1

    # ---- 装配
    event_engine = EventEngine()
    event_engine.start()

    gateway = build_gateway(gateway_name, event_engine, cfg)
    risk_manager = RiskManager(cfg.risk, event_engine)

    store = None
    if not args.no_store:
        from qmtquant.config import DATA_DIR
        store = StateStore(DATA_DIR / "state.db")
        print(f"状态库: {store.path}  {store.summary()}")

    engine = LiveEngine(event_engine, gateway, risk_manager, store=store)

    setting = {
        "qmt_path": cfg.gateway.qmt_path,
        "account_id": cfg.gateway.account_id,
        "account_type": cfg.gateway.account_type,
        "reconnect_max_retry": cfg.gateway.reconnect_max_retry,
        "reconnect_base_delay": cfg.gateway.reconnect_base_delay,
        "request_timeout": cfg.gateway.request_timeout,
    }
    if not gateway.connect(setting):
        print("网关连接失败，详见日志")
        event_engine.stop()
        return 1

    if args.dry_run:
        risk_manager.activate_kill_switch("dry-run 模式，只观察不下单")

    # ---- 加载策略
    if not cfg.strategies:
        print("config.yaml 中未配置任何策略（strategies 为空）")
    for item in cfg.strategies:
        try:
            engine.add_strategy(
                load_strategy_class(item["class"]),
                item["name"], resolve_symbols(item), item.get("setting", {}),
            )
        except Exception as e:
            print(f"加载策略 {item.get('name')} 失败: {e}")

    engine.reconcile()
    engine.init_all()
    engine.start_all()

    # ---- 行情回放（可选）
    feeder = None
    if args.replay:
        from qmtquant.datafeed.replay import BarReplayFeeder
        syms = sorted({s for st in engine.strategies.values()
                       for s in st.vt_symbols})
        feeder = BarReplayFeeder(
            event_engine, cfg.data.store_dir, syms,
            interval=args.replay_interval,
            start=args.replay_start, end=args.replay_end,
            speed=args.replay_speed)
        if feeder.load() > 0:
            feeder.start_replay()
        else:
            print("回放无数据，请检查 --replay-start/--replay-end 与本地行情")
            feeder = None
    print(f"引擎已启动（网关={gateway_name}"
          f"{'，dry-run' if args.dry_run else ''}），Ctrl+C 退出")

    # ---- 主循环
    running = True

    def _stop(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop)
    # 管理台停服务时：Windows 发 CTRL_BREAK_EVENT（表现为 SIGBREAK），
    # 其他平台发 SIGTERM。两者都要走优雅退出（撤单 → 停策略 → 断网关）——
    # Windows 上 os.kill(pid, SIGTERM) 实际是 TerminateProcess，
    # 会直接杀死进程，挂单留在券商那边、状态来不及落库。
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
            # 心跳只在事件引擎仍在处理事件时写。进程活着不等于在干活 ——
            # 卡在一次阻塞的 order_stock 上时进程状态正常，
            # 但事件循环早已冻住，只看进程会得到「一切正常」的假象。
            if tick % 30 == 0 and event_engine.is_active():
                services.beat("live")
    finally:
        print("\n正在退出：撤单 → 停策略 → 断开网关 ...")
        if feeder is not None:
            feeder.stop()
        engine.close()
        event_engine.stop()
        print("已安全退出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
