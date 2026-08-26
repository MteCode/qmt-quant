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
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qmtquant.config import LOG_DIR, get_config  # noqa: E402
from qmtquant.engine.live_engine import LiveEngine  # noqa: E402
from qmtquant.event.engine import EventEngine  # noqa: E402
from qmtquant.risk.risk_manager import RiskManager  # noqa: E402
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


def main() -> int:
    parser = argparse.ArgumentParser(description="qmtquant 实盘引擎")
    parser.add_argument("--gateway", default=None, help="sim / miniqmt")
    parser.add_argument("--dry-run", action="store_true",
                        help="启动后立即开启急停，只跑行情不下单")
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
    engine = LiveEngine(event_engine, gateway, risk_manager)

    setting = {
        "qmt_path": cfg.gateway.qmt_path,
        "account_id": cfg.gateway.account_id,
        "account_type": cfg.gateway.account_type,
        "reconnect_max_retry": cfg.gateway.reconnect_max_retry,
        "reconnect_base_delay": cfg.gateway.reconnect_base_delay,
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
                item["name"], item["vt_symbols"], item.get("setting", {}),
            )
        except Exception as e:
            print(f"加载策略 {item.get('name')} 失败: {e}")

    engine.reconcile()
    engine.init_all()
    engine.start_all()
    print(f"引擎已启动（网关={gateway_name}"
          f"{'，dry-run' if args.dry_run else ''}），Ctrl+C 退出")

    # ---- 主循环
    running = True

    def _stop(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop)
    try:
        while running:
            time.sleep(1)
    finally:
        print("\n正在退出：撤单 → 停策略 → 断开网关 ...")
        engine.close()
        event_engine.stop()
        print("已安全退出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
